//! Exact shared-cell effects of one transactional workspace completion.
//!
//! Dynamic evaluation executes in a transactional child frame, so its stores
//! do not run the ordinary lexical publication hook. Retain only affected
//! compiler-owned bindings here, then apply them after the selected target and
//! its external delegate have committed successfully.

use super::super::frame_assignment::{add, failed, vector_growth};
use super::super::{FrameVec, SlotLifecycle, WorkspaceAssignedReport, WorkspaceFrame};
use super::WorkspaceValueSnapshot;
use crate::bytecode::program::{LexicalBindingKind, LexicalBindingSlot};
use runmat_runtime::call::lexical::{
    LexicalCompletionChange, LexicalCompletionScope, PreparedLexicalCompletionBatch,
};
use runmat_runtime::context::host_allocation::{HostAllocationError, OperationCollectionAllocator};
use runmat_runtime::RuntimeError;
use runmat_value::LexicalEnvironment;

pub(super) enum CompletionLexicalEffect {
    Assign {
        environment: LexicalEnvironment,
        binding: LexicalBindingSlot,
        source: runmat_runtime::workspace::WorkspaceSlotReadLease,
    },
    Retire {
        environment: LexicalEnvironment,
        binding: LexicalBindingSlot,
    },
}

impl CompletionLexicalEffect {
    fn binding(&self) -> LexicalBindingSlot {
        match self {
            Self::Assign { binding, .. } | Self::Retire { binding, .. } => *binding,
        }
    }
}

fn affected(previous: &WorkspaceFrame, report: &WorkspaceAssignedReport, slot: usize) -> bool {
    if report.ids.contains(&slot) || report.removed_ids.contains(&slot) {
        return true;
    }
    let Some(lifecycle) = previous.state.slot_lifecycle.get(&slot) else {
        return false;
    };
    let name = lifecycle.name();
    report.names.contains(name) || report.removed_names.contains(name)
}

pub(super) fn capacity(previous: &WorkspaceFrame, report: &WorkspaceAssignedReport) -> usize {
    previous.lexical.as_ref().map_or(0, |lexical| {
        lexical
            .slots
            .iter()
            .filter(|(slot, binding)| {
                binding.kind != LexicalBindingKind::Transit && affected(previous, report, **slot)
            })
            .count()
    })
}

pub(super) fn quote(total: &mut u64, capacity: usize) -> Result<(), HostAllocationError> {
    add(
        total,
        vector_growth::<CompletionLexicalEffect>(capacity, 0)?.unwrap_or(0),
    )
}

pub(super) fn allocate(
    capacity: usize,
    allocator: OperationCollectionAllocator,
) -> Result<FrameVec<CompletionLexicalEffect>, HostAllocationError> {
    let mut effects = FrameVec::new_in(allocator);
    effects.try_reserve_exact(capacity).map_err(|_| failed())?;
    Ok(effects)
}

pub(super) fn populate(
    effects: &mut FrameVec<CompletionLexicalEffect>,
    previous: &WorkspaceFrame,
    next: &WorkspaceValueSnapshot,
    report: &WorkspaceAssignedReport,
) -> Result<(), RuntimeError> {
    let Some(lexical) = previous.lexical.as_ref() else {
        return Ok(());
    };
    for (previous_slot, binding) in &lexical.slots {
        if binding.kind == LexicalBindingKind::Transit
            || !affected(previous, report, *previous_slot)
        {
            continue;
        }
        let name = previous
            .state
            .slot_lifecycle
            .get(previous_slot)
            .map(SlotLifecycle::name)
            .ok_or_else(|| {
                super::super::workspace_target_error(
                    "affected lexical workspace slot has no name identity",
                )
            })?;
        let effect = match next.names.get(name).copied() {
            Some(slot) if next.assigned.contains(name) => {
                next.vars.get(slot).ok_or_else(|| {
                    super::super::workspace_target_error(
                        "affected lexical workspace slot has no completed value",
                    )
                })?;
                CompletionLexicalEffect::Assign {
                    environment: lexical.environment.clone(),
                    binding: *binding,
                    source: next.vars.retained_lease(slot).ok_or_else(|| {
                        super::super::workspace_target_error(
                            "affected lexical workspace value has no retained generation",
                        )
                    })?,
                }
            }
            _ => CompletionLexicalEffect::Retire {
                environment: lexical.environment.clone(),
                binding: *binding,
            },
        };
        effects.push(effect);
    }
    // The lexical slot is compiler-owned and unique within the activation.
    // Final workspace slots may be remapped by dynamic compilation, so they
    // must not define post-commit publication or error ordering.
    effects.sort_unstable_by_key(|effect| effect.binding().slot);
    debug_assert!(effects.len() <= effects.capacity());
    Ok(())
}

pub(super) fn prepare_batch(
    context: &runmat_runtime::context::RuntimeContext,
    effects: impl ExactSizeIterator<Item = CompletionLexicalEffect>,
) -> Result<PreparedLexicalCompletionBatch, RuntimeError> {
    PreparedLexicalCompletionBatch::prepare(
        context,
        effects.map(|effect| match effect {
            CompletionLexicalEffect::Assign {
                environment,
                binding,
                source,
            } => LexicalCompletionChange::Assign {
                environment,
                binding: binding.binding,
                scope: match binding.kind {
                    LexicalBindingKind::Owned => LexicalCompletionScope::Owned,
                    LexicalBindingKind::Inherited => LexicalCompletionScope::Inherited,
                    LexicalBindingKind::Transit => {
                        unreachable!("transit bindings are excluded during preparation")
                    }
                },
                source,
            },
            CompletionLexicalEffect::Retire {
                environment,
                binding,
            } => LexicalCompletionChange::Remove {
                environment,
                binding: binding.binding,
                scope: match binding.kind {
                    LexicalBindingKind::Owned => LexicalCompletionScope::Owned,
                    LexicalBindingKind::Inherited => LexicalCompletionScope::Inherited,
                    LexicalBindingKind::Transit => {
                        unreachable!("transit bindings are excluded during preparation")
                    }
                },
            },
        }),
    )
}

pub(super) fn commit(
    _frame: super::super::WorkspaceFrameId,
    effects: FrameVec<CompletionLexicalEffect>,
) -> Result<(), RuntimeError> {
    for effect in effects {
        match effect {
            CompletionLexicalEffect::Assign {
                environment,
                binding,
                source,
            } => {
                // Explicit legacy/unbounded completion keeps the historic
                // lexical writer. Managed completion consumes the retained
                // source in PreparedLexicalCompletionBatch instead.
                super::super::publish_lexical_binding(
                    &environment,
                    binding,
                    source.value().clone(),
                )?;
            }
            CompletionLexicalEffect::Retire {
                environment,
                binding,
            } => super::super::retire_lexical_binding(&environment, binding)?,
        }
    }
    Ok(())
}
