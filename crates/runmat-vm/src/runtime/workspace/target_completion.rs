//! Prepared metadata for replacing an existing frame with dynamic evaluation.
//!
//! The completed values keep their original collection allocator. Auxiliary
//! indexes are rebuilt under that same gate before any external publication;
//! committing the selected frame then only moves owned backing.

use super::frame::{with_frame, with_frame_mut_unlocked};
use super::frame_assignment::{add, failed};
use super::frame_metadata::{empty_map, empty_set, FrameMutationToken};
use super::{
    SlotLifecycle, WorkspaceAssignedReport, WorkspaceFrameId, WorkspaceState,
    WorkspaceValueSnapshot,
};
use runmat_runtime::context::host_allocation::{
    frame_hash_table_retained_bytes, HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use runmat_runtime::context::RuntimeContext;
use runmat_runtime::workspace::WorkspaceName;
use runmat_runtime::workspace::{
    WorkspaceCompletionAction, WorkspaceCompletionSource, WorkspaceCompletionValue,
};
use runmat_runtime::RuntimeError;

mod changes;
mod lexical;

fn unavailable() -> RuntimeError {
    super::workspace_target_error("workspace target is closed or a mutation is active")
}

fn sum(left: usize, right: usize) -> Result<usize, HostAllocationError> {
    left.checked_add(right).ok_or(HostAllocationError::Overflow)
}

struct MetadataCapacity {
    changes: usize,
    lexical: usize,
    lifecycle: usize,
    reconciliation: usize,
    assigned_names: usize,
    assigned_ids: usize,
    removed_names: usize,
    removed_ids: usize,
    removed_slots: usize,
}

impl MetadataCapacity {
    fn for_completion(
        previous: &WorkspaceState,
        next: &WorkspaceValueSnapshot,
        report: &WorkspaceAssignedReport,
    ) -> Result<Self, HostAllocationError> {
        let lifecycle = sum(previous.slot_lifecycle.len(), next.names.len())?;
        let removed_slots = sum(previous.removed_slots_this_execution.len(), lifecycle)?;
        Ok(Self {
            changes: 0,
            lexical: 0,
            lifecycle,
            reconciliation: sum(previous.pending_var_reconciliation.len(), lifecycle)?,
            assigned_names: sum(
                previous.assigned_names_this_execution.len(),
                report.names.len(),
            )?,
            assigned_ids: sum(previous.assigned_ids_this_execution.len(), report.ids.len())?,
            removed_names: sum(
                sum(
                    previous.removed_names_this_execution.len(),
                    report.removed_names.len(),
                )?,
                lifecycle,
            )?,
            removed_ids: sum(
                sum(
                    previous.removed_ids_this_execution.len(),
                    report.removed_ids.len(),
                )?,
                lifecycle,
            )?,
            removed_slots,
        })
    }

    fn plan(&self) -> Result<HostAllocationPlan, HostAllocationError> {
        let mut retained = 0;
        for bytes in [
            frame_hash_table_retained_bytes::<usize, SlotLifecycle>(self.lifecycle)?,
            frame_hash_table_retained_bytes::<usize, ()>(self.reconciliation)?,
            frame_hash_table_retained_bytes::<WorkspaceName, ()>(self.assigned_names)?,
            frame_hash_table_retained_bytes::<usize, ()>(self.assigned_ids)?,
            frame_hash_table_retained_bytes::<WorkspaceName, ()>(self.removed_names)?,
            frame_hash_table_retained_bytes::<usize, ()>(self.removed_ids)?,
            frame_hash_table_retained_bytes::<usize, WorkspaceName>(self.removed_slots)?,
        ] {
            add(&mut retained, bytes)?;
        }
        changes::quote(&mut retained, self.changes)?;
        lexical::quote(&mut retained, self.lexical)?;
        HostAllocationPlan::new(0, retained)
    }
}

pub(super) struct PreparedTargetCompletion {
    frame: WorkspaceFrameId,
    token: FrameMutationToken,
    context: Option<RuntimeContext>,
    receipt: Option<HostAllocationReceipt>,
    snapshot: Option<WorkspaceValueSnapshot>,
    metadata: Option<WorkspaceState>,
    changes: super::FrameVec<changes::CompletionChange>,
    lexical: super::FrameVec<lexical::CompletionLexicalEffect>,
    lexical_batch: Option<runmat_runtime::call::lexical::PreparedLexicalCompletionBatch>,
    aliases: Option<runmat_runtime::workspace::bindings::PreparedAliasReplacement>,
}

pub(super) struct TargetCompletionRetirement {
    _frame: WorkspaceState,
    _aliases: runmat_runtime::workspace::bindings::PreparedAliasReplacementRetirement,
}

impl PreparedTargetCompletion {
    pub(super) fn prepare(
        frame: WorkspaceFrameId,
        context: Option<&RuntimeContext>,
        mut snapshot: WorkspaceValueSnapshot,
        report: WorkspaceAssignedReport,
    ) -> Result<Self, RuntimeError> {
        if super::frame_is_detaching(frame) {
            return Err(unavailable());
        }
        // The child compiler may append private temporaries after all named
        // workspace slots. Those slots are not part of the completed target.
        // Preserve the target's prior compiler-frame extent plus every final
        // named slot, then retire only the child's trailing temporaries.
        let retained_value_count = with_frame(frame, |target| {
            let named = match snapshot.names.values().copied().max() {
                Some(slot) => slot.checked_add(1).ok_or_else(|| {
                    super::workspace_target_error("completed workspace slot extent overflowed")
                })?,
                None => 0,
            };
            Ok::<_, RuntimeError>(target.state.values.len().max(named))
        })
        .ok_or_else(unavailable)??;
        if snapshot.vars.len() < retained_value_count {
            return Err(super::workspace_target_error(
                "completed workspace lost a target frame value slot",
            ));
        }
        snapshot.vars.truncate(retained_value_count);
        if !snapshot.vars.has_shared_control() {
            return Err(super::workspace_target_error(
                "completed workspace has no transferable slot control",
            ));
        }

        // Validate before preparing any external effect. The values are
        // already owned by the completion; validation does not copy them.
        for name in &snapshot.assigned {
            let value = snapshot
                .names
                .get(name.as_str())
                .and_then(|slot| snapshot.vars.get(*slot))
                .ok_or_else(|| {
                    super::workspace_target_error("assigned workspace slot has no value")
                })?;
            runmat_runtime::execution::validate_storable_value(&value)?;
        }
        let (token, capacity, phase, aliases) = with_frame(frame, |target| {
            let token = target
                .state
                .mutation_phase
                .acquire(
                    context
                        .map(|context| super::WorkspaceOwner::Runtime(context.local_identity()))
                        .unwrap_or_else(super::current_owner),
                    frame,
                )
                .ok_or_else(unavailable)?;
            let mut capacity = MetadataCapacity::for_completion(&target.state, &snapshot, &report)
                .map_err(super::frame_allocation_error)?;
            capacity.changes = changes::capacity(target, &snapshot, &report)
                .map_err(super::frame_allocation_error)?;
            capacity.lexical = lexical::capacity(target, &report);
            Ok::<_, RuntimeError>((
                token,
                capacity,
                target.state.mutation_phase.clone(),
                target.global_aliases.clone(),
            ))
        })
        .ok_or_else(unavailable)??;
        let aliases = aliases
            .prepare_replace_with_owned(snapshot.global_aliases.clone())
            .map_err(super::frame_allocation_error)?;
        let receipt = HostAllocationReceipt::admit(
            context,
            capacity.plan().map_err(super::frame_allocation_error)?,
        )
        .map_err(super::frame_allocation_error)?;
        let gate = snapshot.collection_gate.clone();
        let _installed = match (context, receipt.as_ref()) {
            (Some(context), Some(receipt)) => gate.install(context, receipt),
            _ => gate.install_unbounded(),
        }
        .map_err(super::frame_allocation_error)?;
        let allocator = gate.allocator();
        let mut changes = changes::allocate(capacity.changes, allocator.clone())
            .map_err(super::frame_allocation_error)?;
        let mut lexical = lexical::allocate(capacity.lexical, allocator.clone())
            .map_err(super::frame_allocation_error)?;
        let mut metadata = WorkspaceState {
            names: empty_map(allocator.clone()),
            assigned: empty_set(allocator.clone()),
            assigned_names_this_execution: empty_set(allocator.clone()),
            assigned_ids_this_execution: empty_set(allocator.clone()),
            removed_slots_this_execution: empty_map(allocator.clone()),
            removed_ids_this_execution: empty_set(allocator.clone()),
            removed_names_this_execution: empty_set(allocator.clone()),
            slot_lifecycle: empty_map(allocator.clone()),
            values: crate::interpreter::vars::InterpreterVars::from_retained(
                super::FrameVec::new_in(allocator.clone()),
            ),
            pending_var_reconciliation: empty_set(allocator),
            collection_gate: gate.clone(),
            mutation_phase: phase,
        };
        metadata
            .slot_lifecycle
            .try_reserve(capacity.lifecycle)
            .map_err(|_| super::frame_allocation_error(failed()))?;
        metadata
            .pending_var_reconciliation
            .try_reserve(capacity.reconciliation)
            .map_err(|_| super::frame_allocation_error(failed()))?;
        metadata
            .assigned_names_this_execution
            .try_reserve(capacity.assigned_names)
            .map_err(|_| super::frame_allocation_error(failed()))?;
        metadata
            .assigned_ids_this_execution
            .try_reserve(capacity.assigned_ids)
            .map_err(|_| super::frame_allocation_error(failed()))?;
        metadata
            .removed_names_this_execution
            .try_reserve(capacity.removed_names)
            .map_err(|_| super::frame_allocation_error(failed()))?;
        metadata
            .removed_ids_this_execution
            .try_reserve(capacity.removed_ids)
            .map_err(|_| super::frame_allocation_error(failed()))?;
        metadata
            .removed_slots_this_execution
            .try_reserve(capacity.removed_slots)
            .map_err(|_| super::frame_allocation_error(failed()))?;
        with_frame(frame, |target| {
            changes::populate(&mut changes, target, &snapshot, &report);
            lexical::populate(&mut lexical, target, &snapshot, &report)?;
            let previous = &target.state;
            metadata
                .assigned_names_this_execution
                .extend(previous.assigned_names_this_execution.iter().cloned());
            metadata.assigned_names_this_execution.extend(report.names);
            metadata
                .assigned_ids_this_execution
                .extend(previous.assigned_ids_this_execution.iter().copied());
            metadata.assigned_ids_this_execution.extend(report.ids);
            metadata
                .removed_names_this_execution
                .extend(previous.removed_names_this_execution.iter().cloned());
            metadata
                .removed_names_this_execution
                .extend(report.removed_names);
            metadata
                .removed_ids_this_execution
                .extend(previous.removed_ids_this_execution.iter().copied());
            metadata
                .removed_ids_this_execution
                .extend(report.removed_ids);
            metadata.removed_slots_this_execution.extend(
                previous
                    .removed_slots_this_execution
                    .iter()
                    .map(|(slot, name)| (*slot, name.clone())),
            );
            metadata
                .pending_var_reconciliation
                .extend(previous.pending_var_reconciliation.iter().copied());
            for (slot, lifecycle) in &previous.slot_lifecycle {
                let name = match lifecycle {
                    SlotLifecycle::Assigned(name) | SlotLifecycle::Unassigned(name) => name,
                };
                metadata
                    .slot_lifecycle
                    .insert(*slot, SlotLifecycle::Unassigned(name.clone()));
                metadata.pending_var_reconciliation.insert(*slot);
                if lifecycle.is_assigned() && !snapshot.assigned.contains(name.as_str()) {
                    metadata
                        .removed_slots_this_execution
                        .insert(*slot, name.clone());
                    metadata.removed_names_this_execution.insert(name.clone());
                    metadata.removed_ids_this_execution.insert(*slot);
                }
            }
            for (name, slot) in &snapshot.names {
                let lifecycle = if snapshot.assigned.contains(name.as_str()) {
                    metadata.removed_slots_this_execution.remove(slot);
                    metadata.removed_ids_this_execution.remove(slot);
                    metadata.removed_names_this_execution.remove(name.as_str());
                    SlotLifecycle::Assigned(name.clone())
                } else {
                    SlotLifecycle::Unassigned(name.clone())
                };
                metadata.slot_lifecycle.insert(*slot, lifecycle);
                metadata.pending_var_reconciliation.insert(*slot);
            }
            Ok::<_, RuntimeError>(())
        })
        .ok_or_else(unavailable)??;
        let lexical_batch = context
            .map(|context| lexical::prepare_batch(context, lexical.drain(..)))
            .transpose()?;
        drop(_installed);
        Ok(Self {
            frame,
            token,
            context: context.cloned(),
            receipt,
            snapshot: Some(snapshot),
            metadata: Some(metadata),
            changes,
            lexical,
            lexical_batch,
            aliases: Some(aliases),
        })
    }

    pub(super) fn checkpoint_before_publication(&self) -> Result<(), RuntimeError> {
        if !self.token.is_prepared() {
            return Err(unavailable());
        }
        if let (Some(context), Some(receipt)) = (&self.context, &self.receipt) {
            receipt
                .checkpoint(context)
                .map_err(super::frame_allocation_error)?;
        }
        self.aliases
            .as_ref()
            .ok_or_else(unavailable)?
            .checkpoint()
            .map_err(super::frame_allocation_error)?;
        if !self.token.begin_publication() {
            return Err(unavailable());
        }
        Ok(())
    }

    pub(super) fn commit_with_base(
        mut self,
        context: &RuntimeContext,
        base: runmat_runtime::workspace::PreparedWorkspaceCompletion,
    ) -> Result<(), RuntimeError> {
        let mut lexical = self.lexical_batch.take().ok_or_else(|| {
            super::workspace_target_error("managed completion has no prepared lexical batch")
        })?;
        let retired = base.commit_with_lexical_and_participant(context, &mut lexical, &mut self)?;
        // The coordinator has released Base/GC guards and completed observer
        // publication before returning this displaced frame generation.
        drop(retired);
        Ok(())
    }

    pub(super) fn commit_without_base(
        mut self,
        context: &RuntimeContext,
    ) -> Result<(), RuntimeError> {
        let mut lexical = self.lexical_batch.take().ok_or_else(|| {
            super::workspace_target_error("managed completion has no prepared lexical batch")
        })?;
        let retired = lexical.commit_with_participant(context, &mut self)?;
        drop(retired);
        Ok(())
    }

    pub(super) fn commit(mut self) -> Result<(), RuntimeError> {
        if self.lexical_batch.is_some() {
            return Err(super::workspace_target_error(
                "managed lexical completion requires its atomic coordinator",
            ));
        }
        assert!(
            self.token.is_publishing(),
            "target publication must be prepared first"
        );
        let retired =
            runmat_runtime::workspace::WorkspaceCompletionParticipant::install_prevalidated(
                &mut self,
            );
        // Retiring replaced slot generations may release GC/foreign owners;
        // do that only after the selected frame borrow is gone.
        drop(retired);
        lexical::commit(self.frame, self.lexical)
    }
}

impl runmat_runtime::workspace::WorkspaceCompletionParticipant for PreparedTargetCompletion {
    type Retirement = TargetCompletionRetirement;

    fn checkpoint_before_install(&mut self, context: &RuntimeContext) -> Result<(), RuntimeError> {
        if self.context.as_ref().is_none_or(|owner| {
            owner.local_identity() != context.local_identity()
                || !owner
                    .cancellation()
                    .same_generation(&context.cancellation())
        }) {
            return Err(unavailable());
        }
        self.checkpoint_before_publication()
    }

    fn install_prevalidated(&mut self) -> Self::Retirement {
        assert!(
            self.token.is_publishing(),
            "target publication must be prepared first"
        );
        let WorkspaceValueSnapshot {
            vars,
            names,
            assigned,
            global_aliases: _,
            collection_gate,
        } = self.snapshot.take().expect("prepared completion snapshot");
        let mut metadata = self.metadata.take().expect("prepared completion metadata");
        metadata.names = names;
        metadata.assigned = assigned;
        metadata.values = vars;
        metadata.collection_gate = collection_gate;
        let aliases = self
            .aliases
            .take()
            .expect("prepared completion aliases")
            .commit_retaining();
        let frame = with_frame_mut_unlocked(self.frame, |frame| {
            std::mem::replace(&mut frame.state, metadata)
        })
        .expect("the completion phase retains its selected frame through commit");
        TargetCompletionRetirement {
            _frame: frame,
            _aliases: aliases,
        }
    }
}

impl WorkspaceCompletionSource for PreparedTargetCompletion {
    fn len(&self) -> usize {
        self.changes.len()
    }

    fn action(&self, index: usize) -> Option<WorkspaceCompletionAction<'_>> {
        let change = self.changes.get(index)?;
        let name = change.name.as_str();
        let snapshot = self.snapshot.as_ref()?;
        let slot = snapshot.names.get(name).copied();
        if let Some(target) = change.target.as_ref() {
            Some(WorkspaceCompletionAction::BindGlobal { name, target })
        } else if snapshot.assigned.contains(name) {
            Some(WorkspaceCompletionAction::AssignLocal {
                name,
                value: WorkspaceCompletionValue::Retained(snapshot.vars.retained_lease(slot?)?),
            })
        } else if change.previous_global {
            Some(WorkspaceCompletionAction::UnbindGlobal { name })
        } else {
            Some(WorkspaceCompletionAction::Remove { name })
        }
    }
}
