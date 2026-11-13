//! Exact retained completion for a dynamically evaluated service workspace.

use std::alloc::Layout;

use runmat_runtime::context::host_allocation::{
    collection_block_retained_bytes, HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use runmat_runtime::context::RuntimeContext;
use runmat_runtime::workspace::{
    DynamicWorkspaceTargetCapture, GlobalBindingTarget, WorkspaceCompletionAction,
    WorkspaceCompletionSource, WorkspaceCompletionValue,
};
use runmat_runtime::RuntimeError;

use crate::runtime::workspace::{FrameVec, WorkspaceAssignedReport, WorkspaceValueSnapshot};

struct CompletionEntry<'a> {
    name: &'a str,
    target: Option<GlobalBindingTarget>,
}

pub(super) struct NativeCompletionSource<'a> {
    names: FrameVec<CompletionEntry<'a>>,
    snapshot: &'a WorkspaceValueSnapshot,
    captured: &'a DynamicWorkspaceTargetCapture,
}

fn allocation(error: HostAllocationError) -> RuntimeError {
    runmat_runtime::build_runtime_error(format!("workspace completion admission failed: {error}"))
        .with_identifier(format!("RunMat:{}", error.runtime_identifier_suffix()))
        .with_source(error)
        .build()
}

fn global_target(snapshot: &WorkspaceValueSnapshot, name: &str) -> Option<GlobalBindingTarget> {
    snapshot
        .names
        .get(name)
        .and_then(|slot| snapshot.global_aliases.binding_handle_for_slot(*slot))
}

impl<'a> NativeCompletionSource<'a> {
    pub(super) fn prepare(
        context: &RuntimeContext,
        snapshot: &'a WorkspaceValueSnapshot,
        report: &'a WorkspaceAssignedReport,
        captured: &'a DynamicWorkspaceTargetCapture,
    ) -> Result<Self, RuntimeError> {
        let capacity = [
            report.names.len(),
            report.removed_names.len(),
            captured.global_bindings().count(),
            snapshot.names.len(),
        ]
        .into_iter()
        .try_fold(0usize, |total, count| total.checked_add(count))
        .ok_or_else(|| allocation(HostAllocationError::Overflow))?;
        let bytes = if capacity == 0 {
            0
        } else {
            collection_block_retained_bytes(
                Layout::array::<CompletionEntry<'_>>(capacity)
                    .map_err(|_| allocation(HostAllocationError::Overflow))?,
            )
            .map_err(allocation)?
        };
        let receipt = HostAllocationReceipt::admit(
            Some(context),
            HostAllocationPlan::new(0, bytes).map_err(allocation)?,
        )
        .map_err(allocation)?;
        let installed = match receipt.as_ref() {
            Some(receipt) => snapshot.collection_gate.install(context, receipt),
            None => snapshot.collection_gate.install_unbounded(),
        }
        .map_err(allocation)?;
        let mut names = FrameVec::new_in(snapshot.collection_gate.allocator());
        names.try_reserve_exact(capacity).map_err(|_| {
            allocation(HostAllocationError::Numeric(
                runmat_value::ExactHostNumericAllocationError::AllocationFailed,
            ))
        })?;
        names.extend(
            report
                .names
                .iter()
                .chain(&report.removed_names)
                .map(|name| CompletionEntry {
                    name: name.as_str(),
                    target: None,
                }),
        );
        names.extend(
            captured
                .global_bindings()
                .filter(|(name, target)| global_target(snapshot, name).as_ref() != Some(*target))
                .map(|(name, _)| CompletionEntry { name, target: None }),
        );
        names.extend(
            snapshot
                .names
                .keys()
                .filter(|name| {
                    global_target(snapshot, name.as_str()).as_ref()
                        != captured.global_target(name.as_str())
                })
                .map(|name| CompletionEntry {
                    name: name.as_str(),
                    target: None,
                }),
        );
        names.sort_unstable_by(|left, right| left.name.cmp(right.name));
        names.dedup_by(|left, right| left.name == right.name);
        for entry in &mut names {
            entry.target = global_target(snapshot, entry.name);
        }
        if let Some(receipt) = &receipt {
            receipt.checkpoint(context).map_err(allocation)?;
        }
        drop(installed);
        Ok(Self {
            names,
            snapshot,
            captured,
        })
    }
}

impl WorkspaceCompletionSource for NativeCompletionSource<'_> {
    fn len(&self) -> usize {
        self.names.len()
    }

    fn action(&self, index: usize) -> Option<WorkspaceCompletionAction<'_>> {
        let entry = self.names.get(index)?;
        let name = entry.name;
        if let Some(target) = entry.target.as_ref() {
            return Some(WorkspaceCompletionAction::BindGlobal { name, target });
        }
        if self.snapshot.assigned.contains(name) {
            Some(WorkspaceCompletionAction::AssignLocal {
                name,
                value: WorkspaceCompletionValue::Retained(
                    self.snapshot
                        .vars
                        .retained_lease(*self.snapshot.names.get(name)?)?,
                ),
            })
        } else if self.captured.global_target(name).is_some() {
            Some(WorkspaceCompletionAction::UnbindGlobal { name })
        } else {
            Some(WorkspaceCompletionAction::Remove { name })
        }
    }
}
