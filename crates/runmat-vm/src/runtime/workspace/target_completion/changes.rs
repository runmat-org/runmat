//! Final named effects of one completed dynamic evaluation.

use super::super::frame_assignment::{add, failed, vector_growth};
use super::super::{FrameVec, WorkspaceAssignedReport, WorkspaceFrame, WorkspaceValueSnapshot};
use runmat_runtime::context::host_allocation::{HostAllocationError, OperationCollectionAllocator};
use runmat_runtime::workspace::{GlobalBindingTarget, WorkspaceName};

pub(super) struct CompletionChange {
    pub(super) name: WorkspaceName,
    pub(super) previous_global: bool,
    pub(super) target: Option<GlobalBindingTarget>,
}

pub(super) fn capacity(
    previous: &WorkspaceFrame,
    next: &WorkspaceValueSnapshot,
    report: &WorkspaceAssignedReport,
) -> Result<usize, HostAllocationError> {
    [
        report.names.len(),
        report.removed_names.len(),
        previous.state.names.len(),
        next.names.len(),
    ]
    .into_iter()
    .try_fold(0usize, |total, count| {
        total
            .checked_add(count)
            .ok_or(HostAllocationError::Overflow)
    })
}

pub(super) fn quote(total: &mut u64, capacity: usize) -> Result<(), HostAllocationError> {
    add(
        total,
        vector_growth::<CompletionChange>(capacity, 0)?.unwrap_or(0),
    )
}

pub(super) fn allocate(
    capacity: usize,
    allocator: OperationCollectionAllocator,
) -> Result<FrameVec<CompletionChange>, HostAllocationError> {
    let mut changes = FrameVec::new_in(allocator);
    changes.try_reserve_exact(capacity).map_err(|_| failed())?;
    Ok(changes)
}

pub(super) fn populate(
    changes: &mut FrameVec<CompletionChange>,
    previous: &WorkspaceFrame,
    next: &WorkspaceValueSnapshot,
    report: &WorkspaceAssignedReport,
) {
    changes.extend(
        report
            .names
            .iter()
            .chain(&report.removed_names)
            .map(|name| CompletionChange {
                name: name.clone(),
                previous_global: false,
                target: None,
            }),
    );
    for (name, slot) in &previous.state.names {
        let previous_target = previous.global_aliases.binding_handle_for_slot(*slot);
        let next_target = next
            .names
            .get(name.as_str())
            .and_then(|slot| next.global_aliases.binding_handle_for_slot(*slot));
        if previous_target != next_target {
            changes.push(CompletionChange {
                name: name.clone(),
                previous_global: previous_target.is_some(),
                target: next_target,
            });
        }
    }
    for (name, slot) in &next.names {
        let next_target = next.global_aliases.binding_handle_for_slot(*slot);
        let previous_target = previous
            .state
            .names
            .get(name.as_str())
            .and_then(|slot| previous.global_aliases.binding_handle_for_slot(*slot));
        if next_target != previous_target {
            changes.push(CompletionChange {
                name: name.clone(),
                previous_global: previous_target.is_some(),
                target: next_target,
            });
        }
    }
    changes.sort_unstable_by(|left, right| left.name.as_str().cmp(right.name.as_str()));
    changes.dedup_by(|left, right| left.name == right.name);
    for change in changes {
        change.previous_global = previous
            .state
            .names
            .get(change.name.as_str())
            .and_then(|slot| previous.global_aliases.binding_handle_for_slot(*slot))
            .is_some();
        change.target = next
            .names
            .get(change.name.as_str())
            .and_then(|slot| next.global_aliases.binding_handle_for_slot(*slot));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::workspace::{
        frame::with_frame, frame_metadata, set_transient_workspace_state, workspace_frame_identity,
    };
    use runmat_runtime::context::host_allocation::FrameCollectionGate;
    use runmat_runtime::workspace::bindings::FrameGlobalBindings;
    use runmat_value::Value;
    use std::collections::{HashMap, HashSet};

    fn aliases(target: &GlobalBindingTarget) -> FrameGlobalBindings {
        let aliases = FrameGlobalBindings::default();
        match target {
            GlobalBindingTarget::Named(name) => aliases.declare_named(0, name.as_str().into()),
            GlobalBindingTarget::LegacySlot(slot) => aliases.declare_legacy_slot(*slot),
        }
        aliases
    }

    #[test]
    fn completion_emits_each_named_retarget_and_preserves_legacy_slot_identity() {
        let g = GlobalBindingTarget::Named(WorkspaceName::unbounded("g"));
        let h = GlobalBindingTarget::Named(WorkspaceName::unbounded("h"));
        let j = GlobalBindingTarget::Named(WorkspaceName::unbounded("j"));
        for (previous_target, next_target) in [
            (g, h.clone()),
            (h, j.clone()),
            (j, GlobalBindingTarget::LegacySlot(0)),
        ] {
            let mut vars =
                crate::interpreter::vars::InterpreterVars::from_owned(vec![Value::Num(1.0)]);
            let _guard = set_transient_workspace_state(
                HashMap::from([("x".into(), 0)]),
                HashSet::from(["x".into()]),
                &mut vars,
                aliases(&previous_target),
            );
            let frame = workspace_frame_identity().expect("active frame");
            let gate = FrameCollectionGate::unbounded();
            let allocator = gate.allocator();
            let _installed = gate.install_unbounded().expect("test collection gate");
            let mut names = frame_metadata::empty_map(allocator.clone());
            names.insert(WorkspaceName::unbounded("x"), 0);
            let mut assigned = frame_metadata::empty_set(allocator.clone());
            assigned.insert(WorkspaceName::unbounded("x"));
            let values =
                crate::interpreter::vars::InterpreterVars::from_owned(vec![Value::Num(1.0)]);
            let snapshot = WorkspaceValueSnapshot {
                vars: values,
                names,
                assigned,
                global_aliases: aliases(&next_target),
                collection_gate: gate,
            };
            let report = WorkspaceAssignedReport::default();
            let mut changes = allocate(2, allocator).expect("admitted test changes");
            with_frame(frame, |previous| {
                populate(&mut changes, previous, &snapshot, &report)
            })
            .expect("active frame");
            assert_eq!(changes.len(), 1);
            assert!(changes[0].previous_global);
            assert_eq!(changes[0].target, Some(next_target));
        }
    }
}
