use super::*;

pub(super) struct Growth {
    pub(super) names: usize,
    pub(super) assigned: usize,
    pub(super) assigned_names: usize,
    pub(super) assigned_ids: usize,
    pub(super) removed_slots: usize,
    pub(super) removed_ids: usize,
    pub(super) removed_names: usize,
    pub(super) lifecycle: usize,
    pub(super) reconciliation: usize,
    pub(super) values: usize,
    pub(super) unassigned: usize,
    pub(super) removed: usize,
    pub(super) incoming: usize,
    pub(super) retired_names: usize,
    pub(super) retired_values: usize,
}

impl Growth {
    pub(super) fn for_frame(
        frame: WorkspaceFrameId,
        values: &[(String, Value)],
    ) -> Result<Self, RuntimeError> {
        with_frame(frame, |frame| {
            let ws = &frame.state;
            let mut new_names = 0usize;
            let mut new_assigned = 0usize;
            let mut new_reported_names = 0usize;
            let mut new_reported_ids = 0usize;
            let mut new_lifecycle = 0usize;
            let mut new_reconciliation = 0usize;
            for (name, _) in values {
                let existing = ws.names.get(name.as_str()).copied();
                if existing.is_some_and(|slot| frame.global_aliases.has_binding_for_slot(slot)) {
                    continue;
                }
                if existing.is_none() {
                    // A previously unassigned compiler slot may be reused;
                    // these are conservative upper bounds until the sorted
                    // unassigned index is prepared below.
                    new_names += 1;
                    new_assigned += 1;
                    new_reported_names += 1;
                    new_reported_ids += 1;
                    new_lifecycle += 1;
                    new_reconciliation += 1;
                    continue;
                }
                let slot = existing.expect("existing slot was checked");
                new_assigned += usize::from(!ws.assigned.contains(name.as_str()));
                new_reported_names +=
                    usize::from(!ws.assigned_names_this_execution.contains(name.as_str()));
                new_reported_ids += usize::from(!ws.assigned_ids_this_execution.contains(&slot));
                new_lifecycle += usize::from(!ws.slot_lifecycle.contains_key(&slot));
                new_reconciliation += usize::from(!ws.pending_var_reconciliation.contains(&slot));
            }
            let removed = ws
                .names
                .iter()
                .filter(|(name, slot)| {
                    ws.assigned.contains(*name)
                        && !frame.global_aliases.has_binding_for_slot(**slot)
                        && values
                            .binary_search_by(|(candidate, _)| {
                                candidate.as_str().cmp(name.as_str())
                            })
                            .is_err()
                })
                .count();
            let unassigned = ws
                .slot_lifecycle
                .values()
                .filter(|entry| matches!(entry, SlotLifecycle::Unassigned(_)))
                .count();
            let bound = |current: usize, extra: usize| target(current, extra);
            // One removal can displace the canonical names-map key, the
            // assigned-set key, a lifecycle name, and a removed-slot name.
            // One assignment can displace a lifecycle name, removed-slot
            // name, and removed-name key. Each change replaces at most one
            // slot value.
            let retired_names = removed
                .checked_mul(4)
                .and_then(|count| {
                    values
                        .len()
                        .checked_mul(3)
                        .and_then(|incoming| count.checked_add(incoming))
                })
                .ok_or(HostAllocationError::Overflow)?;
            let retired_values = removed
                .checked_add(values.len())
                .ok_or(HostAllocationError::Overflow)?;
            Ok::<_, HostAllocationError>(Growth {
                names: bound(ws.names.len(), new_names)?,
                assigned: bound(ws.assigned.len(), new_assigned)?,
                assigned_names: bound(ws.assigned_names_this_execution.len(), new_reported_names)?,
                assigned_ids: bound(ws.assigned_ids_this_execution.len(), new_reported_ids)?,
                removed_slots: bound(ws.removed_slots_this_execution.len(), removed)?,
                removed_ids: bound(ws.removed_ids_this_execution.len(), removed)?,
                removed_names: bound(ws.removed_names_this_execution.len(), removed)?,
                lifecycle: bound(ws.slot_lifecycle.len(), new_lifecycle)?,
                reconciliation: bound(
                    ws.pending_var_reconciliation.len(),
                    removed
                        .checked_add(new_reconciliation)
                        .ok_or(HostAllocationError::Overflow)?,
                )?,
                values: bound(ws.values.len(), new_names)?,
                unassigned,
                removed,
                incoming: values.len(),
                retired_names,
                retired_values,
            })
        })
        .ok_or_else(unavailable)?
        .map_err(allocation)
    }
    pub(super) fn quote(
        &self,
        ws: &super::WorkspaceState,
        values: &[(String, Value)],
    ) -> Result<u64, HostAllocationError> {
        let mut quote = 0u64;
        for bound in [
            table::<WorkspaceName, usize>(self.names, ws.names.capacity())?,
            table::<WorkspaceName, ()>(self.assigned, ws.assigned.capacity())?,
            table::<WorkspaceName, ()>(
                self.assigned_names,
                ws.assigned_names_this_execution.capacity(),
            )?,
            table::<usize, ()>(self.assigned_ids, ws.assigned_ids_this_execution.capacity())?,
            table::<usize, WorkspaceName>(
                self.removed_slots,
                ws.removed_slots_this_execution.capacity(),
            )?,
            table::<usize, ()>(self.removed_ids, ws.removed_ids_this_execution.capacity())?,
            table::<WorkspaceName, ()>(
                self.removed_names,
                ws.removed_names_this_execution.capacity(),
            )?,
            table::<usize, SlotLifecycle>(self.lifecycle, ws.slot_lifecycle.capacity())?,
            table::<usize, ()>(
                self.reconciliation,
                ws.pending_var_reconciliation.capacity(),
            )?,
            vector_growth::<InterpreterSlot>(self.values, ws.values.capacity())?,
        ]
        .into_iter()
        .flatten()
        {
            add(&mut quote, bound)?;
        }
        for bound in [
            bytes::<(WorkspaceName, usize)>(self.unassigned)?,
            bytes::<(WorkspaceName, usize)>(self.removed)?,
            bytes::<Option<(WorkspaceName, usize)>>(self.incoming)?,
            bytes::<WorkspaceName>(self.retired_names)?,
            bytes::<InterpreterSlot>(self.retired_values)?,
        ] {
            add(&mut quote, bound)?;
        }
        for (name, _) in values {
            if !ws.names.contains_key(name.as_str()) {
                let physical = u64::try_from(
                    WorkspaceName::requested_bytes(name).map_err(HostAllocationError::Numeric)?,
                )
                .map_err(|_| HostAllocationError::Overflow)?;
                add(
                    &mut quote,
                    HostAllocationPlan::retained_exact_output_bytes(physical, 1)?,
                )?;
            }
        }
        Ok(quote)
    }
}

#[derive(Default)]
pub(super) struct Replacements {
    pub(super) names: Option<FrameMap<WorkspaceName, usize>>,
    pub(super) assigned: Option<FrameSet<WorkspaceName>>,
    pub(super) assigned_names: Option<FrameSet<WorkspaceName>>,
    pub(super) assigned_ids: Option<FrameSet<usize>>,
    pub(super) removed_slots: Option<FrameMap<usize, WorkspaceName>>,
    pub(super) removed_ids: Option<FrameSet<usize>>,
    pub(super) removed_names: Option<FrameSet<WorkspaceName>>,
    pub(super) lifecycle: Option<FrameMap<usize, SlotLifecycle>>,
    pub(super) reconciliation: Option<FrameSet<usize>>,
    pub(super) values: Option<FrameVec<InterpreterSlot>>,
}
