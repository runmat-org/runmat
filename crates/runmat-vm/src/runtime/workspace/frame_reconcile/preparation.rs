use super::*;

pub(super) fn prepare(
    frame: WorkspaceFrameId,
    context: Option<&RuntimeContext>,
    values: Vec<(String, Value)>,
) -> Result<PreparedFrameReconcile, RuntimeError> {
    if super::super::frame_is_detaching(frame) {
        return Err(unavailable());
    }
    let owner = context.map_or_else(super::super::current_owner, |context| {
        super::super::WorkspaceOwner::Runtime(context.local_identity())
    });
    let token = with_frame(frame, |selected| {
        selected.state.mutation_phase.acquire(owner, frame)
    })
    .flatten()
    .ok_or_else(unavailable)?;
    let growth = Growth::for_frame(frame, &values)?;
    let quote = with_frame(frame, |frame| growth.quote(&frame.state, &values))
        .ok_or_else(unavailable)?
        .map_err(allocation)?;
    let receipt = HostAllocationReceipt::admit(
        context,
        HostAllocationPlan::new(0, quote).map_err(allocation)?,
    )
    .map_err(allocation)?;
    let (incoming, removed, retired_names, retired_values, replacements) =
        with_frame(frame, |frame| {
            let ws = &frame.state;
            let allocator = ws.collection_gate.allocator();
            let _installed = match (context, receipt.as_ref()) {
                (Some(context), Some(receipt)) => ws.collection_gate.install(context, receipt)?,
                _ => ws.collection_gate.install_unbounded()?,
            };
            let mut unassigned = FrameVec::new_in(allocator.clone());
            unassigned
                .try_reserve_exact(growth.unassigned)
                .map_err(|_| failed())?;
            for (slot, entry) in &ws.slot_lifecycle {
                if let SlotLifecycle::Unassigned(name) = entry {
                    unassigned.push((name.clone(), *slot));
                }
            }
            unassigned.sort_unstable_by(|(left, _), (right, _)| left.as_str().cmp(right.as_str()));
            let mut removed = FrameVec::new_in(allocator.clone());
            removed
                .try_reserve_exact(growth.removed)
                .map_err(|_| failed())?;
            for (name, slot) in &ws.names {
                if ws.assigned.contains(name)
                    && !frame.global_aliases.has_binding_for_slot(*slot)
                    && values
                        .binary_search_by(|(candidate, _)| candidate.as_str().cmp(name.as_str()))
                        .is_err()
                {
                    removed.push((name.clone(), *slot));
                }
            }
            let mut incoming = FrameVec::new_in(allocator.clone());
            incoming
                .try_reserve_exact(growth.incoming)
                .map_err(|_| failed())?;
            let mut next_slot = ws.values.len();
            for (name, _) in &values {
                let current = ws.names.get_key_value(name.as_str());
                let unassigned_slot = unassigned
                    .binary_search_by(|(existing, _)| existing.as_str().cmp(name.as_str()))
                    .ok()
                    .map(|at| (&unassigned[at].0, unassigned[at].1));
                let slot = current
                    .map(|(_, slot)| *slot)
                    .or_else(|| unassigned_slot.map(|(_, slot)| slot))
                    .unwrap_or_else(|| {
                        let slot = next_slot;
                        next_slot += 1;
                        slot
                    });
                if (current.is_some() || unassigned_slot.is_some())
                    && frame.global_aliases.has_binding_for_slot(slot)
                {
                    incoming.push(None);
                    continue;
                }
                let canonical = current
                    .map(|(name, _)| name.clone())
                    .or_else(|| unassigned_slot.map(|(name, _)| name.clone()))
                    .map(Ok)
                    .unwrap_or_else(|| match (context, receipt.as_ref()) {
                        (Some(context), Some(receipt)) => receipt
                            .with_output_allocation(context, |permit| {
                                WorkspaceName::prepare(name, permit)
                            }),
                        _ => Ok(WorkspaceName::unbounded(name)),
                    })?;
                incoming.push(Some((canonical, slot)));
            }
            let mut retired_names = FrameVec::new_in(allocator.clone());
            retired_names
                .try_reserve_exact(growth.retired_names)
                .map_err(|_| failed())?;
            let mut retired_values = FrameVec::new_in(allocator.clone());
            retired_values
                .try_reserve_exact(growth.retired_values)
                .map_err(|_| failed())?;
            let growth_map =
                |len: usize, target: usize, capacity: usize| target > capacity && target > len;
            let replacements = Replacements {
                names: growth_map(ws.names.len(), growth.names, ws.names.capacity())
                    .then(|| copied_map(&ws.names, growth.names, allocator.clone()))
                    .transpose()?,
                assigned: growth_map(ws.assigned.len(), growth.assigned, ws.assigned.capacity())
                    .then(|| copied_set(&ws.assigned, growth.assigned, allocator.clone()))
                    .transpose()?,
                assigned_names: growth_map(
                    ws.assigned_names_this_execution.len(),
                    growth.assigned_names,
                    ws.assigned_names_this_execution.capacity(),
                )
                .then(|| {
                    copied_set(
                        &ws.assigned_names_this_execution,
                        growth.assigned_names,
                        allocator.clone(),
                    )
                })
                .transpose()?,
                assigned_ids: growth_map(
                    ws.assigned_ids_this_execution.len(),
                    growth.assigned_ids,
                    ws.assigned_ids_this_execution.capacity(),
                )
                .then(|| {
                    copied_set(
                        &ws.assigned_ids_this_execution,
                        growth.assigned_ids,
                        allocator.clone(),
                    )
                })
                .transpose()?,
                removed_slots: growth_map(
                    ws.removed_slots_this_execution.len(),
                    growth.removed_slots,
                    ws.removed_slots_this_execution.capacity(),
                )
                .then(|| {
                    copied_map(
                        &ws.removed_slots_this_execution,
                        growth.removed_slots,
                        allocator.clone(),
                    )
                })
                .transpose()?,
                removed_ids: growth_map(
                    ws.removed_ids_this_execution.len(),
                    growth.removed_ids,
                    ws.removed_ids_this_execution.capacity(),
                )
                .then(|| {
                    copied_set(
                        &ws.removed_ids_this_execution,
                        growth.removed_ids,
                        allocator.clone(),
                    )
                })
                .transpose()?,
                removed_names: growth_map(
                    ws.removed_names_this_execution.len(),
                    growth.removed_names,
                    ws.removed_names_this_execution.capacity(),
                )
                .then(|| {
                    copied_set(
                        &ws.removed_names_this_execution,
                        growth.removed_names,
                        allocator.clone(),
                    )
                })
                .transpose()?,
                lifecycle: growth_map(
                    ws.slot_lifecycle.len(),
                    growth.lifecycle,
                    ws.slot_lifecycle.capacity(),
                )
                .then(|| copied_map(&ws.slot_lifecycle, growth.lifecycle, allocator.clone()))
                .transpose()?,
                reconciliation: growth_map(
                    ws.pending_var_reconciliation.len(),
                    growth.reconciliation,
                    ws.pending_var_reconciliation.capacity(),
                )
                .then(|| {
                    copied_set(
                        &ws.pending_var_reconciliation,
                        growth.reconciliation,
                        allocator.clone(),
                    )
                })
                .transpose()?,
                values: if growth.values > ws.values.capacity() {
                    let mut backing: FrameVec<InterpreterSlot> = FrameVec::new_in(allocator);
                    backing
                        .try_reserve_exact(growth.values)
                        .map_err(|_| failed())?;
                    Some(backing)
                } else {
                    None
                },
            };
            Ok::<_, HostAllocationError>((
                incoming,
                removed,
                retired_names,
                retired_values,
                replacements,
            ))
        })
        .ok_or_else(unavailable)?
        .map_err(allocation)?;
    Ok(PreparedFrameReconcile {
        frame,
        token,
        context: context.cloned(),
        receipt,
        values,
        incoming,
        removed,
        retired_names: Some(retired_names),
        retired_values: Some(retired_values),
        replacements,
    })
}
