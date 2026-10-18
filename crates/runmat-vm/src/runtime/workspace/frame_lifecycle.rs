//! Pre-admitted clear/remove/unassign mutations of one VM frame.
//!
//! The same operation token fences reentry while a service publishes the
//! corresponding external effect. Retained removal and reconciliation tables
//! are prepared before that effect; final frame mutation is allocation-free.

use super::frame::{with_frame, with_frame_mut_unlocked, WorkspaceFrameId};
use super::frame_assignment::{add, copied_map, copied_set, vector_growth};
use super::frame_metadata::{FrameMap, FrameMutationToken, FrameSet, FrameVec};
use super::{SlotLifecycle, WorkspaceOwner};
use crate::interpreter::vars::InterpreterSlot;
use runmat_runtime::context::host_allocation::{
    frame_hash_table_retained_bytes, HostAllocationError, HostAllocationPlan,
    HostAllocationReceipt, OperationCollectionAllocator,
};
use runmat_runtime::context::RuntimeContext;
use runmat_runtime::workspace::bindings::FrameGlobalBindingsRetirement;
use runmat_runtime::workspace::{DetachedWorkspaceValues, WorkspaceDetachSelection, WorkspaceName};
use runmat_runtime::RuntimeError;
use std::hash::Hash;

enum LifecycleChange<'a> {
    Clear,
    Remove(Option<WorkspaceName>),
    Unassign(usize, Option<WorkspaceName>),
    Detach(&'a WorkspaceDetachSelection, &'a DetachedWorkspaceValues),
}

mod retirement;
use retirement::{FrameLifecycleRetirement, LifecycleReplacements};

fn unavailable() -> RuntimeError {
    runmat_runtime::runtime_error::semantic_error(
        "RunMat:Workspace:FrameUnavailable",
        "workspace frame is closed or a frame mutation is active",
    )
}

fn grow<K: Eq + Hash, V>(
    target: usize,
    capacity: usize,
) -> Result<Option<u64>, HostAllocationError> {
    (target > capacity)
        .then(|| frame_hash_table_retained_bytes::<K, V>(target))
        .transpose()
}

fn target(current: usize, additional: usize) -> Result<usize, HostAllocationError> {
    current
        .checked_add(additional)
        .ok_or(HostAllocationError::Overflow)
}

#[derive(Default)]
struct LifecycleAdditions {
    removed_slots: usize,
    removed_ids: usize,
    removed_names: usize,
    lifecycle: usize,
    reconciliation: usize,
}

struct LifecycleTargets {
    removed_slots: usize,
    removed_ids: usize,
    removed_names: usize,
    lifecycle: usize,
    reconciliation: usize,
    retired_names: usize,
    retired_values: usize,
    alias_retirements: usize,
}

impl LifecycleAdditions {
    fn observe(
        &mut self,
        ws: &super::WorkspaceState,
        slot: usize,
        name: &WorkspaceName,
        reconcile: bool,
    ) -> Result<(), HostAllocationError> {
        let count = |field: &mut usize, absent: bool| {
            *field = target(*field, usize::from(absent))?;
            Ok::<_, HostAllocationError>(())
        };
        count(
            &mut self.removed_slots,
            !ws.removed_slots_this_execution.contains_key(&slot),
        )?;
        count(
            &mut self.removed_ids,
            !ws.removed_ids_this_execution.contains(&slot),
        )?;
        count(
            &mut self.removed_names,
            !ws.removed_names_this_execution.contains(name.as_str()),
        )?;
        count(&mut self.lifecycle, !ws.slot_lifecycle.contains_key(&slot))?;
        count(
            &mut self.reconciliation,
            reconcile && !ws.pending_var_reconciliation.contains(&slot),
        )
    }

    fn for_change(
        ws: &super::WorkspaceState,
        change: &LifecycleChange<'_>,
    ) -> Result<LifecycleTargets, HostAllocationError> {
        let mut additions = Self::default();
        let (affected, alias_retirements) = match change {
            LifecycleChange::Clear | LifecycleChange::Detach(WorkspaceDetachSelection::All, _) => {
                for (name, slot) in &ws.names {
                    additions.observe(ws, *slot, name, true)?;
                }
                (ws.names.len(), 1)
            }
            LifecycleChange::Remove(Some(name)) => {
                if let Some(slot) = ws.names.get(name.as_str()) {
                    additions.observe(ws, *slot, name, true)?;
                    (1, 1)
                } else {
                    (0, 0)
                }
            }
            LifecycleChange::Unassign(slot, Some(name)) => {
                additions.observe(ws, *slot, name, false)?;
                (1, 0)
            }
            LifecycleChange::Detach(WorkspaceDetachSelection::Names(names), _) => {
                let mut affected = 0usize;
                for name in names {
                    if let Some((canonical, slot)) = ws.names.get_key_value(name.as_str()) {
                        additions.observe(ws, *slot, canonical, true)?;
                        affected = target(affected, 1)?;
                    }
                }
                (affected, affected)
            }
            LifecycleChange::Detach(WorkspaceDetachSelection::Except(_), detached) => {
                let mut affected = 0usize;
                for (name, _) in detached.entries() {
                    if let Some((canonical, slot)) = ws.names.get_key_value(name.as_str()) {
                        additions.observe(ws, *slot, canonical, true)?;
                        affected = target(affected, 1)?;
                    }
                }
                (affected, affected)
            }
            LifecycleChange::Remove(None) | LifecycleChange::Unassign(_, None) => (0, 0),
        };
        // Individual remove/unassign operations can displace the names-map
        // key, assigned-set key, lifecycle name, and removed-slot name. Clear
        // retains its whole names/assigned tables, so this bound is strictly
        // conservative while still derived from the number of affected slots.
        let retired_names = affected
            .checked_mul(4)
            .ok_or(HostAllocationError::Overflow)?;
        Ok(LifecycleTargets {
            removed_slots: target(
                ws.removed_slots_this_execution.len(),
                additions.removed_slots,
            )?,
            removed_ids: target(ws.removed_ids_this_execution.len(), additions.removed_ids)?,
            removed_names: target(
                ws.removed_names_this_execution.len(),
                additions.removed_names,
            )?,
            lifecycle: target(ws.slot_lifecycle.len(), additions.lifecycle)?,
            reconciliation: target(
                ws.pending_var_reconciliation.len(),
                additions.reconciliation,
            )?,
            retired_names,
            retired_values: affected,
            alias_retirements,
        })
    }
}

pub(crate) struct PreparedFrameLifecycle<'a> {
    frame: WorkspaceFrameId,
    token: FrameMutationToken,
    context: Option<RuntimeContext>,
    receipt: Option<HostAllocationReceipt>,
    change: LifecycleChange<'a>,
    removed_slots: Option<FrameMap<usize, WorkspaceName>>,
    removed_ids: Option<FrameSet<usize>>,
    removed_names: Option<FrameSet<WorkspaceName>>,
    lifecycle: Option<FrameMap<usize, SlotLifecycle>>,
    reconciliation: Option<FrameSet<usize>>,
    retired_names: Option<FrameVec<WorkspaceName>>,
    retired_values: Option<FrameVec<InterpreterSlot>>,
    alias_retirements: Option<FrameVec<FrameGlobalBindingsRetirement>>,
}

impl PreparedFrameLifecycle<'_> {
    pub(crate) fn checkpoint_before_publication(&self) -> Result<(), RuntimeError> {
        if !self.token.is_prepared() {
            return Err(unavailable());
        }
        if let (Some(context), Some(receipt)) = (&self.context, &self.receipt) {
            receipt
                .checkpoint(context)
                .map_err(super::frame_allocation_error)?;
        }
        if !self.token.begin_publication() {
            return Err(unavailable());
        }
        Ok(())
    }

    pub(crate) fn commit(mut self) {
        assert!(
            self.token.is_publishing(),
            "prepared frame publication was not begun"
        );
        let mut retired_names = self
            .retired_names
            .take()
            .expect("prepared lifecycle retirement names");
        let mut retired_values = self
            .retired_values
            .take()
            .expect("prepared lifecycle retirement values");
        let mut alias_retirements = self
            .alias_retirements
            .take()
            .expect("prepared lifecycle alias retirements");
        let retired = with_frame_mut_unlocked(self.frame, |frame| {
            let mut replacements = LifecycleReplacements::default();
            if let Some(replacement) = self.removed_slots.take() {
                replacements.removed_slots = Some(std::mem::replace(
                    &mut frame.state.removed_slots_this_execution,
                    replacement,
                ));
            }
            if let Some(replacement) = self.removed_ids.take() {
                replacements.removed_ids = Some(std::mem::replace(
                    &mut frame.state.removed_ids_this_execution,
                    replacement,
                ));
            }
            if let Some(replacement) = self.removed_names.take() {
                replacements.removed_names = Some(std::mem::replace(
                    &mut frame.state.removed_names_this_execution,
                    replacement,
                ));
            }
            if let Some(replacement) = self.lifecycle.take() {
                replacements.lifecycle = Some(std::mem::replace(
                    &mut frame.state.slot_lifecycle,
                    replacement,
                ));
            }
            if let Some(replacement) = self.reconciliation.take() {
                replacements.reconciliation = Some(std::mem::replace(
                    &mut frame.state.pending_var_reconciliation,
                    replacement,
                ));
            }
            match &self.change {
                LifecycleChange::Clear => {
                    clear_prepared(
                        frame,
                        &mut replacements,
                        &mut retired_names,
                        &mut retired_values,
                        &mut alias_retirements,
                    );
                }
                LifecycleChange::Remove(Some(name)) => {
                    remove_prepared(
                        frame,
                        name.as_str(),
                        &mut retired_names,
                        &mut retired_values,
                        &mut alias_retirements,
                    );
                }
                LifecycleChange::Unassign(slot, Some(name)) => {
                    let ws = &mut frame.state;
                    if let Some(canonical) = ws.assigned.take(name.as_str()) {
                        retired_names.push(canonical);
                    }
                    retired_names.extend(
                        super::mark_slot_unassigned_retaining(ws, *slot, name.clone())
                            .into_iter()
                            .flatten(),
                    );
                    if let Some(value) = ws
                        .values
                        .replace_owned(*slot, runmat_value::Value::Num(0.0))
                    {
                        retired_values.push(value);
                    }
                }
                LifecycleChange::Remove(None) | LifecycleChange::Unassign(_, None) => {}
                LifecycleChange::Detach(selection, detached) => match selection {
                    WorkspaceDetachSelection::All => {
                        clear_prepared(
                            frame,
                            &mut replacements,
                            &mut retired_names,
                            &mut retired_values,
                            &mut alias_retirements,
                        );
                    }
                    WorkspaceDetachSelection::Names(names) => {
                        for name in names {
                            remove_prepared(
                                frame,
                                name,
                                &mut retired_names,
                                &mut retired_values,
                                &mut alias_retirements,
                            );
                        }
                    }
                    WorkspaceDetachSelection::Except(_) => {
                        for (name, _) in detached.entries() {
                            remove_prepared(
                                frame,
                                name,
                                &mut retired_names,
                                &mut retired_values,
                                &mut alias_retirements,
                            );
                        }
                    }
                },
            }
            FrameLifecycleRetirement {
                _replacements: replacements,
                _names: retired_names,
                _values: retired_values,
                _aliases: alias_retirements,
            }
        })
        .expect("a prepared frame is held until lifecycle publication commits");
        drop(self);
        drop(retired);
    }
}

fn clear_prepared(
    frame: &mut super::WorkspaceFrame,
    replacements: &mut LifecycleReplacements,
    retired_names: &mut FrameVec<WorkspaceName>,
    retired_values: &mut FrameVec<InterpreterSlot>,
    alias_retirements: &mut FrameVec<FrameGlobalBindingsRetirement>,
) {
    let ws = &mut frame.state;
    let names = std::mem::replace(
        &mut ws.names,
        super::frame_metadata::empty_map(ws.collection_gate.allocator()),
    );
    replacements.cleared_names = Some(names);
    replacements.cleared_assigned = Some(std::mem::replace(
        &mut ws.assigned,
        super::frame_metadata::empty_set(ws.collection_gate.allocator()),
    ));
    for (name, slot) in replacements
        .cleared_names
        .as_ref()
        .expect("cleared names were retained")
    {
        if let Some(value) = ws
            .values
            .replace_owned(*slot, runmat_value::Value::Num(0.0))
        {
            retired_values.push(value);
        }
        retired_names.extend(
            super::mark_slot_unassigned_retaining(ws, *slot, name.clone())
                .into_iter()
                .flatten(),
        );
        if !ws.pending_var_reconciliation.contains(slot) {
            ws.pending_var_reconciliation.insert(*slot);
        }
    }
    alias_retirements.push(frame.global_aliases.unbind_all_retaining());
}

fn remove_prepared(
    frame: &mut super::WorkspaceFrame,
    name: &str,
    retired_names: &mut FrameVec<WorkspaceName>,
    retired_values: &mut FrameVec<InterpreterSlot>,
    alias_retirements: &mut FrameVec<FrameGlobalBindingsRetirement>,
) {
    let ws = &mut frame.state;
    if let Some((canonical, slot)) = ws.names.remove_entry(name) {
        if let Some(value) = ws.values.replace_owned(slot, runmat_value::Value::Num(0.0)) {
            retired_values.push(value);
        }
        if let Some(name) = ws.assigned.take(canonical.as_str()) {
            retired_names.push(name);
        }
        alias_retirements.push(frame.global_aliases.unbind_slot_retaining(slot));
        retired_names.extend(
            super::mark_slot_unassigned_retaining(ws, slot, canonical.clone())
                .into_iter()
                .flatten(),
        );
        retired_names.push(canonical);
        if !ws.pending_var_reconciliation.contains(&slot) {
            ws.pending_var_reconciliation.insert(slot);
        }
    }
}

fn prepare<'a>(
    frame: WorkspaceFrameId,
    context: Option<&RuntimeContext>,
    allow_detaching: bool,
    requested: impl FnOnce(&super::WorkspaceState) -> LifecycleChange<'a>,
) -> Result<PreparedFrameLifecycle<'a>, RuntimeError> {
    if super::frame_is_detaching(frame) && !allow_detaching {
        return Err(unavailable());
    }
    let owner = context.map_or_else(super::current_owner, |context| {
        WorkspaceOwner::Runtime(context.local_identity())
    });
    let token = with_frame(frame, |selected| {
        selected.state.mutation_phase.acquire(owner, frame)
    })
    .flatten()
    .ok_or_else(unavailable)?;
    let context = context
        .cloned()
        .or_else(runmat_runtime::context::legacy::active);
    let (change, quote, growth) = with_frame(frame, |frame| {
        let ws = &frame.state;
        let change = requested(ws);
        let growth = LifecycleAdditions::for_change(ws, &change)?;
        let mut quote = 0u64;
        let bounds = [
            grow::<usize, WorkspaceName>(
                growth.removed_slots,
                ws.removed_slots_this_execution.capacity(),
            )?,
            grow::<usize, ()>(growth.removed_ids, ws.removed_ids_this_execution.capacity())?,
            grow::<WorkspaceName, ()>(
                growth.removed_names,
                ws.removed_names_this_execution.capacity(),
            )?,
            grow::<usize, SlotLifecycle>(growth.lifecycle, ws.slot_lifecycle.capacity())?,
            grow::<usize, ()>(
                growth.reconciliation,
                ws.pending_var_reconciliation.capacity(),
            )?,
        ];
        for bytes in bounds.into_iter().flatten() {
            add(&mut quote, bytes)?;
        }
        for bytes in [
            vector_growth::<WorkspaceName>(growth.retired_names, 0)?,
            vector_growth::<InterpreterSlot>(growth.retired_values, 0)?,
            vector_growth::<FrameGlobalBindingsRetirement>(growth.alias_retirements, 0)?,
        ]
        .into_iter()
        .flatten()
        {
            add(&mut quote, bytes)?;
        }
        Ok::<_, HostAllocationError>((change, quote, growth))
    })
    .ok_or_else(unavailable)?
    .map_err(super::frame_allocation_error)?;
    let receipt = HostAllocationReceipt::admit(
        context.as_ref(),
        HostAllocationPlan::new(0, quote).map_err(super::frame_allocation_error)?,
    )
    .map_err(super::frame_allocation_error)?;
    let prepared = with_frame(frame, |frame| {
        let ws = &frame.state;
        let allocator: OperationCollectionAllocator = ws.collection_gate.allocator();
        let _installed = match (context.as_ref(), receipt.as_ref()) {
            (Some(context), Some(receipt)) => ws.collection_gate.install(context, receipt)?,
            _ => ws.collection_gate.install_unbounded()?,
        };
        let slots_target = growth.removed_slots;
        let ids_target = growth.removed_ids;
        let names_target = growth.removed_names;
        let lifecycle_target = growth.lifecycle;
        let reconciliation_target = growth.reconciliation;
        let slots = (slots_target > ws.removed_slots_this_execution.capacity())
            .then(|| {
                copied_map(
                    &ws.removed_slots_this_execution,
                    slots_target,
                    allocator.clone(),
                )
            })
            .transpose()?;
        let ids = (ids_target > ws.removed_ids_this_execution.capacity())
            .then(|| {
                copied_set(
                    &ws.removed_ids_this_execution,
                    ids_target,
                    allocator.clone(),
                )
            })
            .transpose()?;
        let names = (names_target > ws.removed_names_this_execution.capacity())
            .then(|| {
                copied_set(
                    &ws.removed_names_this_execution,
                    names_target,
                    allocator.clone(),
                )
            })
            .transpose()?;
        let lifecycle = (lifecycle_target > ws.slot_lifecycle.capacity())
            .then(|| copied_map(&ws.slot_lifecycle, lifecycle_target, allocator.clone()))
            .transpose()?;
        let reconciliation = (reconciliation_target > ws.pending_var_reconciliation.capacity())
            .then(|| {
                copied_set(
                    &ws.pending_var_reconciliation,
                    reconciliation_target,
                    allocator.clone(),
                )
            })
            .transpose()?;
        let mut retired_names = FrameVec::new_in(allocator.clone());
        retired_names
            .try_reserve_exact(growth.retired_names)
            .map_err(|_| super::frame_assignment::failed())?;
        let mut retired_values = FrameVec::new_in(allocator.clone());
        retired_values
            .try_reserve_exact(growth.retired_values)
            .map_err(|_| super::frame_assignment::failed())?;
        let mut alias_retirements = FrameVec::new_in(allocator);
        alias_retirements
            .try_reserve_exact(growth.alias_retirements)
            .map_err(|_| super::frame_assignment::failed())?;
        Ok::<_, HostAllocationError>((
            slots,
            ids,
            names,
            lifecycle,
            reconciliation,
            retired_names,
            retired_values,
            alias_retirements,
        ))
    })
    .ok_or_else(unavailable)?
    .map_err(super::frame_allocation_error)?;
    let (
        removed_slots,
        removed_ids,
        removed_names,
        lifecycle,
        reconciliation,
        retired_names,
        retired_values,
        alias_retirements,
    ) = prepared;
    Ok(PreparedFrameLifecycle {
        frame,
        token,
        context,
        receipt,
        change,
        removed_slots,
        removed_ids,
        removed_names,
        lifecycle,
        reconciliation,
        retired_names: Some(retired_names),
        retired_values: Some(retired_values),
        alias_retirements: Some(alias_retirements),
    })
}

pub(crate) fn prepare_workspace_frame_clear(
    frame: WorkspaceFrameId,
    context: Option<&RuntimeContext>,
) -> Result<PreparedFrameLifecycle<'static>, RuntimeError> {
    prepare(frame, context, false, |_| LifecycleChange::Clear)
}

pub(crate) fn prepare_workspace_frame_remove(
    frame: WorkspaceFrameId,
    context: Option<&RuntimeContext>,
    name: &str,
) -> Result<PreparedFrameLifecycle<'static>, RuntimeError> {
    prepare(frame, context, false, |ws| {
        LifecycleChange::Remove(ws.names.get_key_value(name).map(|(name, _)| name.clone()))
    })
}

pub(crate) fn prepare_workspace_frame_unassign(
    frame: WorkspaceFrameId,
    context: Option<&RuntimeContext>,
    slot: usize,
) -> Result<PreparedFrameLifecycle<'static>, RuntimeError> {
    prepare(frame, context, false, |ws| {
        let name = ws.slot_lifecycle.get(&slot).map(|entry| match entry {
            SlotLifecycle::Assigned(name) | SlotLifecycle::Unassigned(name) => name.clone(),
        });
        LifecycleChange::Unassign(slot, name)
    })
}

pub(crate) fn prepare_workspace_frame_detach<'a>(
    frame: WorkspaceFrameId,
    context: Option<&RuntimeContext>,
    selection: &'a WorkspaceDetachSelection,
    detached: &'a DetachedWorkspaceValues,
) -> Result<PreparedFrameLifecycle<'a>, RuntimeError> {
    if !super::frame_is_detaching(frame) {
        return Err(unavailable());
    }
    prepare(frame, context, true, |_| {
        LifecycleChange::Detach(selection, detached)
    })
}
