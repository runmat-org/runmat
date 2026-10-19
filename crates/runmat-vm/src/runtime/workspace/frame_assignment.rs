//! Pre-admitted frame value writes and named binding updates.
//!
//! Preparation may allocate. Publication and final frame commit do not.
//! A phase token fences same-frame reentry while the Runtime delegate runs
//! without a borrow of the VM frame stack.

use super::frame::{with_frame, with_frame_mut_unlocked, WorkspaceFrameId};
use super::frame_metadata::{
    empty_map, empty_set, FrameMap, FrameMutationToken, FrameSet, FrameVec,
};
use super::{find_unassigned_slot_for_name, SlotLifecycle, WorkspaceOwner};
use crate::interpreter::vars::{InterpreterSlot, InterpreterVarsRetirement};
use runmat_runtime::context::host_allocation::{
    collection_block_retained_bytes, frame_hash_table_retained_bytes, HostAllocationError,
    HostAllocationPlan, HostAllocationReceipt, OperationCollectionAllocator,
};
use runmat_runtime::workspace::{WorkspaceName, WorkspaceSlotReadLease};
use runmat_runtime::{context::RuntimeContext, RuntimeError};
use runmat_value::{ExactHostNumericAllocationError, Value};
use std::alloc::Layout;
use std::hash::Hash;

fn unavailable() -> RuntimeError {
    runmat_runtime::runtime_error::semantic_error(
        "RunMat:Workspace:FrameUnavailable",
        "workspace frame is closed or a frame mutation is active",
    )
}

fn allocation(error: HostAllocationError) -> RuntimeError {
    super::frame_allocation_error(error)
}

pub(super) fn failed() -> HostAllocationError {
    HostAllocationError::Numeric(ExactHostNumericAllocationError::AllocationFailed)
}

pub(super) fn add(total: &mut u64, next: u64) -> Result<(), HostAllocationError> {
    *total = total
        .checked_add(next)
        .ok_or(HostAllocationError::Overflow)?;
    Ok(())
}

pub(super) fn map_growth<K: Eq + Hash, V>(
    len: usize,
    capacity: usize,
    insert: bool,
) -> Result<Option<u64>, HostAllocationError> {
    let target = len
        .checked_add(usize::from(insert))
        .ok_or(HostAllocationError::Overflow)?;
    (target > capacity)
        .then(|| frame_hash_table_retained_bytes::<K, V>(target))
        .transpose()
}

pub(super) fn vector_growth<T>(
    target: usize,
    capacity: usize,
) -> Result<Option<u64>, HostAllocationError> {
    if target <= capacity {
        return Ok(None);
    }
    let layout = Layout::array::<T>(target).map_err(|_| HostAllocationError::Overflow)?;
    collection_block_retained_bytes(layout).map(Some)
}

pub(super) fn copied_map<K: Eq + Hash + Clone, V: Clone>(
    source: &FrameMap<K, V>,
    target: usize,
    allocator: OperationCollectionAllocator,
) -> Result<FrameMap<K, V>, HostAllocationError> {
    let mut replacement = empty_map(allocator);
    replacement.try_reserve(target).map_err(|_| failed())?;
    replacement.extend(
        source
            .iter()
            .map(|(key, value)| (key.clone(), value.clone())),
    );
    Ok(replacement)
}

pub(super) fn copied_set<K: Eq + Hash + Clone>(
    source: &FrameSet<K>,
    target: usize,
    allocator: OperationCollectionAllocator,
) -> Result<FrameSet<K>, HostAllocationError> {
    let mut replacement = empty_set(allocator);
    replacement.try_reserve(target).map_err(|_| failed())?;
    replacement.extend(source.iter().cloned());
    Ok(replacement)
}

pub(crate) struct PreparedFrameAssignment {
    frame: WorkspaceFrameId,
    token: FrameMutationToken,
    receipt: Option<HostAllocationReceipt>,
    context: Option<RuntimeContext>,
    name: Option<WorkspaceName>,
    slot: usize,
    value: InterpreterSlot,
    reconcile_var: bool,
    report_assignment: bool,
    names: Option<FrameMap<WorkspaceName, usize>>,
    assigned: Option<FrameSet<WorkspaceName>>,
    assigned_names: Option<FrameSet<WorkspaceName>>,
    assigned_ids: Option<FrameSet<usize>>,
    lifecycle: Option<FrameMap<usize, SlotLifecycle>>,
    reconciliation: Option<FrameSet<usize>>,
    values: Option<FrameVec<InterpreterSlot>>,
}

pub(crate) struct FrameAssignmentRetirement {
    _names: Option<FrameMap<WorkspaceName, usize>>,
    _assigned: Option<FrameSet<WorkspaceName>>,
    _assigned_names: Option<FrameSet<WorkspaceName>>,
    _assigned_ids: Option<FrameSet<usize>>,
    _lifecycle: Option<FrameMap<usize, SlotLifecycle>>,
    _reconciliation: Option<FrameSet<usize>>,
    _value_backing: Option<InterpreterVarsRetirement>,
    _value: Option<InterpreterSlot>,
    _names_displaced_by_lifecycle: [Option<WorkspaceName>; 3],
}

impl FrameAssignmentRetirement {
    fn empty() -> Self {
        Self {
            _names: None,
            _assigned: None,
            _assigned_names: None,
            _assigned_ids: None,
            _lifecycle: None,
            _reconciliation: None,
            _value_backing: None,
            _value: None,
            _names_displaced_by_lifecycle: [None, None, None],
        }
    }
}

impl PreparedFrameAssignment {
    pub(crate) fn slot(&self) -> usize {
        self.slot
    }

    pub(crate) fn checkpoint_before_publication(&self) -> Result<(), RuntimeError> {
        if !self.token.is_prepared() {
            return Err(unavailable());
        }
        if let (Some(context), Some(receipt)) = (&self.context, &self.receipt) {
            receipt.checkpoint(context).map_err(allocation)?;
        }
        if !self.token.begin_publication() {
            return Err(unavailable());
        }
        Ok(())
    }

    /// Install the already-validated assignment without allocating, invoking
    /// callbacks, or retiring a displaced owner. A lexical coordinator can
    /// call this while holding its short mutation guard and retire the result
    /// only after both authorities are coherent.
    fn install_prevalidated_inner(&mut self) -> FrameAssignmentRetirement {
        assert!(
            self.token.is_publishing(),
            "prepared frame publication was not begun"
        );
        with_frame_mut_unlocked(self.frame, |frame| {
            let ws = &mut frame.state;
            let mut retired = FrameAssignmentRetirement::empty();
            if let Some(replacement) = self.names.take() {
                retired._names = Some(std::mem::replace(&mut ws.names, replacement));
            }
            if let Some(replacement) = self.assigned.take() {
                retired._assigned = Some(std::mem::replace(&mut ws.assigned, replacement));
            }
            if let Some(replacement) = self.assigned_names.take() {
                retired._assigned_names = Some(std::mem::replace(
                    &mut ws.assigned_names_this_execution,
                    replacement,
                ));
            }
            if let Some(replacement) = self.assigned_ids.take() {
                retired._assigned_ids = Some(std::mem::replace(
                    &mut ws.assigned_ids_this_execution,
                    replacement,
                ));
            }
            if let Some(replacement) = self.lifecycle.take() {
                retired._lifecycle = Some(std::mem::replace(&mut ws.slot_lifecycle, replacement));
            }
            if let Some(replacement) = self.reconciliation.take() {
                retired._reconciliation = Some(std::mem::replace(
                    &mut ws.pending_var_reconciliation,
                    replacement,
                ));
            }
            if let Some(replacement) = self.values.take() {
                retired._value_backing = Some(ws.values.install_replacement(replacement));
            }
            if ws.values.len() <= self.slot {
                ws.values.resize_owned(self.slot + 1, Value::Num(0.0));
            }
            let value = std::mem::replace(&mut self.value, InterpreterSlot::Owned(Value::Num(0.0)));
            retired._value = match value {
                InterpreterSlot::Owned(value) => ws.values.replace_owned(self.slot, value),
                InterpreterSlot::Retained(value) => ws.values.replace_retained(self.slot, value),
                InterpreterSlot::Captured(value) => ws.values.replace_captured(self.slot, value),
            };
            if let Some(name) = self.name.take() {
                if let Some(slot) = ws.names.get_mut(name.as_str()) {
                    *slot = self.slot;
                } else {
                    ws.names.insert(name.clone(), self.slot);
                }
                if self.reconcile_var && !ws.pending_var_reconciliation.contains(&self.slot) {
                    ws.pending_var_reconciliation.insert(self.slot);
                }
                if !ws.assigned.contains(name.as_str()) {
                    ws.assigned.insert(name.clone());
                }
                if self.report_assignment {
                    if !ws.assigned_names_this_execution.contains(name.as_str()) {
                        ws.assigned_names_this_execution.insert(name.clone());
                    }
                    if !ws.assigned_ids_this_execution.contains(&self.slot) {
                        ws.assigned_ids_this_execution.insert(self.slot);
                    }
                }
                retired._names_displaced_by_lifecycle =
                    super::mark_slot_assigned_retaining(ws, self.slot, &name);
                // Keep the prepared name owner alive until after the frame
                // borrow and mutation phase have both been released.
                self.name = Some(name);
            }
            retired
        })
        .expect("a prepared frame is held until its publication commits")
    }

    /// After the delegate is visible, no allocation or fallible validation is
    /// allowed here. The token keeps the exact frame alive until this returns.
    pub(crate) fn commit(mut self) {
        let retired = self.install_prevalidated_inner();
        // Release the mutation phase before retiring GC/foreign owners so a
        // destructor may safely re-enter the now-coherent frame.
        drop(self);
        drop(retired);
    }

    /// Persistent declaration obtains its final value from the session owner
    /// only after preparation. Collection metadata is already admitted; the
    /// Value alias remains a separate value-owner admission boundary.
    pub(crate) fn commit_owned_value(mut self, value: Value) {
        self.value = InterpreterSlot::Owned(value);
        self.commit();
    }
}

impl runmat_runtime::workspace::WorkspaceCompletionParticipant for PreparedFrameAssignment {
    type Retirement = FrameAssignmentRetirement;

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
        self.install_prevalidated_inner()
    }
}

pub(crate) fn prepare_workspace_frame_assignment(
    frame: WorkspaceFrameId,
    context: Option<&RuntimeContext>,
    name: &str,
    value: &Value,
    reconcile_var: bool,
) -> Result<PreparedFrameAssignment, RuntimeError> {
    prepare_frame_value_write(frame, context, name, None, value, reconcile_var, true)
}

pub(crate) fn prepare_workspace_frame_assignment_at_slot(
    frame: WorkspaceFrameId,
    context: Option<&RuntimeContext>,
    slot: usize,
    name: &str,
    value: &Value,
    reconcile_var: bool,
) -> Result<PreparedFrameAssignment, RuntimeError> {
    prepare_frame_value_write(frame, context, name, Some(slot), value, reconcile_var, true)
}

pub(crate) fn prepare_workspace_frame_assignment_lease_at_slot(
    frame: WorkspaceFrameId,
    context: &RuntimeContext,
    slot: usize,
    name: &str,
    value: WorkspaceSlotReadLease,
    reconcile_var: bool,
) -> Result<PreparedFrameAssignment, RuntimeError> {
    prepare_frame_slot_write(
        frame,
        Some(context),
        name,
        Some(slot),
        InterpreterSlot::Retained(value),
        reconcile_var,
        true,
    )
}

pub(crate) fn prepare_workspace_frame_assignment_lease(
    frame: WorkspaceFrameId,
    context: &RuntimeContext,
    name: &str,
    value: WorkspaceSlotReadLease,
    reconcile_var: bool,
) -> Result<PreparedFrameAssignment, RuntimeError> {
    prepare_frame_slot_write(
        frame,
        Some(context),
        name,
        None,
        InterpreterSlot::Retained(value),
        reconcile_var,
        true,
    )
}

/// Refresh an already-declared lexical or persistent slot without reporting
/// a language-visible assignment. The selected slot, not a name lookup, is
/// the authority for this import.
pub(crate) fn prepare_workspace_frame_import(
    frame: WorkspaceFrameId,
    context: Option<&RuntimeContext>,
    slot: usize,
    value: &Value,
) -> Result<PreparedFrameAssignment, RuntimeError> {
    let name = with_frame(frame, |frame| {
        frame.state.slot_lifecycle.get(&slot).map(|entry| {
            let name = match entry {
                SlotLifecycle::Assigned(name) | SlotLifecycle::Unassigned(name) => name.clone(),
            };
            (name, frame.state.names.get(entry.name()).copied())
        })
    })
    .ok_or_else(unavailable)?;
    name.map(|(name, bound)| {
        if bound.is_some_and(|bound| bound != slot) {
            return Err(runmat_runtime::runtime_error::semantic_error(
                "RunMat:Workspace:LexicalSlot",
                "shared binding name resolves to another workspace slot",
            ));
        }
        prepare_frame_value_write(frame, context, &name, Some(slot), value, false, false)
    })
    .unwrap_or_else(|| prepare_anonymous_import(frame, context, slot, value))
}

pub(crate) fn prepare_workspace_frame_import_lease(
    frame: WorkspaceFrameId,
    context: &RuntimeContext,
    slot: usize,
    value: runmat_runtime::call::lexical::LexicalBindingReadLease,
) -> Result<PreparedFrameAssignment, RuntimeError> {
    let name = with_frame(frame, |frame| {
        frame.state.slot_lifecycle.get(&slot).map(|entry| {
            let name = match entry {
                SlotLifecycle::Assigned(name) | SlotLifecycle::Unassigned(name) => name.clone(),
            };
            (name, frame.state.names.get(entry.name()).copied())
        })
    })
    .ok_or_else(unavailable)?;
    let value = InterpreterSlot::Retained(value.into());
    match name {
        Some((name, bound)) => {
            if bound.is_some_and(|bound| bound != slot) {
                return Err(runmat_runtime::runtime_error::semantic_error(
                    "RunMat:Workspace:LexicalSlot",
                    "shared binding name resolves to another workspace slot",
                ));
            }
            prepare_frame_slot_write(frame, Some(context), &name, Some(slot), value, false, false)
        }
        None => prepare_anonymous_import_slot(frame, Some(context), slot, value),
    }
}

fn prepare_anonymous_import(
    frame: WorkspaceFrameId,
    context: Option<&RuntimeContext>,
    slot: usize,
    value: &Value,
) -> Result<PreparedFrameAssignment, RuntimeError> {
    prepare_anonymous_import_slot(frame, context, slot, InterpreterSlot::Owned(value.clone()))
}

fn prepare_anonymous_import_slot(
    frame: WorkspaceFrameId,
    context: Option<&RuntimeContext>,
    slot: usize,
    value: InterpreterSlot,
) -> Result<PreparedFrameAssignment, RuntimeError> {
    if super::frame_is_detaching(frame) {
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
    let target = slot
        .checked_add(1)
        .ok_or_else(|| allocation(HostAllocationError::Overflow))?;
    let quote = with_frame(frame, |frame| {
        if frame.state.values.uses_owned_storage() {
            vector_growth::<InterpreterSlot>(frame.state.values.len().max(target), 0)
        } else {
            vector_growth::<InterpreterSlot>(target, frame.state.values.capacity())
        }
    })
    .ok_or_else(unavailable)?
    .map_err(allocation)?
    .unwrap_or(0);
    let receipt = HostAllocationReceipt::admit(
        context.as_ref(),
        HostAllocationPlan::new(0, quote).map_err(allocation)?,
    )
    .map_err(allocation)?;
    let values = with_frame(frame, |frame| {
        let ws = &frame.state;
        let _installed = match (context.as_ref(), receipt.as_ref()) {
            (Some(context), Some(receipt)) => ws.collection_gate.install(context, receipt)?,
            _ => ws.collection_gate.install_unbounded()?,
        };
        let target = ws.values.len().max(target);
        if !ws.values.uses_owned_storage() && target <= ws.values.capacity() {
            return Ok::<_, HostAllocationError>(None);
        }
        let mut replacement = FrameVec::new_in(ws.collection_gate.allocator());
        replacement
            .try_reserve_exact(target)
            .map_err(|_| failed())?;
        Ok(Some(replacement))
    })
    .ok_or_else(unavailable)?
    .map_err(allocation)?;
    Ok(PreparedFrameAssignment {
        frame,
        token,
        context,
        receipt,
        name: None,
        slot,
        value,
        reconcile_var: false,
        report_assignment: false,
        names: None,
        assigned: None,
        assigned_names: None,
        assigned_ids: None,
        lifecycle: None,
        reconciliation: None,
        values,
    })
}

fn prepare_frame_value_write(
    frame: WorkspaceFrameId,
    context: Option<&RuntimeContext>,
    name: &str,
    fixed_slot: Option<usize>,
    value: &Value,
    reconcile_var: bool,
    report_assignment: bool,
) -> Result<PreparedFrameAssignment, RuntimeError> {
    prepare_frame_slot_write(
        frame,
        context,
        name,
        fixed_slot,
        InterpreterSlot::Owned(value.clone()),
        reconcile_var,
        report_assignment,
    )
}

fn prepare_frame_slot_write(
    frame: WorkspaceFrameId,
    context: Option<&RuntimeContext>,
    name: &str,
    fixed_slot: Option<usize>,
    value: InterpreterSlot,
    reconcile_var: bool,
    report_assignment: bool,
) -> Result<PreparedFrameAssignment, RuntimeError> {
    let owner = context.map_or_else(super::current_owner, |context| {
        WorkspaceOwner::Runtime(context.local_identity())
    });
    if super::frame_is_detaching(frame) {
        return Err(unavailable());
    }
    let token = with_frame(frame, |selected| {
        selected.state.mutation_phase.acquire(owner, frame)
    })
    .flatten()
    .ok_or_else(unavailable)?;
    if let Some(slot) = fixed_slot {
        let matches = with_frame(frame, |frame| {
            let ws = &frame.state;
            ws.names.get(name).is_none_or(|mapped| *mapped == slot)
                && ws
                    .slot_lifecycle
                    .get(&slot)
                    .is_none_or(|existing| existing.name() == name)
        })
        .ok_or_else(unavailable)?;
        if !matches {
            return Err(runmat_runtime::runtime_error::semantic_error(
                "RunMat:Workspace:GlobalSlot",
                "global declaration conflicts with an existing compiler slot",
            ));
        }
    }
    let context = context
        .cloned()
        .or_else(runmat_runtime::context::legacy::active);
    let (slot, quote, needs_name) = with_frame(frame, |frame| {
        let ws = &frame.state;
        let slot = fixed_slot.unwrap_or_else(|| {
            ws.names
                .get(name)
                .copied()
                .or_else(|| find_unassigned_slot_for_name(ws, name))
                .unwrap_or(ws.values.len())
        });
        let needs_name = ws.names.get_key_value(name).is_none()
            && ws
                .slot_lifecycle
                .get(&slot)
                .is_none_or(|entry| entry.name() != name);
        let mut quote = 0u64;
        for bytes in [
            map_growth::<WorkspaceName, usize>(
                ws.names.len(),
                ws.names.capacity(),
                !ws.names.contains_key(name),
            )?,
            map_growth::<WorkspaceName, ()>(
                ws.assigned.len(),
                ws.assigned.capacity(),
                !ws.assigned.contains(name),
            )?,
            map_growth::<WorkspaceName, ()>(
                ws.assigned_names_this_execution.len(),
                ws.assigned_names_this_execution.capacity(),
                report_assignment && !ws.assigned_names_this_execution.contains(name),
            )?,
            map_growth::<usize, ()>(
                ws.assigned_ids_this_execution.len(),
                ws.assigned_ids_this_execution.capacity(),
                report_assignment && !ws.assigned_ids_this_execution.contains(&slot),
            )?,
            map_growth::<usize, SlotLifecycle>(
                ws.slot_lifecycle.len(),
                ws.slot_lifecycle.capacity(),
                !ws.slot_lifecycle.contains_key(&slot),
            )?,
            map_growth::<usize, ()>(
                ws.pending_var_reconciliation.len(),
                ws.pending_var_reconciliation.capacity(),
                reconcile_var && !ws.pending_var_reconciliation.contains(&slot),
            )?,
            if ws.values.uses_owned_storage() {
                vector_growth::<InterpreterSlot>(
                    ws.values
                        .len()
                        .max(slot.checked_add(1).ok_or(HostAllocationError::Overflow)?),
                    0,
                )?
            } else {
                vector_growth::<InterpreterSlot>(
                    slot.checked_add(1).ok_or(HostAllocationError::Overflow)?,
                    ws.values.capacity(),
                )?
            },
        ]
        .into_iter()
        .flatten()
        {
            add(&mut quote, bytes)?;
        }
        if needs_name {
            let physical = u64::try_from(
                WorkspaceName::requested_bytes(name).map_err(HostAllocationError::Numeric)?,
            )
            .map_err(|_| HostAllocationError::Overflow)?;
            add(
                &mut quote,
                HostAllocationPlan::retained_exact_output_bytes(physical, 1)?,
            )?;
        }
        Ok::<_, HostAllocationError>((slot, quote, needs_name))
    })
    .ok_or_else(unavailable)?
    .map_err(allocation)?;
    let plan = HostAllocationPlan::new(0, quote).map_err(allocation)?;
    let receipt = HostAllocationReceipt::admit(context.as_ref(), plan).map_err(allocation)?;
    let prepared = with_frame(frame, |frame| {
        let ws = &frame.state;
        let allocator = ws.collection_gate.allocator();
        let _installed = match (context.as_ref(), receipt.as_ref()) {
            (Some(context), Some(receipt)) => ws.collection_gate.install(context, receipt)?,
            _ => ws.collection_gate.install_unbounded()?,
        };
        let canonical = ws
            .names
            .get_key_value(name)
            .map(|(name, _)| name.clone())
            .or_else(|| {
                ws.slot_lifecycle.get(&slot).and_then(|entry| {
                    (entry.name() == name).then(|| match entry {
                        SlotLifecycle::Assigned(name) | SlotLifecycle::Unassigned(name) => {
                            name.clone()
                        }
                    })
                })
            })
            .map(Ok)
            .unwrap_or_else(|| match (context.as_ref(), receipt.as_ref()) {
                (Some(context), Some(receipt)) => receipt
                    .with_output_allocation(context, |permit| WorkspaceName::prepare(name, permit)),
                _ => Ok(WorkspaceName::unbounded(name)),
            })?;
        let names = (ws.names.len() + usize::from(!ws.names.contains_key(name))
            > ws.names.capacity())
        .then(|| copied_map(&ws.names, ws.names.len() + 1, allocator.clone()))
        .transpose()?;
        let assigned = (ws.assigned.len() + usize::from(!ws.assigned.contains(name))
            > ws.assigned.capacity())
        .then(|| copied_set(&ws.assigned, ws.assigned.len() + 1, allocator.clone()))
        .transpose()?;
        let assigned_names = (report_assignment
            && ws.assigned_names_this_execution.len()
                + usize::from(!ws.assigned_names_this_execution.contains(name))
                > ws.assigned_names_this_execution.capacity())
        .then(|| {
            copied_set(
                &ws.assigned_names_this_execution,
                ws.assigned_names_this_execution.len() + 1,
                allocator.clone(),
            )
        })
        .transpose()?;
        let assigned_ids = (report_assignment
            && ws.assigned_ids_this_execution.len()
                + usize::from(!ws.assigned_ids_this_execution.contains(&slot))
                > ws.assigned_ids_this_execution.capacity())
        .then(|| {
            copied_set(
                &ws.assigned_ids_this_execution,
                ws.assigned_ids_this_execution.len() + 1,
                allocator.clone(),
            )
        })
        .transpose()?;
        let lifecycle = (ws.slot_lifecycle.len()
            + usize::from(!ws.slot_lifecycle.contains_key(&slot))
            > ws.slot_lifecycle.capacity())
        .then(|| {
            copied_map(
                &ws.slot_lifecycle,
                ws.slot_lifecycle.len() + 1,
                allocator.clone(),
            )
        })
        .transpose()?;
        let reconciliation = (ws.pending_var_reconciliation.len()
            + usize::from(reconcile_var && !ws.pending_var_reconciliation.contains(&slot))
            > ws.pending_var_reconciliation.capacity())
        .then(|| {
            copied_set(
                &ws.pending_var_reconciliation,
                ws.pending_var_reconciliation.len() + 1,
                allocator.clone(),
            )
        })
        .transpose()?;
        let target = ws.values.len().max(slot + 1);
        let values = if ws.values.uses_owned_storage() || target > ws.values.capacity() {
            let mut replacement: FrameVec<InterpreterSlot> = FrameVec::new_in(allocator);
            replacement
                .try_reserve_exact(target)
                .map_err(|_| failed())?;
            Some(replacement)
        } else {
            None
        };
        Ok::<_, HostAllocationError>((
            canonical,
            names,
            assigned,
            assigned_names,
            assigned_ids,
            lifecycle,
            reconciliation,
            values,
        ))
    })
    .ok_or_else(unavailable)?
    .map_err(allocation)?;
    let (name, names, assigned, assigned_names, assigned_ids, lifecycle, reconciliation, values) =
        prepared;
    let _ = needs_name;
    Ok(PreparedFrameAssignment {
        frame,
        token,
        receipt,
        context,
        name: Some(name),
        slot,
        value,
        reconcile_var,
        report_assignment,
        names,
        assigned,
        assigned_names,
        assigned_ids,
        lifecycle,
        reconciliation,
        values,
    })
}
