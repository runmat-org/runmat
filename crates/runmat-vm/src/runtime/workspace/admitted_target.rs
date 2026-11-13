//! Admitted child-frame metadata for an exact dynamic workspace capture.
//!
//! The capture remains the mutation authority. This projection owns only the
//! child's derived name/slot indexes and retained read leases; it never copies
//! the language Values or publishes an independent workspace generation.

use std::alloc::Layout;

use runmat_runtime::context::host_allocation::{
    collection_block_retained_bytes, frame_hash_table_retained_bytes, FrameCollectionGate,
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use runmat_runtime::context::RuntimeContext;
use runmat_runtime::workspace::bindings::FrameGlobalBindings;
use runmat_runtime::workspace::{DynamicWorkspaceTargetCapture, WorkspaceName};
use runmat_runtime::RuntimeError;
use runmat_value::Value;

use super::frame_metadata::{empty_map, empty_set, FrameMutationPhase};
use super::{FrameMap, FrameSet, FrameVec, SlotLifecycle, WorkspaceFramePolicy, WorkspaceState};
use crate::interpreter::vars::{InterpreterSlot, InterpreterVars};

pub(crate) struct AdmittedDynamicInput {
    context: runmat_runtime::context::RuntimeContextLocalId,
    names: FrameMap<WorkspaceName, usize>,
    assigned: FrameSet<WorkspaceName>,
    ordered: FrameVec<WorkspaceName>,
    slots: InterpreterVars,
    aliases: FrameGlobalBindings,
    gate: FrameCollectionGate,
}

pub(crate) struct AdmittedWorkspaceMetadata {
    pub(crate) names: FrameMap<WorkspaceName, usize>,
    pub(crate) assigned: FrameSet<WorkspaceName>,
    pub(crate) aliases: FrameGlobalBindings,
    pub(crate) gate: FrameCollectionGate,
    // Retained only until compiler slot ordering is no longer needed. It is
    // moved through ingress so its claimed block retires outside frame borrows.
    pub(crate) ordered: FrameVec<WorkspaceName>,
}

pub(crate) struct AdmittedBindingView<'a> {
    input: &'a AdmittedDynamicInput,
    assigned_only: bool,
}

impl runmat_hir::WorkspaceBindingView for AdmittedBindingView<'_> {
    fn len(&self) -> usize {
        if self.assigned_only {
            self.input.assigned.len()
        } else {
            self.input.names.len()
        }
    }

    fn get(&self, name: &str) -> Option<usize> {
        if self.assigned_only && !self.input.assigned.contains(name) {
            return None;
        }
        self.input.names.get(name).copied()
    }

    fn visit(&self, visitor: &mut dyn FnMut(&str, usize)) {
        for (name, slot) in &self.input.names {
            if !self.assigned_only || self.input.assigned.contains(name.as_str()) {
                visitor(name.as_str(), *slot);
            }
        }
    }

    fn visit_in_slot_order(&self, visitor: &mut dyn FnMut(&str, usize)) {
        for (slot, name) in self.input.ordered.iter().enumerate() {
            if !self.assigned_only || self.input.assigned.contains(name.as_str()) {
                visitor(name.as_str(), slot);
            }
        }
    }
}

impl AdmittedDynamicInput {
    pub(crate) fn bindings(&self) -> AdmittedBindingView<'_> {
        AdmittedBindingView {
            input: self,
            assigned_only: false,
        }
    }

    pub(crate) fn assigned_bindings(&self) -> AdmittedBindingView<'_> {
        AdmittedBindingView {
            input: self,
            assigned_only: true,
        }
    }

    pub(crate) fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Prepare the compiler-private suffix before the admitted input becomes
    /// an interpreter frame. The source capture can only quote its live
    /// namespace; the compiler's final local extent is known later. Grow the
    /// same retained slot owner under its existing collection gate, then move
    /// the captured slots into the replacement without copying Values.
    pub(crate) fn prepare_compiled_extent(
        mut self,
        context: &RuntimeContext,
        extent: usize,
    ) -> Result<Self, RuntimeError> {
        if context.local_identity() != self.context {
            return Err(super::workspace_target_error(
                "compiled dynamic workspace frame belongs to another runtime context",
            ));
        }
        HostAllocationReceipt::policy(Some(context))
            .map_err(|error| super::frame_allocation_error_in_context(Some(context), error))?;
        if extent < self.slots.len() {
            return Err(super::workspace_target_error(
                "compiled dynamic workspace frame is smaller than its captured namespace",
            ));
        }
        if extent == self.slots.len() {
            return Ok(self);
        }
        if self.slots.capacity() >= extent {
            self.slots.resize_owned(extent, Value::Num(0.0));
            return Ok(self);
        }

        let retained = Layout::array::<InterpreterSlot>(extent)
            .map_err(|_| HostAllocationError::Overflow)
            .and_then(collection_block_retained_bytes)
            .map_err(|error| super::frame_allocation_error_in_context(Some(context), error))?;
        let plan = HostAllocationPlan::new(0, retained)
            .map_err(|error| super::frame_allocation_error_in_context(Some(context), error))?;
        let receipt = HostAllocationReceipt::admit(Some(context), plan)
            .map_err(|error| super::frame_allocation_error_in_context(Some(context), error))?;
        let installed = match &receipt {
            Some(receipt) => self.gate.install(context, receipt),
            None => self.gate.install_unbounded(),
        }
        .map_err(|error| super::frame_allocation_error_in_context(Some(context), error))?;
        let mut replacement = FrameVec::new_in(self.gate.allocator());
        replacement.try_reserve_exact(extent).map_err(|_| {
            super::frame_allocation_error_in_context(
                Some(context),
                HostAllocationError::Numeric(
                    runmat_value::ExactHostNumericAllocationError::AllocationFailed,
                ),
            )
        })?;
        if let Some(receipt) = &receipt {
            receipt
                .checkpoint(context)
                .map_err(|error| super::frame_allocation_error_in_context(Some(context), error))?;
        }

        let retired = self.slots.install_replacement(replacement);
        self.slots.resize_owned(extent, Value::Num(0.0));
        drop(installed);
        drop(retired);
        Ok(self)
    }

    pub(crate) fn into_execution_parts(self) -> (InterpreterVars, AdmittedWorkspaceMetadata) {
        (
            self.slots,
            AdmittedWorkspaceMetadata {
                names: self.names,
                assigned: self.assigned,
                aliases: self.aliases,
                gate: self.gate,
                ordered: self.ordered,
            },
        )
    }

    pub(crate) fn capture(
        context: &RuntimeContext,
        source: &DynamicWorkspaceTargetCapture,
    ) -> Result<Self, RuntimeError> {
        if source.context() != context.local_identity() {
            return Err(super::workspace_target_error(
                "captured dynamic target belongs to another runtime context",
            ));
        }
        let value_count = source.iter().count();
        let global_count = source.global_bindings().count();
        let maximum = value_count
            .checked_add(global_count)
            .ok_or(HostAllocationError::Overflow)
            .map_err(super::frame_allocation_error)?;
        let mut retained =
            FrameCollectionGate::retained_bytes().map_err(super::frame_allocation_error)?;
        for bytes in [
            frame_hash_table_retained_bytes::<WorkspaceName, usize>(maximum),
            frame_hash_table_retained_bytes::<WorkspaceName, ()>(value_count),
            Layout::array::<InterpreterSlot>(maximum)
                .map_err(|_| HostAllocationError::Overflow)
                .and_then(collection_block_retained_bytes),
            Layout::array::<WorkspaceName>(maximum)
                .map_err(|_| HostAllocationError::Overflow)
                .and_then(collection_block_retained_bytes),
        ] {
            retained = retained
                .checked_add(bytes.map_err(super::frame_allocation_error)?)
                .ok_or(HostAllocationError::Overflow)
                .map_err(super::frame_allocation_error)?;
        }
        for (name, _) in source.iter() {
            retained = quote_name(retained, name).map_err(super::frame_allocation_error)?;
        }
        for (name, _) in source.global_bindings() {
            if source.get(name).is_none() {
                retained = quote_name(retained, name).map_err(super::frame_allocation_error)?;
            }
        }
        let plan = HostAllocationPlan::new(0, retained).map_err(super::frame_allocation_error)?;
        let receipt = HostAllocationReceipt::admit(Some(context), plan)
            .map_err(super::frame_allocation_error)?;
        let gate = match &receipt {
            Some(receipt) => FrameCollectionGate::prepare(context, receipt),
            None => Ok(FrameCollectionGate::unbounded()),
        }
        .map_err(super::frame_allocation_error)?;
        let installed = match &receipt {
            Some(receipt) => gate.install(context, receipt),
            None => gate.install_unbounded(),
        }
        .map_err(super::frame_allocation_error)?;
        let allocator = gate.allocator();
        let mut names = empty_map(allocator.clone());
        let mut assigned = empty_set(allocator.clone());
        let mut slots = FrameVec::new_in(allocator);
        let mut ordered = FrameVec::new_in(gate.allocator());
        names
            .try_reserve(maximum)
            .map_err(|_| allocation_failed())?;
        assigned
            .try_reserve(value_count)
            .map_err(|_| allocation_failed())?;
        slots
            .try_reserve_exact(maximum)
            .map_err(|_| allocation_failed())?;
        ordered
            .try_reserve_exact(maximum)
            .map_err(|_| allocation_failed())?;
        let aliases =
            FrameGlobalBindings::prepare_empty(context).map_err(super::frame_allocation_error)?;
        for entry in source.iter_leases() {
            let name = prepare_name(context, receipt.as_ref(), entry.name())?;
            let slot = slots.len();
            slots.push(InterpreterSlot::retained(entry.into_value_lease()));
            assigned.insert(name.clone());
            names.insert(name.clone(), slot);
            ordered.push(name.clone());
            if let Some(target) = source.global_target(name.as_str()) {
                let alias = aliases
                    .prepare_target(context, slot, target)
                    .map_err(super::frame_allocation_error)?;
                alias.checkpoint().map_err(super::frame_allocation_error)?;
                alias.commit();
            }
        }
        for (name, target) in source.global_bindings() {
            if names.contains_key(name) {
                continue;
            }
            let owner = prepare_name(context, receipt.as_ref(), name)?;
            let slot = slots.len();
            slots.push(InterpreterSlot::Owned(Value::Num(0.0)));
            names.insert(owner.clone(), slot);
            ordered.push(owner);
            let alias = aliases
                .prepare_target(context, slot, target)
                .map_err(super::frame_allocation_error)?;
            alias.checkpoint().map_err(super::frame_allocation_error)?;
            alias.commit();
        }
        if let Some(receipt) = &receipt {
            receipt
                .checkpoint(context)
                .map_err(super::frame_allocation_error)?;
        }
        drop(installed);
        Ok(Self {
            context: source.context(),
            names,
            assigned,
            ordered,
            slots: InterpreterVars::from_retained(slots),
            aliases,
            gate,
        })
    }
}

fn quote_name(mut total: u64, name: &str) -> Result<u64, HostAllocationError> {
    let physical = WorkspaceName::requested_bytes(name).map_err(HostAllocationError::Numeric)?;
    let physical = u64::try_from(physical).map_err(|_| HostAllocationError::Overflow)?;
    total = total
        .checked_add(HostAllocationPlan::retained_exact_output_bytes(
            physical, 1,
        )?)
        .ok_or(HostAllocationError::Overflow)?;
    Ok(total)
}

fn prepare_name(
    context: &RuntimeContext,
    receipt: Option<&HostAllocationReceipt>,
    name: &str,
) -> Result<WorkspaceName, RuntimeError> {
    match receipt {
        Some(receipt) => receipt
            .with_output_allocation(context, |permit| WorkspaceName::prepare(name, permit))
            .map_err(super::frame_allocation_error),
        None => Ok(WorkspaceName::unbounded(name)),
    }
}

fn allocation_failed() -> RuntimeError {
    super::frame_allocation_error(HostAllocationError::Numeric(
        runmat_value::ExactHostNumericAllocationError::AllocationFailed,
    ))
}

pub(crate) fn build_frame_state(
    metadata: AdmittedWorkspaceMetadata,
    vars: &mut InterpreterVars,
    context: &RuntimeContext,
    policy: WorkspaceFramePolicy,
) -> Result<(WorkspaceState, Option<HostAllocationReceipt>), RuntimeError> {
    let AdmittedWorkspaceMetadata {
        names,
        assigned,
        aliases: _,
        gate,
        ordered,
    } = metadata;
    let mut retained =
        FrameMutationPhase::retained_bytes().map_err(super::frame_allocation_error)?;
    for bytes in [
        frame_hash_table_retained_bytes::<usize, SlotLifecycle>(ordered.len()),
        InterpreterVars::shared_control_retained_bytes(),
        if policy.captures_completion() {
            super::completion::CompletionHandle::retained_bytes()
        } else {
            Ok(0)
        },
    ] {
        retained = retained
            .checked_add(bytes.map_err(super::frame_allocation_error)?)
            .ok_or(HostAllocationError::Overflow)
            .map_err(super::frame_allocation_error)?;
    }
    let plan = HostAllocationPlan::new(0, retained).map_err(super::frame_allocation_error)?;
    let receipt =
        HostAllocationReceipt::admit(Some(context), plan).map_err(super::frame_allocation_error)?;
    let phase = match &receipt {
        Some(receipt) => FrameMutationPhase::prepare(context, receipt),
        None => Ok(FrameMutationPhase::unbounded()),
    }
    .map_err(super::frame_allocation_error)?;
    let installed = match &receipt {
        Some(receipt) => gate.install(context, receipt),
        None => gate.install_unbounded(),
    }
    .map_err(super::frame_allocation_error)?;
    let allocator = gate.allocator();
    let mut lifecycle = empty_map(allocator.clone());
    lifecycle
        .try_reserve(ordered.len())
        .map_err(|_| allocation_failed())?;
    let frame_values = vars
        .promote_shared(Some(context), receipt.as_ref())
        .map_err(super::frame_allocation_error)?;
    for (slot, name) in ordered.iter().enumerate() {
        lifecycle.insert(
            slot,
            if assigned.contains(name.as_str()) {
                SlotLifecycle::Assigned(name.clone())
            } else {
                SlotLifecycle::Unassigned(name.clone())
            },
        );
    }
    if let Some(receipt) = &receipt {
        receipt
            .checkpoint(context)
            .map_err(super::frame_allocation_error)?;
    }
    drop(installed);
    drop(ordered);
    Ok((
        WorkspaceState {
            names,
            assigned,
            assigned_names_this_execution: empty_set(allocator.clone()),
            assigned_ids_this_execution: empty_set(allocator.clone()),
            removed_slots_this_execution: empty_map(allocator.clone()),
            removed_ids_this_execution: empty_set(allocator.clone()),
            removed_names_this_execution: empty_set(allocator.clone()),
            slot_lifecycle: lifecycle,
            values: frame_values,
            pending_var_reconciliation: empty_set(allocator),
            collection_gate: gate,
            mutation_phase: phase,
        },
        receipt,
    ))
}
