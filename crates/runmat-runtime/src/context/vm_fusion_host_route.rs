//! Request-local evidence for an installed VM region selected to run on host.
//!
//! This is not a GPU execution prohibition. It is a one-region, one-instruction
//! proof that an admitted host branch actually ran. Missing proof only retires
//! comparable CPU feedback; ordinary execution keeps its existing route.

use super::{
    RuntimeContext, RuntimeContextLocalId, RuntimePlacementService, VmFusionRegionRegistration,
};
use runmat_builtins::BuiltinBindingIdentity;
use runmat_execution::cancellation::CancellationToken;
use runmat_execution::ProgramRevision;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Weak};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VmFusionHostInstruction {
    Structural,
    ElementwiseBinary,
    ElementwiseCall,
    Intrinsic(runmat_types::OperatorKind),
    Reduction,
    Unproved,
}

/// Numeric intrinsic execution is not a named builtin binding. Keep its
/// language operator identity distinct from canonical dispatch dependencies.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub enum VmFusionHostOperation {
    Builtin(BuiltinBindingIdentity),
    Intrinsic(runmat_types::OperatorKind),
}

#[derive(Clone)]
pub struct VmFusionHostRoute {
    backing: Rc<RouteBacking>,
}

// This proof travels in every invocation context, including unfused calls.
// Keep the carrier pointer-sized; facts and progress belong to its region.
const _: () =
    assert!(std::mem::size_of::<VmFusionHostRoute>() == std::mem::size_of::<Rc<RouteBacking>>());

// Immutable admission facts and mutable progress have one existing region
// owner. Cloning a route does not copy its registration/revocation metadata.
struct RouteBacking {
    origin: VmFusionHostOrigin,
    expected_operations: Box<[VmFusionHostOperation]>,
    state: RefCell<RouteState>,
}

#[derive(Clone)]
pub(super) struct VmFusionHostOrigin {
    pub registration: VmFusionRegionRegistration,
    pub owner: RuntimeContextLocalId,
    pub program: Option<Arc<ProgramRevision>>,
    pub cancellation: CancellationToken,
    pub placement_owner: std::rc::Weak<dyn RuntimePlacementService>,
    pub issuer_live: Weak<()>,
    pub generation: Arc<dyn super::RuntimeVmFusionGenerationPermit>,
}

struct RouteState {
    pending: Option<(usize, VmFusionHostInstruction, bool)>,
    invalid: bool,
    next_pc: usize,
    terminal_confirmed: bool,
    numerical_outputs: Option<super::VmFusionHostOutputs>,
    retirement_active: bool,
}

impl VmFusionHostRoute {
    pub(super) fn new(
        origin: VmFusionHostOrigin,
        expected_operations: Vec<VmFusionHostOperation>,
    ) -> Option<Self> {
        if expected_operations.is_empty() {
            return None;
        }
        Some(Self {
            backing: Rc::new(RouteBacking {
                state: RefCell::new(RouteState {
                    pending: None,
                    invalid: false,
                    next_pc: origin.registration.span_start(),
                    terminal_confirmed: false,
                    numerical_outputs: None,
                    retirement_active: false,
                }),
                origin,
                expected_operations: expected_operations.into_boxed_slice(),
            }),
        })
    }

    pub fn contains_pc(&self, pc: usize) -> bool {
        self.backing.origin.registration.contains_pc(pc)
    }

    pub fn current_for(&self, runtime: &RuntimeContext) -> bool {
        let same_placement_owner = self
            .backing
            .origin
            .placement_owner
            .upgrade()
            .zip(runtime.service_ports().placement())
            .is_some_and(|(selected, current)| Rc::ptr_eq(&selected, current));
        runtime.local_identity() == self.backing.origin.owner
            && runtime.program_revision() == self.backing.origin.program.as_deref()
            && runtime
                .cancellation()
                .same_generation(&self.backing.origin.cancellation)
            && same_placement_owner
            && !runtime.is_cancelled()
            && self.backing.origin.issuer_live.upgrade().is_some()
            && self.backing.origin.generation.is_current()
    }

    pub fn install_numerical_outputs(
        &self,
        runtime: &RuntimeContext,
        outputs: super::VmFusionHostOutputs,
    ) -> Result<(), crate::RuntimeError> {
        if !self.current_for(runtime) {
            return Err(super::vm_fusion_host_outputs::missing_output(runtime));
        }
        let mut state = self.backing.state.borrow_mut();
        if state.pending.is_some()
            || state.next_pc != self.backing.origin.registration.span_start()
            || state.numerical_outputs.is_some()
            || state.retirement_active
        {
            state.invalid = true;
            return Err(crate::build_runtime_error(
                "VM host numerical outputs were installed outside their selected region boundary",
            )
            .with_identifier("RunMat:Placement:InvalidDecision")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build());
        }
        state.numerical_outputs = Some(outputs);
        Ok(())
    }

    pub fn take_numerical_output(
        &self,
        runtime: &RuntimeContext,
        instruction: VmFusionHostInstruction,
        binding: BuiltinBindingIdentity,
    ) -> Result<Option<crate::context::preparation::PreparedHostNumericalOutput>, crate::RuntimeError>
    {
        self.take_selected_numerical_output(
            runtime,
            instruction,
            VmFusionHostOperation::Builtin(binding),
        )
    }

    pub fn take_intrinsic_numerical_output(
        &self,
        runtime: &RuntimeContext,
        operator: runmat_types::OperatorKind,
    ) -> Result<Option<crate::context::preparation::PreparedHostNumericalOutput>, crate::RuntimeError>
    {
        if !self.allows_intrinsic_host(runtime, operator) {
            return Err(super::vm_fusion_host_outputs::missing_output(runtime));
        }
        self.take_selected_numerical_output(
            runtime,
            VmFusionHostInstruction::Intrinsic(operator),
            VmFusionHostOperation::Intrinsic(operator),
        )
    }

    fn take_selected_numerical_output(
        &self,
        runtime: &RuntimeContext,
        instruction: VmFusionHostInstruction,
        operation: VmFusionHostOperation,
    ) -> Result<Option<crate::context::preparation::PreparedHostNumericalOutput>, crate::RuntimeError>
    {
        if !self.current_for(runtime) {
            return Err(super::vm_fusion_host_outputs::missing_output(runtime));
        }
        let mut outputs = {
            let mut state = self.backing.state.borrow_mut();
            if state.invalid
                || state.retirement_active
                || !state
                    .pending
                    .is_some_and(|(_, expected, _)| self.instruction_matches(expected, instruction))
                || !self.backing.expected_operations.contains(&operation)
            {
                state.invalid = true;
                return Err(super::vm_fusion_host_outputs::invalid_output());
            }
            let Some(outputs) = state.numerical_outputs.take() else {
                drop(state);
                return match super::host_allocation::HostAllocationReceipt::policy(Some(runtime)) {
                    Ok(super::host_allocation::HostAllocationPolicy::Unbounded) => Ok(None),
                    _ => Err(super::vm_fusion_host_outputs::missing_output(runtime)),
                };
            };
            state.retirement_active = true;
            outputs
        };
        // Binding a preallocated Tensor can retire a failed owner. Do this
        // outside the route's RefCell borrow, just like operand retirement.
        let result = match operation {
            VmFusionHostOperation::Builtin(binding) => {
                outputs.take_builtin(runtime, instruction, binding)
            }
            VmFusionHostOperation::Intrinsic(operator) => {
                outputs.take_intrinsic(runtime, instruction, operator)
            }
        };
        let mut state = self.backing.state.borrow_mut();
        state.retirement_active = false;
        if result.is_err() {
            state.invalid = true;
        }
        state.numerical_outputs = Some(outputs);
        result.map(Some)
    }
    /// Return a completed operation's moved source operands as one batch.
    /// The carrier is temporarily moved out of the RefCell so redundant
    /// aliases and retired claims drop outside the route's mutable borrow.
    pub fn retire_numerical_operands(
        &self,
        runtime: &RuntimeContext,
        instruction: VmFusionHostInstruction,
        operands: [Option<runmat_value::Value>; 2],
    ) -> Result<(), crate::RuntimeError> {
        self.retire_selected_numerical_operands(
            runtime,
            Some(instruction),
            move |outputs, pending| outputs.retire_numerical_operands(pending, operands),
        )
    }

    /// A named builtin returns its original owned argument vector after the
    /// awaited result is rooted. The entry consumed by `take_numerical_output`
    /// already authenticated the binding; retirement verifies that same
    /// source-ordered entry without relying on transient selected-binding TLS.
    pub fn retire_builtin_operands(
        &self,
        runtime: &RuntimeContext,
        operands: Vec<runmat_value::Value>,
    ) -> Result<(), crate::RuntimeError> {
        self.retire_selected_numerical_operands(runtime, None, move |outputs, pending| {
            let selected = outputs
                .selected_builtin_retirement_instruction()?
                .unwrap_or(pending);
            if !self.instruction_matches(pending, selected) {
                return Err(super::vm_fusion_host_outputs::invalid_output());
            }
            outputs.retire_builtin_operands(selected, operands)
        })
    }

    fn retire_selected_numerical_operands(
        &self,
        runtime: &RuntimeContext,
        instruction: Option<VmFusionHostInstruction>,
        retire: impl FnOnce(
            &mut super::VmFusionHostOutputs,
            VmFusionHostInstruction,
        ) -> Result<(), crate::RuntimeError>,
    ) -> Result<(), crate::RuntimeError> {
        if !self.current_for(runtime) {
            return Err(super::vm_fusion_host_outputs::missing_output(runtime));
        }
        let (mut outputs, pending_instruction) = {
            let mut state = self.backing.state.borrow_mut();
            if state.invalid
                || state.retirement_active
                || !state.pending.is_some_and(|(_, expected, _)| {
                    instruction
                        .is_none_or(|instruction| self.instruction_matches(expected, instruction))
                })
            {
                state.invalid = true;
                return Err(super::vm_fusion_host_outputs::invalid_output());
            }
            let Some((_, pending_instruction, _)) = state.pending else {
                state.invalid = true;
                return Err(super::vm_fusion_host_outputs::invalid_output());
            };
            let Some(outputs) = state.numerical_outputs.take() else {
                return match super::host_allocation::HostAllocationReceipt::policy(Some(runtime)) {
                    Ok(super::host_allocation::HostAllocationPolicy::Unbounded) => {
                        drop(state);
                        Ok(())
                    }
                    _ => {
                        state.invalid = true;
                        Err(super::vm_fusion_host_outputs::missing_output(runtime))
                    }
                };
            };
            state.retirement_active = true;
            (outputs, pending_instruction)
        };
        let result = retire(&mut outputs, pending_instruction);
        let mut state = self.backing.state.borrow_mut();
        state.retirement_active = false;
        if result.is_err() {
            state.invalid = true;
        }
        state.numerical_outputs = Some(outputs);
        result
    }

    pub fn begin_instruction(&self, pc: usize, instruction: VmFusionHostInstruction) {
        let mut state = self.backing.state.borrow_mut();
        if !self.contains_pc(pc)
            || state.pending.is_some()
            || state.next_pc != pc
            || state.retirement_active
        {
            state.invalid = true;
            return;
        }
        state.pending = Some((
            pc,
            instruction,
            instruction == VmFusionHostInstruction::Structural,
        ));
        if instruction == VmFusionHostInstruction::Unproved {
            state.invalid = true;
        }
    }

    pub fn instruction_exit(&self, pc: usize) {
        let mut state = self.backing.state.borrow_mut();
        if state.retirement_active {
            state.invalid = true;
            return;
        }
        match state.pending.take() {
            Some((pending, instruction, true)) if pending == pc => {
                if pc == self.backing.origin.registration.span_end() {
                    if !matches!(
                        instruction,
                        VmFusionHostInstruction::Structural | VmFusionHostInstruction::Unproved
                    ) {
                        state.terminal_confirmed = true;
                    } else {
                        state.invalid = true;
                    }
                }
                match pc.checked_add(1) {
                    Some(next) => state.next_pc = next,
                    None => state.invalid = true,
                }
            }
            _ => state.invalid = true,
        }
    }

    pub fn invalidate(&self) {
        self.backing.state.borrow_mut().invalid = true;
    }

    pub fn complete(&self) -> bool {
        let state = self.backing.state.borrow();
        !state.invalid
            && !state.retirement_active
            && state.pending.is_none()
            && state.terminal_confirmed
            && self.backing.origin.registration.span_end().checked_add(1) == Some(state.next_pc)
            && self.backing.origin.generation.is_current()
            && state
                .numerical_outputs
                .as_ref()
                .is_none_or(super::VmFusionHostOutputs::complete)
    }

    pub fn allows_host_promotion(&self, runtime: &RuntimeContext) -> bool {
        if !self.current_for(runtime) {
            return false;
        }
        let state = self.backing.state.borrow();
        !state.invalid
            && state.pending.is_some_and(|(_, instruction, _)| {
                matches!(
                    instruction,
                    VmFusionHostInstruction::ElementwiseBinary
                        | VmFusionHostInstruction::Intrinsic(
                            runmat_types::OperatorKind::MatrixMultiply
                        )
                )
            })
    }

    pub fn allows_host_preparation(
        &self,
        runtime: &RuntimeContext,
        instruction: VmFusionHostInstruction,
    ) -> bool {
        if !self.current_for(runtime) {
            return false;
        }
        let state = self.backing.state.borrow();
        !state.invalid
            && state
                .pending
                .is_some_and(|(_, expected, _)| self.instruction_matches(expected, instruction))
            && runtime.vm_fusion_selected_binding().is_some_and(|binding| {
                self.backing
                    .expected_operations
                    .contains(&VmFusionHostOperation::Builtin(binding))
            })
    }

    pub fn confirm_host_branch(
        &self,
        runtime: &RuntimeContext,
        instruction: VmFusionHostInstruction,
    ) {
        let valid = self.current_for(runtime)
            && runtime.vm_fusion_selected_binding().is_some_and(|binding| {
                self.backing
                    .expected_operations
                    .contains(&VmFusionHostOperation::Builtin(binding))
            });
        let mut state = self.backing.state.borrow_mut();
        match state.pending.as_mut() {
            Some((_, expected, confirmed))
                if valid && self.instruction_matches(*expected, instruction) =>
            {
                *confirmed = true;
            }
            _ => state.invalid = true,
        }
    }

    fn instruction_matches(
        &self,
        expected: VmFusionHostInstruction,
        actual: VmFusionHostInstruction,
    ) -> bool {
        expected == actual
            || matches!(
                (expected, actual),
                (
                    VmFusionHostInstruction::ElementwiseCall,
                    VmFusionHostInstruction::ElementwiseBinary
                )
            )
            || (matches!(
                (expected, actual),
                (
                    VmFusionHostInstruction::Intrinsic(runmat_types::OperatorKind::UnaryMinus),
                    VmFusionHostInstruction::ElementwiseBinary
                )
            ) && self
                .backing
                .expected_operations
                .contains(&VmFusionHostOperation::Intrinsic(
                    runmat_types::OperatorKind::UnaryMinus,
                )))
    }

    pub fn allows_intrinsic_host(
        &self,
        runtime: &RuntimeContext,
        operator: runmat_types::OperatorKind,
    ) -> bool {
        self.current_for(runtime)
            && self
                .backing
                .expected_operations
                .contains(&VmFusionHostOperation::Intrinsic(operator))
            && {
                let state = self.backing.state.borrow();
                !state.invalid
                    && state.pending.is_some_and(|(_, instruction, _)| {
                        instruction == VmFusionHostInstruction::Intrinsic(operator)
                    })
            }
    }

    /// Called only by the intrinsic numeric owner after its proven host branch.
    pub fn confirm_intrinsic_host(
        &self,
        runtime: &RuntimeContext,
        operator: runmat_types::OperatorKind,
    ) {
        let valid = self.allows_intrinsic_host(runtime, operator);
        let mut state = self.backing.state.borrow_mut();
        match state.pending.as_mut() {
            Some((_, _, confirmed)) if valid => *confirmed = true,
            _ => state.invalid = true,
        }
    }
}
