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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VmFusionHostOperation {
    Builtin(BuiltinBindingIdentity),
    Intrinsic(runmat_types::OperatorKind),
}

#[derive(Clone)]
pub struct VmFusionHostRoute {
    origin: VmFusionHostOrigin,
    expected_operations: Rc<[VmFusionHostOperation]>,
    state: Rc<RefCell<RouteState>>,
}

#[derive(Clone)]
pub(super) struct VmFusionHostOrigin {
    pub registration: VmFusionRegionRegistration,
    pub owner: RuntimeContextLocalId,
    pub program: Option<ProgramRevision>,
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
            state: Rc::new(RefCell::new(RouteState {
                pending: None,
                invalid: false,
                next_pc: origin.registration.span_start(),
                terminal_confirmed: false,
            })),
            origin,
            expected_operations: expected_operations.into(),
        })
    }

    pub fn contains_pc(&self, pc: usize) -> bool {
        self.origin.registration.contains_pc(pc)
    }

    pub fn current_for(&self, runtime: &RuntimeContext) -> bool {
        let same_placement_owner = self
            .origin
            .placement_owner
            .upgrade()
            .zip(runtime.service_ports().placement())
            .is_some_and(|(selected, current)| Rc::ptr_eq(&selected, current));
        runtime.local_identity() == self.origin.owner
            && runtime.program_revision() == self.origin.program.as_ref()
            && runtime
                .cancellation()
                .same_generation(&self.origin.cancellation)
            && same_placement_owner
            && !runtime.is_cancelled()
            && self.origin.issuer_live.upgrade().is_some()
            && self.origin.generation.is_current()
    }

    pub fn begin_instruction(&self, pc: usize, instruction: VmFusionHostInstruction) {
        let mut state = self.state.borrow_mut();
        if !self.contains_pc(pc) || state.pending.is_some() || state.next_pc != pc {
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
        let mut state = self.state.borrow_mut();
        match state.pending.take() {
            Some((pending, instruction, true)) if pending == pc => {
                if pc == self.origin.registration.span_end() {
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
        self.state.borrow_mut().invalid = true;
    }

    pub fn complete(&self) -> bool {
        let state = self.state.borrow();
        !state.invalid
            && state.pending.is_none()
            && state.terminal_confirmed
            && self.origin.registration.span_end().checked_add(1) == Some(state.next_pc)
            && self.origin.generation.is_current()
    }

    pub fn allows_host_promotion(&self, runtime: &RuntimeContext) -> bool {
        if !self.current_for(runtime) {
            return false;
        }
        let state = self.state.borrow();
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
        let state = self.state.borrow();
        !state.invalid
            && state
                .pending
                .is_some_and(|(_, expected, _)| self.instruction_matches(expected, instruction))
            && runtime.vm_fusion_selected_binding().is_some_and(|binding| {
                self.expected_operations
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
                self.expected_operations
                    .contains(&VmFusionHostOperation::Builtin(binding))
            });
        let mut state = self.state.borrow_mut();
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
                .expected_operations
                .contains(&VmFusionHostOperation::Intrinsic(operator))
            && {
                let state = self.state.borrow();
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
        let mut state = self.state.borrow_mut();
        match state.pending.as_mut() {
            Some((_, _, confirmed)) if valid => *confirmed = true,
            _ => state.invalid = true,
        }
    }
}
