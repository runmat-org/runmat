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
    Reduction,
    Unproved,
}

#[derive(Clone)]
pub struct VmFusionHostRoute {
    origin: VmFusionHostOrigin,
    expected_bindings: Rc<[BuiltinBindingIdentity]>,
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
        expected_bindings: Vec<BuiltinBindingIdentity>,
    ) -> Option<Self> {
        if expected_bindings.is_empty() {
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
            expected_bindings: expected_bindings.into(),
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
                    if instruction == VmFusionHostInstruction::Reduction {
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
                instruction == VmFusionHostInstruction::ElementwiseBinary
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
                .is_some_and(|(_, expected, _)| expected == instruction)
            && runtime
                .vm_fusion_selected_binding()
                .is_some_and(|binding| self.expected_bindings.contains(&binding))
    }

    pub fn confirm_host_branch(
        &self,
        runtime: &RuntimeContext,
        instruction: VmFusionHostInstruction,
    ) {
        let valid = self.current_for(runtime)
            && runtime
                .vm_fusion_selected_binding()
                .is_some_and(|binding| self.expected_bindings.contains(&binding));
        let mut state = self.state.borrow_mut();
        match state.pending.as_mut() {
            Some((_, expected, confirmed)) if valid && *expected == instruction => {
                *confirmed = true;
            }
            _ => state.invalid = true,
        }
    }
}
