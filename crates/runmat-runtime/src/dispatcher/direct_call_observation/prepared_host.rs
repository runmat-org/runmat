//! Actual canonical host-branch proof for one prepared binding, not a VM PC.
use super::{CallAdmissionSink, RegistrationAuthority, State};
use crate::builtin::RuntimeBuiltinBinding;
use crate::context::{
    RuntimeContext, RuntimeContextLocalId, RuntimePlacementService, VmFusionHostInstruction,
};
use runmat_builtins::{
    BuiltinPreparationForm, BuiltinPreparationFormId, BuiltinPreparationOperation,
};
use std::cell::RefCell;
use std::rc::{Rc, Weak};

#[derive(Default)]
pub(super) struct HostBranchState {
    selected: Option<(
        BuiltinPreparationFormId,
        BuiltinPreparationOperation,
        VmFusionHostInstruction,
    )>,
    confirmed: bool,
}

/// Read-only outside the canonical runtime operation owners. A backend can
/// request an admitted route but cannot confirm that its host body executed.
#[derive(Clone)]
pub struct PreparedHostExecution {
    pub(super) state: Weak<RefCell<State>>,
    authority: RegistrationAuthority,
    binding: RuntimeBuiltinBinding,
    owner: RuntimeContextLocalId,
    cancellation: runmat_execution::cancellation::CancellationToken,
    placement: Weak<dyn RuntimePlacementService>,
    requested_outputs: Option<usize>,
}

impl CallAdmissionSink {
    pub(crate) fn prepared_host_execution(&self) -> Option<PreparedHostExecution> {
        Some(PreparedHostExecution {
            state: self.state.clone(),
            authority: self.authority.clone(),
            binding: self.binding,
            owner: self.runtime.local_identity(),
            cancellation: self.runtime.cancellation(),
            placement: Rc::downgrade(self.runtime.service_ports().placement()?),
            requested_outputs: self.requested_outputs,
        })
    }
}

impl PreparedHostExecution {
    pub fn supports_form(form: &BuiltinPreparationForm) -> bool {
        Self::instruction(form.operation).is_some()
    }

    fn instruction(operation: BuiltinPreparationOperation) -> Option<VmFusionHostInstruction> {
        Some(match operation {
            BuiltinPreparationOperation::ElementwiseBinary
            | BuiltinPreparationOperation::RealPower
            | BuiltinPreparationOperation::FloatingRemainder(_) => {
                VmFusionHostInstruction::ElementwiseBinary
            }
            BuiltinPreparationOperation::Unary
            | BuiltinPreparationOperation::FloatingCast(_)
            | BuiltinPreparationOperation::RealUnary(_)
            | BuiltinPreparationOperation::RoundDigits => VmFusionHostInstruction::ElementwiseCall,
            _ => return None,
        })
    }
    pub(crate) fn claim_right(
        &self,
        binding: runmat_builtins::BuiltinBindingIdentity,
        requested_outputs: Option<usize>,
    ) -> Option<super::PreparedCompletionRight> {
        if binding != self.binding.identity
            || requested_outputs != self.requested_outputs
            || !self.live()
        {
            return None;
        }
        let state = self.state.upgrade()?;
        {
            let mut state = state.borrow_mut();
            if state.invalidated
                || state.prepared_claimed
                || state.selected.is_some()
                || state.prepared.is_some()
            {
                return None;
            }
            state.prepared_claimed = true;
        }
        Some(super::PreparedCompletionRight {
            state: self.state.clone(),
            authority: self.authority.clone(),
            binding: self.binding,
        })
    }

    pub fn is_current(&self) -> bool {
        self.live()
            && self.state.upgrade().is_some_and(|state| {
                let state = state.borrow();
                !state.invalidated && state.prepared_claimed
            })
    }
    pub(crate) fn matches_admission(
        &self,
        admission: crate::context::preparation::BuiltinPreparationAdmission,
    ) -> bool {
        self.binding.identity == admission.binding
            && self.requested_outputs == admission.requested_outputs
            && self.is_current()
    }

    pub(crate) fn is_current_for(&self, runtime: &RuntimeContext) -> bool {
        self.is_current()
            && !runtime.is_cancelled()
            && runtime.local_identity() == self.owner
            && runtime.vm_fusion_selected_binding() == Some(self.binding.identity)
            && crate::builtins::common::spec::selected_builtin_binding(
                self.binding.identity.builtin.name,
                Some(self.binding.identity),
                Some(runtime),
            )
            .is_some()
            && runtime
                .service_ports()
                .placement()
                .zip(self.placement.upgrade().as_ref())
                .is_some_and(|(actual, original)| Rc::ptr_eq(actual, original))
    }
    fn live(&self) -> bool {
        !self.cancellation.is_cancelled()
            && self.authority.is_current(self.binding)
            && self.placement.upgrade().is_some()
    }

    pub fn is_confirmed(&self) -> bool {
        self.live()
            && self.state.upgrade().is_some_and(|state| {
                let state = state.borrow();
                !state.invalidated && state.prepared_claimed && state.host_branch.confirmed
            })
    }

    pub(crate) fn select(&self, form: &'static BuiltinPreparationForm) -> bool {
        if !self.live() || form.binding_variant != self.binding.identity.variant {
            return false;
        }
        let Some(entry) =
            runmat_builtins::builtin_catalog_entry_by_name(self.binding.identity.builtin.name)
        else {
            return false;
        };
        let runmat_builtins::BuiltinOperandPreparation::Forms(forms) = entry.placement.preparation
        else {
            return false;
        };
        if !forms.iter().any(|candidate| candidate == form) {
            return false;
        }
        let Some(instruction) = Self::instruction(form.operation) else {
            return false;
        };
        let Some(state) = self.state.upgrade() else {
            return false;
        };
        let mut state = state.borrow_mut();
        if state.invalidated || !state.prepared_claimed || state.host_branch.selected.is_some() {
            return false;
        }
        state.host_branch.selected = Some((form.id, form.operation, instruction));
        true
    }

    pub(crate) fn allows(
        &self,
        runtime: &RuntimeContext,
        instruction: VmFusionHostInstruction,
    ) -> bool {
        if crate::builtins::common::spec::selected_builtin_binding(
            self.binding.identity.builtin.name,
            Some(self.binding.identity),
            Some(runtime),
        )
        .is_none()
        {
            return false;
        }
        if !self.live()
            || runtime.is_cancelled()
            || runtime.local_identity() != self.owner
            || runtime.vm_fusion_selected_binding() != Some(self.binding.identity)
            || !runtime
                .service_ports()
                .placement()
                .zip(self.placement.upgrade().as_ref())
                .is_some_and(|(actual, original)| Rc::ptr_eq(actual, original))
        {
            return false;
        }
        self.state.upgrade().is_some_and(|state| {
            let state = state.borrow();
            !state.invalidated
                && state.prepared_claimed
                && !state.host_branch.confirmed
                && state
                    .host_branch
                    .selected
                    .is_some_and(|(_, _, expected)| expected == instruction)
        })
    }

    pub(crate) fn confirm(&self, runtime: &RuntimeContext, instruction: VmFusionHostInstruction) {
        self.confirm_numerical(
            runtime,
            instruction,
            runmat_accelerate_api::NumericalDomainEvidence::Unknown,
        );
    }

    pub(crate) fn confirm_numerical(
        &self,
        runtime: &RuntimeContext,
        instruction: VmFusionHostInstruction,
        evidence: runmat_accelerate_api::NumericalDomainEvidence,
    ) {
        if self.allows(runtime, instruction) {
            if let Some(state) = self.state.upgrade() {
                let mut state = state.borrow_mut();
                let Some((_, operation, _)) = state.host_branch.selected else {
                    return;
                };
                if numerical_evidence_belongs(operation, evidence) {
                    state.host_branch.confirmed = true;
                }
            }
        }
    }
}

fn numerical_evidence_belongs(
    operation: BuiltinPreparationOperation,
    evidence: runmat_accelerate_api::NumericalDomainEvidence,
) -> bool {
    use runmat_accelerate_api::NumericalDomainEvidence as Evidence;
    match (operation, evidence) {
        (
            BuiltinPreparationOperation::RealUnary(expected),
            Evidence::RealUnary { predicate, .. },
        ) => expected == predicate,
        (BuiltinPreparationOperation::RealPower, Evidence::RealPower(_)) => true,
        (BuiltinPreparationOperation::RealUnary(_) | BuiltinPreparationOperation::RealPower, _) => {
            false
        }
        (_, Evidence::Unknown) => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_accelerate_api::{NumericalDomainEvidence as Evidence, PowerResultDomain};
    use runmat_types::{RealUnaryClassification, RealUnaryPredicate};
    #[test]
    fn promoting_host_confirmation_requires_its_actual_operation_evidence() {
        let unary = BuiltinPreparationOperation::RealUnary(RealUnaryPredicate::Nonnegative);
        assert!(!numerical_evidence_belongs(unary, Evidence::Unknown));
        assert!(!numerical_evidence_belongs(
            BuiltinPreparationOperation::RealPower,
            Evidence::Unknown
        ));
        for classification in [
            RealUnaryClassification::Inside,
            RealUnaryClassification::Outside,
            RealUnaryClassification::Indeterminate,
        ] {
            assert!(numerical_evidence_belongs(
                unary,
                Evidence::RealUnary {
                    predicate: RealUnaryPredicate::Nonnegative,
                    classification
                }
            ));
            assert!(!numerical_evidence_belongs(
                unary,
                Evidence::RealUnary {
                    predicate: RealUnaryPredicate::AtLeastOne,
                    classification
                }
            ));
        }
        for domain in [PowerResultDomain::Real, PowerResultDomain::Complex] {
            assert!(numerical_evidence_belongs(
                BuiltinPreparationOperation::RealPower,
                Evidence::RealPower(domain)
            ));
            assert!(!numerical_evidence_belongs(
                unary,
                Evidence::RealPower(domain)
            ));
        }
    }
}
