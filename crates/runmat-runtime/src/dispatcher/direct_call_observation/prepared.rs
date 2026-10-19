//! One selected binding's prepared-body completion, owned by its outer call.
use super::{CallAdmissionSink, RegistrationAuthority, State};
use crate::builtin::RuntimeBuiltinBinding;
use crate::context::preparation::PreparedCallCompletion;
use runmat_builtins::BuiltinBindingIdentity;
use runmat_execution::Digest;
use std::cell::RefCell;
use std::rc::Weak;

pub(super) struct PreparedSelection {
    pub(super) completion: Box<dyn PreparedCallCompletion>,
    pub(super) authority: RegistrationAuthority,
    pub(super) binding: RuntimeBuiltinBinding,
}

/// Non-clone right minted against the actually selected binding and output
/// mode. It can deposit at most one cleaned-up, body-complete observation.
/// There is no ambient slot to be borrowed by a sibling or nested call.
pub struct PreparedCompletionRight {
    pub(super) state: Weak<RefCell<State>>,
    pub(super) authority: RegistrationAuthority,
    pub(super) binding: RuntimeBuiltinBinding,
}

/// Session-local authority for a prepared observation. This is not installed
/// code identity and cannot be used for persisted or cross-process reuse.
#[derive(Clone, Copy)]
pub struct PreparedBindingAuthority {
    binding: BuiltinBindingIdentity,
    generation: Digest,
}

impl PreparedBindingAuthority {
    pub fn binding(self) -> BuiltinBindingIdentity {
        self.binding
    }

    pub fn generation(self) -> Digest {
        self.generation
    }
}

impl CallAdmissionSink {
    pub(crate) fn prepared_right(
        self,
        binding: BuiltinBindingIdentity,
        requested_outputs: Option<usize>,
    ) -> Option<PreparedCompletionRight> {
        self.prepared_host_execution()?
            .claim_right(binding, requested_outputs)
    }
}

impl PreparedCompletionRight {
    /// Arm only the exact original call's selected preparation form. Numeric
    /// operation owners, not the acceleration service, supply confirmation.
    pub fn select_host_execution(
        &self,
        runtime: &crate::context::RuntimeContext,
        form: &'static runmat_builtins::BuiltinPreparationForm,
    ) -> Option<super::PreparedHostExecution> {
        let proof = runtime.prepared_host_execution()?.clone();
        if !Weak::ptr_eq(&proof.state, &self.state) {
            return None;
        }
        proof.select(form).then_some(proof)
    }
    pub fn binding(&self) -> BuiltinBindingIdentity {
        self.binding.identity
    }

    pub fn session_authority(&self) -> PreparedBindingAuthority {
        PreparedBindingAuthority {
            binding: self.binding.identity,
            generation: self.authority.generation(),
        }
    }

    pub fn deposit(self, completion: Box<dyn PreparedCallCompletion>) {
        if completion.binding() != self.binding.identity || !self.authority.is_current(self.binding)
        {
            return;
        }
        let Some(state) = self.state.upgrade() else {
            return;
        };
        let mut state = state.borrow_mut();
        if state.invalidated || state.selected.is_some() || state.prepared.is_some() {
            return;
        }
        state.prepared = Some(PreparedSelection {
            completion,
            authority: self.authority,
            binding: self.binding,
        });
    }
}
