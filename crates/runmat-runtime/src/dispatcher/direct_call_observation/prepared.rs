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
    state: Weak<RefCell<State>>,
    authority: RegistrationAuthority,
    binding: RuntimeBuiltinBinding,
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
        if self.binding.identity != binding
            || self.requested_outputs != requested_outputs
            || self.runtime.is_cancelled()
            || !self.authority.certifies(self.binding)
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
        Some(PreparedCompletionRight {
            state: self.state,
            authority: self.authority,
            binding: self.binding,
        })
    }
}

impl PreparedCompletionRight {
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
