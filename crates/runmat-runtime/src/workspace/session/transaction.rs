use super::*;
use std::cell::Cell;

#[derive(Debug, Default)]
pub(crate) struct SessionMutationGate(Cell<bool>);

/// Stack-only guard until the reservation transfers to a prepared token.
pub(super) struct PendingSessionReservation<'a> {
    gate: &'a SessionMutationGate,
    armed: bool,
}

impl PendingSessionReservation<'_> {
    pub(super) fn transfer(mut self) {
        self.armed = false;
    }
}

impl Drop for PendingSessionReservation<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.gate.release();
        }
    }
}

impl SessionMutationGate {
    pub(crate) fn ensure_idle(&self) -> Result<(), crate::RuntimeError> {
        if self.0.get() {
            Err(session_write_reserved())
        } else {
            Ok(())
        }
    }

    pub(super) fn reserve(&self) -> Result<PendingSessionReservation<'_>, crate::RuntimeError> {
        if self.0.replace(true) {
            return Err(session_write_reserved());
        }
        Ok(PendingSessionReservation {
            gate: self,
            armed: true,
        })
    }

    pub(super) fn active(&self) -> bool {
        self.0.get()
    }

    pub(super) fn release(&self) {
        self.0.set(false);
    }
}

/// Pre-admitted session mutation. This owns no RefCell borrow and blocks
/// conflicting session-global writes until an allocation-free commit or drop.
pub struct PreparedSessionGlobalWrite {
    pub(super) state: Weak<crate::context::RuntimeContextState>,
    pub(super) cancellation: runmat_execution::cancellation::CancellationToken,
    pub(super) expected_revision: GlobalRevision,
    pub(super) published_revision: Option<GlobalRevision>,
    pub(super) prepared: Option<PreparedGlobalWrite>,
    pub(super) _roots: crate::execution::RootedValueLease,
}

impl PreparedSessionGlobalWrite {
    /// Borrow the final value while the reservation still excludes other
    /// global writes. Frame metadata can prepare against it before publication.
    pub fn value(&self) -> &Value {
        self.prepared.as_ref().expect("write not consumed").value()
    }

    pub fn checkpoint(
        &self,
        context: &crate::context::RuntimeContext,
    ) -> Result<(), crate::RuntimeError> {
        let state = self.state.upgrade().ok_or_else(session_variables_closed)?;
        if !Rc::ptr_eq(&state, context.state()) {
            return Err(session_wrong_owner());
        }
        if !self.cancellation.same_generation(&context.cancellation()) {
            return Err(globals::session_allocation_error(
                context,
                crate::context::host_allocation::HostAllocationError::StaleAuthority,
            ));
        }
        if self.cancellation.is_cancelled() {
            return Err(globals::session_allocation_error(
                context,
                crate::context::host_allocation::HostAllocationError::Cancelled,
            ));
        }
        let variables = state
            .session_variables
            .try_borrow()
            .map_err(session_variables_read_borrow)?;
        if !state.session_global_mutation.active()
            || variables.global_revision != self.expected_revision
        {
            return Err(session_write_reserved());
        }
        drop(variables);
        self.prepared
            .as_ref()
            .expect("write not consumed")
            .checkpoint(context)
    }

    /// Call immediately after the paired delegate publication. The caller
    /// must invoke checkpoint before that publication; no await or callback
    /// may interleave the two commits.
    pub fn commit(mut self) {
        let state = self.state.upgrade().expect("prepared session remains live");
        let mut variables = state.session_variables.borrow_mut();
        assert!(
            state.session_global_mutation.active(),
            "prepared session reservation was lost"
        );
        let retired = variables
            .globals
            .commit(self.prepared.take().expect("write not consumed"));
        if let Some(revision) = self.published_revision {
            variables.global_revision = revision;
        }
        drop(variables);
        state.session_global_mutation.release();
        self.state = Weak::new();
        drop(retired);
    }
}

impl Drop for PreparedSessionGlobalWrite {
    fn drop(&mut self) {
        drop(self.prepared.take());
        if let Some(state) = self.state.upgrade() {
            state.session_global_mutation.release();
        }
    }
}
