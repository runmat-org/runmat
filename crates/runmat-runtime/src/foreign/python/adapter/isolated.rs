//! Owns an isolated call across suspension, including abandonment cleanup.

use std::rc::Rc;

use super::{IsolatedPythonClient, IsolatedPythonState, NestedPythonQueue};

/// A missing client is reusable only after the process owner proved retirement.
/// Retained private bytes and unverified evidence must not be overwritten by
/// spawning a replacement which could later clear the same evidence flag.
pub(super) fn admit_missing_client(
    evidence: &runmat_process_host::ProcessSpawnEvidence,
    has_client: bool,
) -> Result<(), crate::RuntimeError> {
    if evidence.is_unverified() && !has_client {
        return Err(crate::build_runtime_error(
            "Python child spawn or containment remains unverified",
        )
        .with_identifier("RunMat:Python:RetirementUnverified")
        .build());
    }
    Ok(())
}

/// Removes a not-yet-dispatched callback reentrant call when its waiter dies.
/// Once dispatched, dropping the oneshot receiver wakes the client-side
/// terminal branch instead.
pub(super) struct NestedCallLease {
    queue: NestedPythonQueue,
    ownership: Rc<()>,
}

impl NestedCallLease {
    pub(super) fn new(queue: NestedPythonQueue, ownership: Rc<()>) -> Self {
        Self { queue, ownership }
    }
}

impl Drop for NestedCallLease {
    fn drop(&mut self) {
        if let Ok(mut queue) = self.queue.try_borrow_mut() {
            queue.retain(|call| !Rc::ptr_eq(&call.ownership, &self.ownership));
        }
        // If the queue is momentarily borrowed, receiver Drop still closes
        // the oneshot and dispatch_nested_call refuses to run that entry.
    }
}

pub(super) struct IsolatedCallGuard {
    state: IsolatedPythonState,
    client: Option<IsolatedPythonClient>,
    finished: bool,
}

impl IsolatedCallGuard {
    pub(super) fn begin(state: IsolatedPythonState) -> Self {
        let client = state.client.borrow_mut().take();
        Self {
            state,
            client,
            finished: false,
        }
    }

    pub(super) fn client(&mut self) -> &mut Option<IsolatedPythonClient> {
        &mut self.client
    }

    pub(super) fn admit_replacement_spawn(&self) -> Result<(), crate::RuntimeError> {
        // Inspect this actual guard's owned client, not the shared slot which
        // is empty while a callback's parent invocation owns that same client.
        admit_missing_client(&self.state.spawn_unverified, self.client.is_some())
    }

    pub(super) fn finish_running(mut self) {
        *self.state.client.borrow_mut() = self.client.take();
        self.state.call_active.set(false);
        self.finished = true;
    }

    pub(super) fn finish_terminated(mut self) {
        // The normal path has already restarted the registry. Drop the old
        // process before another call can acquire the session slot.
        self.client.take();
        self.state.nested_calls.borrow_mut().clear();
        self.clear_releases();
        self.state.terminated.set(true);
        self.state.call_active.set(false);
        self.finished = true;
    }

    /// Preserve the invocation's primary failure while retaining registry
    /// restart failure as cleanup evidence. On restart failure the existing
    /// Drop path still revokes old handles through abandon_host_sync.
    pub(super) fn finish_terminal<T>(
        self,
        outcome: Result<T, crate::RuntimeError>,
    ) -> Result<T, crate::RuntimeError> {
        let restart = self
            .state
            .handles
            .restart_host(&self.state.host_identity)
            .map(|_| ());
        if restart.is_ok() {
            self.finish_terminated();
        } else {
            drop(self);
        }
        crate::runtime_error::complete_with_cleanup(outcome, restart)
    }
}

impl Drop for IsolatedCallGuard {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        // The process owner performs bounded whole-tree/member retirement in
        // Drop and preserves exact snapshot bytes if that proof fails.
        self.client.take();
        self.state
            .handles
            .abandon_host_sync(&self.state.host_identity);
        self.state.nested_calls.borrow_mut().clear();
        self.clear_releases();
        self.state.terminated.set(true);
        self.state.call_active.set(false);
    }
}

impl IsolatedCallGuard {
    fn clear_releases(&self) {
        self.state
            .released
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
    }
}

/// Replacement retirement is retryable: cancellation must restore the same
/// child and shutdown progress, never abandon the registry or old byte owner.
pub(super) struct ProjectRetirementGuard {
    state: IsolatedPythonState,
    client: Option<IsolatedPythonClient>,
    verified: bool,
}

impl ProjectRetirementGuard {
    pub(super) fn begin(state: IsolatedPythonState) -> Self {
        let client = state.client.borrow_mut().take();
        state.call_active.set(true);
        Self {
            state,
            client,
            verified: false,
        }
    }

    pub(super) fn client(&mut self) -> &mut Option<IsolatedPythonClient> {
        &mut self.client
    }

    pub(super) fn finish_verified(mut self) {
        self.verified = true;
        self.client.take();
        self.state.nested_calls.borrow_mut().clear();
        self.state
            .released
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
        self.state.terminated.set(true);
        self.state.call_active.set(false);
    }
}

#[cfg(all(test, unix))]
#[path = "isolated/tests.rs"]
mod tests;

impl Drop for ProjectRetirementGuard {
    fn drop(&mut self) {
        if !self.verified {
            *self.state.client.borrow_mut() = self.client.take();
            self.state.call_active.set(false);
        }
    }
}
