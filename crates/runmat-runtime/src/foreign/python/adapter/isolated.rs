//! Owns an isolated call across suspension, including abandonment cleanup.

use super::{IsolatedPythonClient, IsolatedPythonState};

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
}

impl Drop for IsolatedCallGuard {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        // Owned ChildProcess has kill_on_drop; no async Drop is required.
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
