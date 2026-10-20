//! A compact invocation cancellation view, independent of finite allocation.

use std::{cell::RefCell, rc::Rc};

use runmat_execution::cancellation::CancellationToken;

use super::state::RuntimeContextState;

/// Shares the session's namespace owner and captures its cancellation token
/// without cloning invocation projections or workspace values.
pub(crate) struct RuntimeCancellationContext {
    token: CancellationToken,
    state: Rc<RuntimeContextState>,
}

impl RuntimeCancellationContext {
    pub(super) fn new(token: CancellationToken, state: Rc<RuntimeContextState>) -> Self {
        Self { token, state }
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        self.token.is_cancelled()
    }

    pub(crate) fn error_namespace(&self) -> String {
        self.state.error_namespace.borrow().clone()
    }
}

/// A lifecycle owner's temporary binding of one cancellation-token slot.
///
/// Scopes on the same slot retire in reverse order. Invocation contexts that
/// must outlive the scope use `RuntimeContext::with_cancellation_token` to
/// retain their own generation; this guard never resets a cancellation source.
#[must_use = "dropping the scope restores the previous cancellation binding"]
pub struct RuntimeCancellationScope {
    slot: Rc<RefCell<CancellationToken>>,
    previous: Option<CancellationToken>,
}

impl RuntimeCancellationScope {
    pub(super) fn new(slot: Rc<RefCell<CancellationToken>>, token: CancellationToken) -> Self {
        let previous = slot.replace(token);
        Self {
            slot,
            previous: Some(previous),
        }
    }

    /// Retire to a successor chosen by the generation's lifecycle owner.
    /// This consumes the scope rather than exposing a persistent token setter.
    /// Nested owners restore their parent; outer owners can publish a fresh
    /// generation without making retained invocation tokens observe it.
    pub fn retire_to(mut self, successor: CancellationToken) {
        self.previous = Some(successor);
    }
}

impl Drop for RuntimeCancellationScope {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.take() {
            self.slot.replace(previous);
        }
    }
}
