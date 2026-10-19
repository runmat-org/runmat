//! A compact invocation cancellation view, independent of finite allocation.

use std::rc::Rc;

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
