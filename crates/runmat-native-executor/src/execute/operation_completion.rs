use super::state::HostState;
use crate::NativeExecutorResult;

/// A fully polled runtime operation, with no live future or operand borrows.
/// Reconcile dynamic workspace writes before inspecting either success or
/// failure, so subsequent native expressions and catch handlers see them.
#[must_use = "completed runtime operations must finish against their owning native state"]
pub(super) struct CompletedOperation<T>(NativeExecutorResult<T>);

impl<T> CompletedOperation<T> {
    pub(super) fn new(result: NativeExecutorResult<T>) -> Self {
        Self(result)
    }

    pub(super) fn finish(self, state: &mut HostState) -> NativeExecutorResult<T> {
        state.reconcile_external_workspace()?;
        self.0
    }
}
