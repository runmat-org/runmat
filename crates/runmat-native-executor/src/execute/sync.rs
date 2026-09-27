use std::future::Future;

use futures::FutureExt;

use super::state::HostState;
use crate::{NativeExecutorError, NativeExecutorResult};

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

/// Complete one R13 semantic operation without creating a nested executor.
/// A pending future is a real suspension boundary and remains R14 work.
pub(super) fn complete<T, E>(
    runtime: &runmat_runtime::context::RuntimeContext,
    future: impl Future<Output = Result<T, E>>,
    operation: &str,
) -> CompletedOperation<T>
where
    E: Into<NativeExecutorError>,
{
    let result = runtime
        .scope(future)
        .now_or_never()
        .ok_or_else(|| {
            NativeExecutorError::UnsupportedSite(format!(
                "{operation} suspended and requires the R14 continuation cohort"
            ))
        })
        .and_then(|result| result.map_err(Into::into));
    CompletedOperation::new(result)
}
