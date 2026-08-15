use std::future::Future;

use futures::FutureExt;

use crate::{NativeExecutorError, NativeExecutorResult};

/// Complete one R13 semantic operation without creating a nested executor.
/// A pending future is a real suspension boundary and remains R14 work.
pub(super) fn complete<T, E>(
    runtime: &runmat_runtime::context::RuntimeContext,
    future: impl Future<Output = Result<T, E>>,
    operation: &str,
) -> NativeExecutorResult<T>
where
    E: Into<NativeExecutorError>,
{
    runtime
        .scope(future)
        .now_or_never()
        .ok_or_else(|| {
            NativeExecutorError::UnsupportedSite(format!(
                "{operation} suspended and requires the R14 continuation cohort"
            ))
        })?
        .map_err(Into::into)
}
