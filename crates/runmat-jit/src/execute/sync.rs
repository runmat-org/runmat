use std::future::Future;

use futures::FutureExt;

use crate::{JitError, JitResult};

/// Complete one R13 semantic operation without creating a nested executor.
/// A pending future is a real suspension boundary and remains R14 work.
pub(super) fn complete<T, E>(
    runtime: &runmat_runtime::context::RuntimeContext,
    future: impl Future<Output = Result<T, E>>,
    operation: &str,
) -> JitResult<T>
where
    E: Into<JitError>,
{
    runtime
        .scope(future)
        .now_or_never()
        .ok_or_else(|| {
            JitError::UnsupportedSite(format!(
                "{operation} suspended and requires the R14 continuation cohort"
            ))
        })?
        .map_err(Into::into)
}
