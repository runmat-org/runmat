//! Invocation-owned file I/O awaits use the execution service's cancellation
//! boundary. This supplies no file-local turn policy or scheduler.

use std::future::Future;

use crate::context::RuntimeContext;
use crate::execution::ExecutionServiceError;
use crate::{build_runtime_error, BuiltinResult};

pub(crate) async fn checkpoint(
    runtime: Option<&RuntimeContext>,
    builtin: &'static str,
) -> BuiltinResult<()> {
    let Some(runtime) = runtime else {
        return Ok(());
    };
    runtime
        .cpu_cooperation()
        .checkpoint(0)
        .await
        .map_err(|source| execution_error(runtime, builtin, source))
}

pub(crate) async fn await_io<T, E, F>(
    runtime: Option<&RuntimeContext>,
    builtin: &'static str,
    work: F,
) -> BuiltinResult<Result<T, E>>
where
    F: Future<Output = Result<T, E>>,
{
    let Some(runtime) = runtime else {
        return Ok(work.await);
    };
    runtime
        .cpu_cooperation()
        .await_cancellable(work)
        .await
        .map_err(|source| execution_error(runtime, builtin, source))
}

fn execution_error(
    runtime: &RuntimeContext,
    builtin: &'static str,
    source: ExecutionServiceError,
) -> crate::RuntimeError {
    match source {
        cancelled @ ExecutionServiceError::Cancelled => {
            build_runtime_error(format!("{builtin}: execution was cancelled"))
                .with_builtin(builtin)
                .with_identifier(format!("{}:Cancelled", runtime.error_namespace()))
                .with_source(cancelled)
                .build()
        }
        other => other.into_runtime_error(),
    }
}
