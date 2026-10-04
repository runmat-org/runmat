//! One complete data-family effect receipt, cancellation, and cooperative scheduling.

use crate::context::host_effect::HostEffectReceipt;
use crate::context::RuntimeContext;
use crate::execution::ExecutionServiceError;
use crate::BuiltinResult;
use runmat_execution::WholeWorkObservationOutcome;
use std::future::Future;

pub(in crate::builtins::io::data) async fn execute<T>(
    runtime: Option<&RuntimeContext>,
    receipt: Option<HostEffectReceipt>,
    effect: impl Future<Output = BuiltinResult<T>>,
) -> BuiltinResult<T> {
    if let Some(receipt) = receipt.as_ref() {
        receipt.checkpoint()?;
    }
    let mut cancelled = false;
    let result = if let Some(runtime) = runtime {
        let mut budget = runtime.cpu_cooperation();
        match budget.checkpoint(0).await {
            Ok(()) => match budget.await_cancellable(effect).await {
                Ok(result) => result,
                Err(ExecutionServiceError::Cancelled) => {
                    cancelled = true;
                    Err(cancelled_error())
                }
                Err(error) => Err(error.into_runtime_error()),
            },
            Err(ExecutionServiceError::Cancelled) => {
                cancelled = true;
                Err(cancelled_error())
            }
            Err(error) => Err(error.into_runtime_error()),
        }
    } else {
        effect.await
    };
    if result.as_ref().err().is_some_and(|error| {
        error.source.as_deref().is_some_and(|source| {
            matches!(
                source.downcast_ref::<ExecutionServiceError>(),
                Some(ExecutionServiceError::Cancelled)
            )
        })
    }) {
        cancelled = true;
    }
    let result = if result.is_ok() {
        if let Some(receipt) = receipt.as_ref() {
            receipt.checkpoint()?;
        }
        if let Some(runtime) = runtime {
            if runtime.is_cancelled() {
                cancelled = true;
                Err(cancelled_error())
            } else {
                result
            }
        } else {
            result
        }
    } else {
        result
    };
    if let Some(receipt) = receipt {
        receipt.finish(if cancelled {
            WholeWorkObservationOutcome::Cancelled
        } else if result.is_ok() {
            WholeWorkObservationOutcome::Success
        } else {
            WholeWorkObservationOutcome::OperationalFailure
        });
    }
    result
}

fn cancelled_error() -> crate::RuntimeError {
    crate::runtime_error::semantic_error("RunMat:Cancelled", "data invocation was cancelled")
}
