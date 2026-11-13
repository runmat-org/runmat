//! Request-local host admission for the observation wrapper's pinned future.
//! The source/program preparation receipt is a separate, simultaneously live
//! owner; neither receipt substitutes for the other's physical allocation.

use std::error::Error;
use std::future::Future;

use runmat_accelerate_api::observation::{observe_provider_future, ProviderOperationRecorder};
use runmat_accelerate_api::{ProviderExecutionError, ProviderExecutionResult};
use runmat_runtime::context::host_allocation::{HostAllocationPlan, HostAllocationReceipt};
use runmat_runtime::context::RuntimeContext;

fn admission_error<E: Error + Send + Sync + 'static>(
    error: E,
    after_effect: bool,
) -> ProviderExecutionError {
    let error = ProviderExecutionError::from_source(error);
    if after_effect {
        error.after_effect()
    } else {
        error
    }
}

pub(super) async fn observe_stage<F, T>(
    runtime: &RuntimeContext,
    recorder: Option<ProviderOperationRecorder>,
    future: F,
    after_effect: bool,
) -> ProviderExecutionResult<T>
where
    F: Future<Output = ProviderExecutionResult<T>>,
{
    // observe_provider_future owns exactly one Pin<Box<F>>. Charge its
    // concrete payload before Box::pin, and retain this additive receipt until
    // the future has completed or been dropped under the observation scope.
    let boxed_bytes = u64::try_from(std::mem::size_of::<F>().max(1))
        .map_err(|error| admission_error(error, after_effect))?;
    let plan = HostAllocationPlan::new(boxed_bytes, 0)
        .map_err(|error| admission_error(error, after_effect))?;
    let receipt = HostAllocationReceipt::admit(Some(runtime), plan)
        .map_err(|error| admission_error(error, after_effect))?;
    let output = observe_provider_future(recorder, future).await;
    if output.is_ok() {
        if let Some(receipt) = &receipt {
            receipt
                .checkpoint(runtime)
                // The first provider hook has already completed successfully.
                // A stale/cancelled authority now cannot reopen pre-effect
                // replay merely because this was the mean stage.
                .map_err(|error| admission_error(error, true))?;
        }
    }
    output
}
