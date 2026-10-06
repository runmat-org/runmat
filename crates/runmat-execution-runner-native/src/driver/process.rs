//! Local adapter to the common native process provider.
use super::{LocalDriver, TaskCompletion};
use crate::process_provider::{NativeProcessProvider, ObservedAttempt};

pub(super) fn execute_attempt(
    driver: &LocalDriver,
    request: &runmat_execution_runner::AttemptRequest,
    completion: &TaskCompletion,
) -> ObservedAttempt {
    let started = std::time::Instant::now();
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => return ObservedAttempt::not_started(request, started, error.to_string()),
    };
    runtime.block_on(crate::process_provider::execute_attempt(
        &NativeProcessProvider {
            config: &driver.config,
            artifacts: &driver.artifacts,
            objects: &driver.objects,
        },
        request,
        completion,
        started,
    ))
}

#[cfg(test)]
mod tests;
