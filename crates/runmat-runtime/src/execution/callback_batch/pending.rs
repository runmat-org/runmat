//! Accepted callback receipts, original token and physical cleanup owner.
use super::*;
pub(super) struct PreparedWave {
    pub(super) calls: Vec<DeferredCall>,
    pub(super) roots: Vec<RootedValueLease>,
}

impl PreparedWave {
    pub(super) fn start<'a>(
        self,
        runtime: &'a RuntimeContext,
        pool: &PoolHandle,
        cancellation: runmat_execution::cancellation::CancellationToken,
    ) -> Result<Option<PendingBatch<'a>>, RuntimeError> {
        let expected = self.calls.len();
        if expected == 0 {
            return Ok(None);
        }
        if cancellation.is_cancelled() {
            return Err(execution_error(runtime, ExecutionServiceError::Cancelled));
        }
        let tasks = self.submit(runtime, pool, cancellation.clone())?;
        let pending = PendingBatch {
            runtime,
            tasks,
            complete: false,
            cancellation,
        };
        if pending.tasks.len() != expected {
            return Err(unavailable(
                Some(runtime),
                "worker publication returned an incomplete callback wave",
            ));
        }
        Ok(Some(pending))
    }

    fn submit(
        self,
        runtime: &RuntimeContext,
        pool: &PoolHandle,
        cancellation: runmat_execution::cancellation::CancellationToken,
    ) -> Result<Vec<TaskHandle>, RuntimeError> {
        let tasks = runtime
            .execution()
            .submit_callback_batch_in_context(self.calls, pool, cancellation, runtime)
            .map_err(|error| execution_error(runtime, error))?;
        drop(self.roots);
        Ok(tasks)
    }
}

pub(super) struct PendingBatch<'a> {
    runtime: &'a RuntimeContext,
    tasks: Vec<TaskHandle>,
    complete: bool,
    cancellation: runmat_execution::cancellation::CancellationToken,
}
impl PendingBatch<'_> {
    pub(super) async fn collect(mut self) -> Result<Vec<RootedValueSequence>, RuntimeError> {
        let runtime = self.runtime;
        let cancellation = self.cancellation.clone();
        let expected = self.tasks.len();
        use futures::stream::{FuturesUnordered, StreamExt};
        let mut waiting = FuturesUnordered::new();
        for (index, task) in self.tasks.iter().enumerate() {
            let cancellation = cancellation.clone();
            waiting.push(async move {
                loop {
                    if cancellation.is_cancelled() {
                        return Err(execution_error(runtime, ExecutionServiceError::Cancelled));
                    }
                    match runtime
                        .execution()
                        .begin_await_in_context(Value::Task(task.clone().into()), runtime)
                        .map_err(|error| execution_error(runtime, error))?
                    {
                        AwaitAction::Completed(value) => return Ok((index, value)),
                        AwaitAction::Pending(value) => {
                            let waiting = runtime
                                .execution()
                                .wait_pending(&value, runtime)
                                .map_err(|error| execution_error(runtime, error))?
                                .ok_or_else(|| {
                                    unavailable(
                                        Some(runtime),
                                        "worker task-change wake unavailable",
                                    )
                                })?;
                            waiting
                                .await
                                .map_err(|error| execution_error(runtime, error))?;
                        }
                        _ => {
                            return Err(unavailable(
                                Some(runtime),
                                "worker callback selected unexpected caller-local execution",
                            ))
                        }
                    }
                }
            });
        }
        let mut results = (0..expected).map(|_| None).collect::<Vec<_>>();
        while let Some(completed) = crate::execution::cooperation::await_original_cancellation(
            &cancellation,
            waiting.next(),
        )
        .await
        .map_err(|error| execution_error(runtime, error))?
        {
            let (index, value) = completed?;
            results[index] = Some(value);
        }
        drop(waiting);
        runtime
            .execution()
            .release_callback_batch(&self.tasks)
            .map_err(|error| execution_error(runtime, error))?;
        self.complete = true;
        Ok(results
            .into_iter()
            .map(|value| value.expect("completed callback slot"))
            .collect())
    }
}
impl Drop for PendingBatch<'_> {
    fn drop(&mut self) {
        if !self.complete {
            for task in &self.tasks {
                // Only exact owned handles: never drain an enclosing scope.
                let reason = self
                    .cancellation
                    .reason()
                    .unwrap_or(CancellationReason::ParentScope);
                let _ = self
                    .runtime
                    .execution()
                    .cancel(&Value::Task(task.clone().into()), reason);
            }
            let _ = self.runtime.execution().release_callback_batch(&self.tasks);
        }
    }
}
