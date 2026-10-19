use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use runmat_execution::CancellationReason;

use super::pool_execution::RemotePoolExecutionOutcome;
use super::RemotePoolDriver;
use crate::NativeExecutionResult;

pub(super) type ExecutionFuture<'a> =
    Pin<Box<dyn Future<Output = NativeExecutionResult<RemotePoolExecutionOutcome>> + 'a>>;

pub(super) enum RunningWait<T> {
    Operation(T, Option<NativeExecutionResult<RemotePoolExecutionOutcome>>),
    Execution(NativeExecutionResult<RemotePoolExecutionOutcome>),
    Deadline(Option<NativeExecutionResult<RemotePoolExecutionOutcome>>),
}

pub(super) async fn await_running_operation<'a, T, F>(
    operation: F,
    execution: &mut ExecutionFuture<'a>,
    driver_ticks: &mut tokio::time::Interval,
    pool: &Arc<RemotePoolDriver>,
    cancellation: &mut tokio::sync::watch::Receiver<bool>,
    cancellation_requested: &mut bool,
    deadline: tokio::time::Instant,
) -> NativeExecutionResult<RunningWait<T>>
where
    F: Future<Output = T>,
{
    tokio::pin!(operation);
    let mut execution_result = None;
    let mut cancellation_open = true;
    let mut cancellation_failure = None;
    loop {
        if *cancellation.borrow() && !*cancellation_requested {
            cancellation_failure = pool.cancel(CancellationReason::User).err();
            *cancellation_requested = true;
        }
        if *cancellation_requested {
            if let Some(result) = execution_result.take() {
                if let Some(error) = cancellation_failure.take() {
                    return Err(error);
                }
                return Ok(RunningWait::Execution(result));
            }
        }
        tokio::select! {
            result = &mut operation => {
                if let Some(error) = cancellation_failure.take() {
                    return Err(error);
                }
                return Ok(RunningWait::Operation(result, execution_result));
            }
            result = execution.as_mut(), if execution_result.is_none() => {
                if *cancellation_requested {
                    if let Some(error) = cancellation_failure.take() {
                        return Err(error);
                    }
                    return Ok(RunningWait::Execution(result));
                }
                execution_result = Some(result);
            }
            _ = driver_ticks.tick() => pool.tick()?,
            changed = cancellation.changed(), if cancellation_open => {
                if changed.is_err() {
                    cancellation_open = false;
                } else if *cancellation.borrow() && !*cancellation_requested {
                    cancellation_failure = pool.cancel(CancellationReason::User).err();
                    *cancellation_requested = true;
                }
            }
            _ = tokio::time::sleep_until(deadline) => {
                if let Some(error) = cancellation_failure.take() {
                    return Err(error);
                }
                return Ok(RunningWait::Deadline(execution_result));
            }
        }
    }
}

pub(super) enum StartupWait<T> {
    Completed(T, bool),
    Deadline(bool),
}

pub(super) async fn await_startup_operation<T, F>(
    operation: F,
    driver_ticks: &mut tokio::time::Interval,
    pool: &Arc<RemotePoolDriver>,
    cancellation: &mut tokio::sync::watch::Receiver<bool>,
    deadline: tokio::time::Instant,
) -> NativeExecutionResult<StartupWait<T>>
where
    F: Future<Output = T>,
{
    tokio::pin!(operation);
    let mut cancelled = false;
    let mut cancellation_open = true;
    let mut cancellation_failure = None;
    loop {
        if *cancellation.borrow() && !cancelled {
            cancellation_failure = pool.cancel(CancellationReason::User).err();
            cancelled = true;
        }
        tokio::select! {
            result = &mut operation => {
                if let Some(error) = cancellation_failure.take() {
                    return Err(error);
                }
                return Ok(StartupWait::Completed(result, cancelled));
            }
            _ = driver_ticks.tick() => pool.tick()?,
            changed = cancellation.changed(), if cancellation_open => {
                if changed.is_err() {
                    cancellation_open = false;
                } else if *cancellation.borrow() && !cancelled {
                    cancellation_failure = pool.cancel(CancellationReason::User).err();
                    cancelled = true;
                }
            }
            _ = tokio::time::sleep_until(deadline) => {
                if let Some(error) = cancellation_failure.take() {
                    return Err(error);
                }
                return Ok(StartupWait::Deadline(cancelled));
            }
        }
    }
}
