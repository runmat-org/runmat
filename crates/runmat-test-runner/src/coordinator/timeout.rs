use futures::future::select_all;
use futures::FutureExt;

use crate::host::{CancellationPort, Clock};
use crate::worker::{BackendError, ExecutionRequest, WorkerBackend, WorkerExecution};

pub(super) enum ExecutionRace {
    Completed(Result<WorkerExecution, BackendError>),
    TimedOut,
    Cancelled(String),
}

pub(super) async fn execute_with_controls<B, C, X>(
    backend: &B,
    session: &B::Session,
    request: ExecutionRequest,
    clock: &C,
    cancellation: &X,
) -> ExecutionRace
where
    B: WorkerBackend,
    C: Clock,
    X: CancellationPort,
{
    let mut futures = Vec::new();
    futures.push(
        backend
            .execute(session, request.clone())
            .map(ExecutionRace::Completed)
            .boxed_local(),
    );
    if let Some(deadline) = request.deadline_ms {
        futures.push(
            clock
                .sleep_until(deadline)
                .map(|()| ExecutionRace::TimedOut)
                .boxed_local(),
        );
    }
    futures.push(
        cancellation
            .cancelled()
            .map(ExecutionRace::Cancelled)
            .boxed_local(),
    );
    select_all(futures).await.0
}
