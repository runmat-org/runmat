use std::collections::VecDeque;
use std::sync::Mutex;

use runmat_execution::identity::AttemptId;
use runmat_execution::SpmdTaskContext;
use runmat_execution_runner::{AttemptReport, AttemptSuccess};
use tokio::sync::oneshot;

use super::{RemoteAttempt, RemoteWorkerChannel};
use crate::{NativeExecutionError, NativeExecutionResult, ProgramProgress};

const MAX_BUFFERED_PROGRESS: usize = 256;

#[derive(Default)]
struct State {
    active_attempt: Option<AttemptId>,
    last_attempt_sequence: u64,
    next_task_sequence: u64,
    queue: VecDeque<ProgramProgress>,
}

#[derive(Default)]
pub(super) struct RemoteProgressBuffer(Mutex<State>);

pub(super) struct RemoteCollectiveAssignment {
    broker: std::sync::Arc<crate::driver::collective::ProcessCollectiveBroker>,
    context: SpmdTaskContext,
}

impl RemoteCollectiveAssignment {
    pub(super) fn new(
        broker: std::sync::Arc<crate::driver::collective::ProcessCollectiveBroker>,
        context: SpmdTaskContext,
    ) -> Self {
        Self { broker, context }
    }
}

pub struct RemoteTaskCompletion {
    receiver: oneshot::Receiver<(u64, Result<AttemptSuccess, crate::NativeProgramFailure>)>,
    progress: std::sync::Arc<RemoteProgressBuffer>,
}

impl RemoteTaskCompletion {
    pub(super) fn new(
        receiver: oneshot::Receiver<(u64, Result<AttemptSuccess, crate::NativeProgramFailure>)>,
        progress: std::sync::Arc<RemoteProgressBuffer>,
    ) -> Self {
        Self { receiver, progress }
    }

    pub async fn wait(self) -> Result<AttemptSuccess, crate::NativeProgramFailure> {
        self.wait_ordered().await.1
    }

    pub(super) async fn wait_ordered(
        self,
    ) -> (u64, Result<AttemptSuccess, crate::NativeProgramFailure>) {
        self.receiver.await.unwrap_or_else(|_| {
            (
                u64::MAX,
                Err(crate::NativeProgramFailure::Infrastructure(
                    "remote task completion channel closed".into(),
                )),
            )
        })
    }

    pub fn drain_progress(&self) -> Vec<ProgramProgress> {
        self.progress.drain()
    }
}

impl RemoteProgressBuffer {
    pub(super) fn drain(&self) -> Vec<ProgramProgress> {
        self.0
            .lock()
            .expect("remote task progress poisoned")
            .queue
            .drain(..)
            .collect()
    }

    fn append(
        &self,
        channel: &dyn RemoteWorkerChannel,
        attempt_id: AttemptId,
    ) -> NativeExecutionResult<()> {
        let mut state = self.0.lock().expect("remote task progress poisoned");
        if state.active_attempt != Some(attempt_id) {
            state.active_attempt = Some(attempt_id);
            state.last_attempt_sequence = 0;
        }
        for mut event in channel.drain_progress(attempt_id) {
            event.validate().map_err(NativeExecutionError::Protocol)?;
            if event.sequence <= state.last_attempt_sequence {
                return Err(NativeExecutionError::Protocol(
                    "remote attempt progress sequence is not monotone".into(),
                ));
            }
            state.last_attempt_sequence = event.sequence;
            state.next_task_sequence =
                state.next_task_sequence.checked_add(1).ok_or_else(|| {
                    NativeExecutionError::Protocol(
                        "remote task progress sequence overflowed".into(),
                    )
                })?;
            event.sequence = state.next_task_sequence;
            if state.queue.len() == MAX_BUFFERED_PROGRESS {
                state.queue.pop_front();
            }
            state.queue.push_back(event);
        }
        Ok(())
    }
}

pub(super) async fn execute(
    channel: &dyn RemoteWorkerChannel,
    attempt: RemoteAttempt,
    progress: Option<&RemoteProgressBuffer>,
    collectives: Option<RemoteCollectiveAssignment>,
) -> NativeExecutionResult<AttemptReport> {
    let attempt_id = attempt.scheduling.id;
    let execution = channel.execute(attempt);
    tokio::pin!(execution);
    loop {
        tokio::select! {
            result = &mut execution => {
                if let Some(progress) = progress {
                    progress.append(channel, attempt_id)?;
                }
                forward_collectives(channel, attempt_id, collectives.as_ref()).await?;
                return result;
            }
            _ = tokio::time::sleep(std::time::Duration::from_millis(10)) => {
                if let Some(progress) = progress {
                    progress.append(channel, attempt_id)?;
                }
                forward_collectives(channel, attempt_id, collectives.as_ref()).await?;
            }
        }
    }
}

async fn forward_collectives(
    channel: &dyn RemoteWorkerChannel,
    attempt_id: AttemptId,
    collectives: Option<&RemoteCollectiveAssignment>,
) -> NativeExecutionResult<()> {
    let requests = channel.drain_collective_requests(attempt_id);
    if requests.is_empty() {
        return Ok(());
    }
    let assignment = collectives.ok_or_else(|| {
        NativeExecutionError::Protocol(
            "non-SPMD remote attempt emitted a collective request".into(),
        )
    })?;
    for request in requests {
        if request.context != assignment.context {
            return Err(NativeExecutionError::Protocol(
                "remote collective request differs from its scheduled rank assignment".into(),
            ));
        }
        let broker = std::sync::Arc::clone(&assignment.broker);
        let submitted = request.clone();
        let result = tokio::task::spawn_blocking(move || broker.execute_remote(submitted))
            .await
            .map_err(|error| {
                NativeExecutionError::Protocol(format!(
                    "remote collective coordinator stopped: {error}"
                ))
            })?;
        let result = match result {
            Ok(response) => crate::protocol::CollectiveProcessResult::Completed { response },
            Err(message) => crate::protocol::CollectiveProcessResult::Failed { message },
        };
        channel
            .complete_collective(attempt_id, &request, result)
            .await?;
    }
    Ok(())
}
