use std::collections::VecDeque;
use std::sync::Mutex;

use runmat_execution::identity::AttemptId;
use runmat_execution_runner::{AttemptReport, AttemptSuccess};
use tokio::sync::oneshot;

use super::{RemoteAttempt, RemoteWorkerChannel};
use crate::{NativeExecutionError, NativeExecutionResult, ProgramProgress};

const MAX_BUFFERED_PROGRESS: usize = 256;

#[derive(Default)]
struct State {
    last_sequence: u64,
    queue: VecDeque<ProgramProgress>,
}

#[derive(Default)]
pub(super) struct RemoteProgressBuffer(Mutex<State>);

pub struct RemoteTaskCompletion {
    receiver: oneshot::Receiver<Result<AttemptSuccess, String>>,
    progress: std::sync::Arc<RemoteProgressBuffer>,
}

impl RemoteTaskCompletion {
    pub(super) fn new(
        receiver: oneshot::Receiver<Result<AttemptSuccess, String>>,
        progress: std::sync::Arc<RemoteProgressBuffer>,
    ) -> Self {
        Self { receiver, progress }
    }

    pub async fn wait(self) -> Result<AttemptSuccess, String> {
        self.receiver
            .await
            .unwrap_or_else(|_| Err("remote task completion channel closed".into()))
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
        for event in channel.drain_progress(attempt_id) {
            event.validate().map_err(NativeExecutionError::Protocol)?;
            if event.sequence <= state.last_sequence {
                return Err(NativeExecutionError::Protocol(
                    "remote task progress sequence is not monotone".into(),
                ));
            }
            state.last_sequence = event.sequence;
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
                return result;
            }
            _ = tokio::time::sleep(std::time::Duration::from_millis(10)) => {
                if let Some(progress) = progress {
                    progress.append(channel, attempt_id)?;
                }
            }
        }
    }
}
