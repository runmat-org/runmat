//! Publish the scheduler's accepted terminal result together with its full history.
//! A worker response alone cannot override cancellation, rejection or retry fences.
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use runmat_execution::state::TaskState;
use runmat_execution_runner::task::AttemptRecord;
use runmat_execution_runner::{AttemptRequest, AttemptSuccess, DriverSnapshot};

use super::TransferResult;

const MAX_BUFFERED_PROGRESS: usize = 256;

pub(super) struct CompletedTask {
    pub result: TransferResult,
    pub attempts: Vec<AttemptRecord>,
}

pub(crate) struct TaskCompletion {
    value: Mutex<Option<CompletedTask>>,
    progress: Mutex<VecDeque<crate::protocol::ProgramProgress>>,
    pub(super) cancelled: AtomicBool,
}

impl TaskCompletion {
    pub(super) fn new() -> Self {
        Self {
            value: Mutex::new(None),
            progress: Mutex::new(VecDeque::new()),
            cancelled: AtomicBool::new(false),
        }
    }

    pub(crate) fn try_value(&self) -> Option<TransferResult> {
        self.value
            .lock()
            .expect("task completion poisoned")
            .as_ref()
            .map(|value| value.result.clone())
    }

    pub(crate) fn try_attempt_history(&self) -> Option<Vec<AttemptRecord>> {
        self.value
            .lock()
            .expect("task completion poisoned")
            .as_ref()
            .map(|value| value.attempts.clone())
    }

    pub(crate) fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub(crate) fn record_progress(&self, progress: crate::protocol::ProgramProgress) {
        let mut buffered = self.progress.lock().expect("task progress poisoned");
        if buffered.len() == MAX_BUFFERED_PROGRESS {
            buffered.pop_front();
        }
        buffered.push_back(progress);
    }

    pub(crate) fn drain_progress(&self) -> Vec<crate::protocol::ProgramProgress> {
        self.progress
            .lock()
            .expect("task progress poisoned")
            .drain(..)
            .collect()
    }

    pub(super) fn complete(&self, value: CompletedTask) {
        let mut result = self.value.lock().expect("task completion poisoned");
        if result.is_none() {
            *result = Some(value);
        }
    }
}

pub(super) fn from_snapshot(
    snapshot: &DriverSnapshot,
    request: &AttemptRequest,
    observed: TransferResult,
) -> Option<CompletedTask> {
    let task = snapshot.tasks.get(&request.task_id)?;
    if !matches!(
        task.state,
        TaskState::Succeeded | TaskState::Failed | TaskState::Cancelled | TaskState::Indeterminate
    ) {
        return None;
    }
    let history: Vec<AttemptRecord> = match snapshot.task_attempt_history(request.task_id) {
        Ok(history) => history.into_iter().cloned().collect(),
        Err(error) => {
            return Some(CompletedTask {
                result: Err(format!(
                    "local scheduler terminal evidence is invalid: {error}"
                )),
                attempts: Vec::new(),
            })
        }
    };
    if history
        .iter()
        .any(|attempt: &AttemptRecord| attempt.terminal.is_none())
    {
        return None;
    }
    let result = match task.state {
        TaskState::Succeeded => {
            let commit = task.committed.as_ref()?;
            Ok(AttemptSuccess {
                outputs: commit.outputs.clone(),
                result_objects: commit.result_objects.clone(),
                resources: commit.resources.clone(),
            })
        }
        TaskState::Failed => Err(observed
            .err()
            .unwrap_or_else(|| "local scheduler rejected worker result".into())),
        TaskState::Cancelled => Err("execution was cancelled".into()),
        TaskState::Indeterminate => Err("local task outcome is indeterminate".into()),
        _ => unreachable!("terminal task checked above"),
    };
    Some(CompletedTask {
        result,
        attempts: history,
    })
}

#[cfg(test)]
mod tests;
