//! Publish the scheduler's accepted terminal result together with its full history.
//! A worker response alone cannot override cancellation, rejection or retry fences.
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use runmat_execution::state::TaskState;
use runmat_execution_runner::task::AttemptRecord;
use runmat_execution_runner::{
    AttemptRequest, AttemptSuccess, DriverSnapshot, TaskExecutionEvidence,
};

use super::TransferResult;
use crate::NativeCheckpointEvidence;

const MAX_BUFFERED_PROGRESS: usize = 256;

pub(super) struct CompletedTask {
    pub result: TransferResult,
    pub evidence: Option<TaskExecutionEvidence>,
}

struct PersistedCompletion {
    original: CompletedTask,
    checkpoint: NativeCheckpointEvidence,
}

pub(crate) struct TaskCompletion {
    value: Mutex<Option<PersistedCompletion>>,
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

    pub(crate) fn try_native_completion(&self) -> Option<crate::NativeProgramCompletion> {
        self.value
            .lock()
            .expect("task completion poisoned")
            .as_ref()
            .map(|value| {
                crate::NativeProgramCompletion::new(
                    value.original.result.clone(),
                    value.original.evidence.clone(),
                    value.checkpoint.clone(),
                )
            })
    }

    pub(crate) fn try_value(&self) -> Option<TransferResult> {
        self.value
            .lock()
            .expect("task completion poisoned")
            .as_ref()
            .map(|value| match value.checkpoint.delivery_error() {
                Some(error) => Err(error),
                None => value.original.result.clone(),
            })
    }

    pub(crate) fn try_attempt_history(&self) -> Option<Vec<AttemptRecord>> {
        self.value
            .lock()
            .expect("task completion poisoned")
            .as_ref()
            .map(|value| {
                value
                    .original
                    .evidence
                    .as_ref()
                    .map_or_else(Vec::new, |evidence| evidence.attempts.clone())
            })
    }

    pub(crate) fn try_execution_evidence(&self) -> Option<TaskExecutionEvidence> {
        self.value
            .lock()
            .expect("task completion poisoned")
            .as_ref()
            .and_then(|value| value.original.evidence.clone())
    }

    /// Scheduler result for inspection, before the separate durable-delivery gate.
    pub(crate) fn try_scheduler_result(&self) -> Option<TransferResult> {
        self.value
            .lock()
            .expect("task completion poisoned")
            .as_ref()
            .map(|value| value.original.result.clone())
    }

    pub(crate) fn try_checkpoint_evidence(&self) -> Option<NativeCheckpointEvidence> {
        self.value
            .lock()
            .expect("task completion poisoned")
            .as_ref()
            .map(|value| value.checkpoint.clone())
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

    pub(super) fn complete(&self, value: CompletedTask, checkpoint: NativeCheckpointEvidence) {
        let mut result = self.value.lock().expect("task completion poisoned");
        if result.is_none() {
            *result = Some(PersistedCompletion {
                original: value,
                checkpoint,
            });
        }
    }
}

pub(super) fn from_snapshot(
    snapshot: &DriverSnapshot,
    request: &AttemptRequest,
    observed: TransferResult,
) -> Option<CompletedTask> {
    from_task_snapshot(snapshot, request.task_id, observed)
}

pub(super) fn from_task_snapshot(
    snapshot: &DriverSnapshot,
    task_id: runmat_execution::TaskId,
    observed: TransferResult,
) -> Option<CompletedTask> {
    let task = snapshot.tasks.get(&task_id)?;
    if !matches!(
        task.state,
        TaskState::Succeeded | TaskState::Failed | TaskState::Cancelled | TaskState::Indeterminate
    ) {
        return None;
    }
    let evidence = match snapshot.task_execution_evidence(task_id) {
        Ok(evidence) => evidence,
        Err(error) => {
            return Some(CompletedTask {
                result: Err(format!(
                    "local scheduler terminal evidence is invalid: {error}"
                )),
                evidence: None,
            })
        }
    };
    if evidence
        .attempts
        .iter()
        .any(|attempt| attempt.terminal.is_none())
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
        evidence: Some(evidence),
    })
}

#[cfg(test)]
mod tests;

impl crate::process_provider::ProcessControl for TaskCompletion {
    fn cancellation(&self) -> &AtomicBool {
        &self.cancelled
    }
    fn record_progress(&self, progress: crate::protocol::ProgramProgress) {
        TaskCompletion::record_progress(self, progress);
    }
}
