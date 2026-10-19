//! Publish the scheduler's accepted terminal result together with its full history.
//! A worker response alone cannot override cancellation, rejection or retry fences.
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::Mutex;

use runmat_execution::identity::ArtifactId;
use runmat_execution::state::TaskState;
use runmat_execution::{CancellationReason, Digest, ProgramCallable, ProgramInvocationContext};
use runmat_execution_artifact::GraphResponseAuthority;
use runmat_execution_runner::task::AttemptRecord;
use runmat_execution_runner::{AttemptRequest, DriverSnapshot, TaskExecutionEvidence};

use super::{TransferFailure, TransferResult};
use crate::local_store::StreamedGraphArtifact;
use crate::NativeCheckpointEvidence;

const MAX_BUFFERED_PROGRESS: usize = 256;
const NO_COMPLETION_ORDER: u64 = u64::MAX;
static NEXT_TASK_COMPLETION_ORDER: AtomicU64 = AtomicU64::new(0);

pub(crate) fn next_task_completion_order() -> u64 {
    NEXT_TASK_COMPLETION_ORDER.fetch_add(1, Ordering::Relaxed)
}

pub(super) struct CompletedTask {
    pub result: TransferResult,
    pub evidence: Option<TaskExecutionEvidence>,
}

struct PersistedCompletion {
    original: CompletedTask,
    checkpoint: NativeCheckpointEvidence,
}

/// Exact staged Graph authority; the child receives only these scalars.
pub(crate) struct GraphAttemptAuthority {
    pub(super) artifact_id: ArtifactId,
    pub(super) stored_digest: Digest,
    pub(super) stored_length: u64,
    pub(super) context: ProgramInvocationContext,
    pub(super) requested_outputs: u16,
    pub(super) callable: ProgramCallable,
    pub(super) response: GraphResponseAuthority,
}

pub(crate) struct TaskCompletion {
    value: Mutex<Option<PersistedCompletion>>,
    progress: Mutex<VecDeque<crate::protocol::ProgramProgress>>,
    cancellation: runmat_execution::cancellation::CancellationSource,
    /// 0 idle, 1 attempt launched, 2 cancelled before any launch.
    pub(super) launch_state: AtomicU8,
    completion_order: AtomicU64,
    pub(super) graph_authority: Option<GraphAttemptAuthority>,
}

impl TaskCompletion {
    pub(super) fn new() -> Self {
        Self {
            value: Mutex::new(None),
            progress: Mutex::new(VecDeque::new()),
            cancellation: runmat_execution::cancellation::CancellationSource::new(),
            launch_state: AtomicU8::new(0),
            completion_order: AtomicU64::new(NO_COMPLETION_ORDER),
            graph_authority: None,
        }
    }

    pub(super) fn new_graph(
        stored: StreamedGraphArtifact,
        context: ProgramInvocationContext,
        requested_outputs: u16,
        callable: ProgramCallable,
        response: GraphResponseAuthority,
    ) -> Self {
        let mut completion = Self::new();
        completion.graph_authority = Some(GraphAttemptAuthority {
            artifact_id: stored.id,
            stored_digest: stored.digest,
            stored_length: stored.length,
            context,
            requested_outputs,
            callable,
            response,
        });
        completion
    }

    pub(crate) fn try_native_completion(&self) -> Option<crate::NativeProgramCompletion> {
        self.value
            .lock()
            .expect("task completion poisoned")
            .as_ref()
            .map(|value| {
                crate::NativeProgramCompletion::new(
                    value
                        .original
                        .result
                        .clone()
                        .map_err(|failure| failure.to_string()),
                    value.original.evidence.clone(),
                    value.checkpoint.clone(),
                )
            })
    }

    /// Delivered result: an uncertain checkpoint blocks delivery without
    /// rewriting the original execution.
    pub(crate) fn try_value(&self) -> Option<TransferResult> {
        self.value
            .lock()
            .expect("task completion poisoned")
            .as_ref()
            .map(|value| match value.checkpoint.delivery_error() {
                Some(error) => Err(TransferFailure::Infrastructure(error)),
                None => value.original.result.clone(),
            })
    }

    pub(crate) fn verified_terminal(&self) -> Option<bool> {
        self.value
            .lock()
            .expect("task completion poisoned")
            .as_ref()
            .map(|value| {
                !matches!(
                    value.original.result,
                    Err(TransferFailure::RetirementUnverified(_))
                )
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
        self.cancel_with_reason(CancellationReason::User);
    }

    pub(crate) fn cancel_with_reason(&self, reason: CancellationReason) {
        self.cancellation.cancel(reason);
        // A Graph worker may still be executing foreign shutdown callbacks.
        // Its terminal order is published only by complete(), after the
        // process driver has observed cleanup or killed and reaped the child.
        if self.graph_authority.is_none() {
            self.mark_completed();
        }
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancellation.token().is_cancelled()
    }

    pub(crate) fn cancellation_reason(&self) -> Option<CancellationReason> {
        self.cancellation.token().reason()
    }

    pub(crate) fn completion_order(&self) -> Option<u64> {
        let order = self.completion_order.load(Ordering::Acquire);
        (order != NO_COMPLETION_ORDER).then_some(order)
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
            self.mark_completed();
        }
    }

    /// Publish cancellation only if no attempt was ever launched; a launched
    /// attempt publishes its own terminal after its process outcome.
    pub(super) fn complete_cancelled_if_no_active_attempt(
        &self,
        checkpoint: NativeCheckpointEvidence,
    ) {
        if self
            .launch_state
            .compare_exchange(0, 2, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            self.complete(
                CompletedTask {
                    result: Err(TransferFailure::Cancelled),
                    evidence: None,
                },
                checkpoint,
            );
        }
    }

    fn mark_completed(&self) {
        let _ = self.completion_order.compare_exchange(
            NO_COMPLETION_ORDER,
            next_task_completion_order(),
            Ordering::AcqRel,
            Ordering::Acquire,
        );
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
                result: Err(TransferFailure::Infrastructure(format!(
                    "local scheduler terminal evidence is invalid: {error}"
                ))),
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
    // An unverified process-tree retirement remains visible through any
    // terminal decision; otherwise the scheduler's state is authoritative.
    let unverified = matches!(&observed, Err(TransferFailure::RetirementUnverified(_)));
    let result = match task.state {
        TaskState::Succeeded => Ok(task.committed.as_ref()?.result.clone()),
        TaskState::Failed | TaskState::Indeterminate | TaskState::Cancelled if unverified => {
            observed
        }
        TaskState::Failed => Err(observed.err().unwrap_or_else(|| {
            TransferFailure::Execution("local scheduler rejected worker result".into())
        })),
        TaskState::Cancelled => Err(TransferFailure::Cancelled),
        TaskState::Indeterminate => Err(match observed {
            Err(failure @ TransferFailure::WorkerLost(_)) => failure,
            _ => TransferFailure::WorkerLost("local task outcome is indeterminate".into()),
        }),
        _ => unreachable!("terminal task checked above"),
    };
    Some(CompletedTask {
        result,
        evidence: Some(evidence),
    })
}

#[cfg(test)]
mod tests;
