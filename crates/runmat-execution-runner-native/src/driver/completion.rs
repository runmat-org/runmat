//! Publish the scheduler's accepted terminal result together with its full history.
//! A worker response alone cannot override cancellation, rejection or retry fences.
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use runmat_execution::identity::ArtifactId;
use runmat_execution::state::TaskState;
use runmat_execution::{CancellationReason, Digest, ProgramCallable, ProgramInvocationContext};
use runmat_execution_artifact::GraphResponseAuthority;
use runmat_execution_runner::task::AttemptRecord;
use runmat_execution_runner::{AttemptRequest, DriverSnapshot, TaskExecutionEvidence};
use runmat_runtime::context::host_allocation::{HostControlAllocationOwner, HostControlVec};

use super::{TransferFailure, TransferResult};
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
    /// Exact caller-visible source identity, distinct from private store key.
    pub(super) request_artifact_id: ArtifactId,
    pub(super) artifact_id: ArtifactId,
    pub(super) stored_digest: Digest,
    pub(super) stored_length: u64,
    pub(super) materialized_scratch_bytes: u64,
    pub(super) context: ProgramInvocationContext,
    pub(super) requested_outputs: u16,
    pub(super) invocation_controls: runmat_execution_artifact::GraphInvocationControls,
    pub(super) callable: ProgramCallable,
    pub(super) response: GraphResponseAuthority,
}

pub(crate) struct TaskCompletion {
    pub(super) fea_provider: crate::fea_proxy::Provider,
    pub(super) fea_parent: Result<Option<crate::fea_delegation::Source>, String>,
    value: Mutex<Option<PersistedCompletion>>,
    terminal_wake: Arc<tokio::sync::Notify>,
    progress: Mutex<VecDeque<crate::protocol::ProgramProgress>>,
    cancellation: runmat_execution::cancellation::CancellationSource,
    control_failure: Mutex<Option<String>>,
    raw_output: Mutex<HostControlVec<crate::process_output::RawOutputReadHandle>>,
    raw_output_wake: tokio::sync::Notify,
    raw_output_failure: Mutex<Option<Arc<crate::process_output::RawOutputError>>>,
    output_delivery_stop: AtomicBool,
    /// 0 idle, 1 attempt launched, 2 cancelled before any launch.
    pub(super) launch_state: AtomicU8,
    completion_order: AtomicU64,
    pub(super) graph_authority: Option<GraphAttemptAuthority>,
}

impl TaskCompletion {
    pub(super) fn new() -> Self {
        Self {
            fea_provider: crate::fea_proxy::capture(),
            fea_parent: crate::fea_delegation::capture(),
            value: Mutex::new(None),
            terminal_wake: Arc::new(tokio::sync::Notify::new()),
            progress: Mutex::new(VecDeque::new()),
            cancellation: runmat_execution::cancellation::CancellationSource::new(),
            control_failure: Mutex::new(None),
            raw_output: Mutex::new(
                HostControlVec::with_capacity(0, None).expect("empty reader inventory"),
            ),
            raw_output_wake: tokio::sync::Notify::new(),
            raw_output_failure: Mutex::new(None),
            output_delivery_stop: AtomicBool::new(false),
            launch_state: AtomicU8::new(0),
            completion_order: AtomicU64::new(NO_COMPLETION_ORDER),
            graph_authority: None,
        }
    }

    pub(super) fn new_graph(authority: GraphAttemptAuthority) -> Self {
        let mut completion = Self::new();
        completion.graph_authority = Some(authority);
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

    /// Inspect a failure without cloning a successful numerical payload.
    pub(crate) fn original_failure(&self) -> Option<TransferFailure> {
        self.value
            .lock()
            .expect("task completion poisoned")
            .as_ref()
            .and_then(|value| value.original.result.as_ref().err().cloned())
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

    pub(crate) fn terminal_succeeded(&self) -> Option<bool> {
        self.value
            .lock()
            .expect("task completion poisoned")
            .as_ref()
            .map(|value| {
                value.original.result.is_ok()
                    && matches!(value.checkpoint, NativeCheckpointEvidence::Durable)
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

    pub(crate) fn fail_control(&self, message: String) {
        *self.control_failure.lock().expect("task control poisoned") = Some(message);
        // Shutdown stops the physical owner but is not reported as user
        // cancellation; process::execute_attempt retains the protocol failure.
        self.cancel_with_reason(CancellationReason::Shutdown);
    }

    pub(crate) fn control_failure(&self) -> Option<String> {
        self.control_failure
            .lock()
            .expect("task control poisoned")
            .clone()
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

    pub(crate) async fn wait_terminal(self: Arc<Self>) {
        loop {
            let notified = self.terminal_wake.clone().notified_owned();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.verified_terminal().is_some() {
                return;
            }
            notified.await;
        }
    }

    /// Called only on an unpublished, uniquely owned completion.
    pub(super) fn install_raw_control_owner(
        &mut self,
        owner: Option<HostControlAllocationOwner>,
    ) -> Result<(), crate::RawOutputError> {
        self.raw_output = Mutex::new(HostControlVec::with_capacity(0, owner.as_ref())?);
        Ok(())
    }

    pub(crate) fn register_raw_output(
        &self,
        reader: &crate::process_output::RawOutputReader,
    ) -> Result<(), crate::RawOutputError> {
        let mut output = self.raw_output.lock().expect("task raw output poisoned");
        let handle =
            crate::process_output::RawOutputReadHandle::new(reader, output.allocation_owner())?;
        if output.len() == output.capacity() {
            let capacity = output
                .capacity()
                .saturating_mul(2)
                .max(1)
                .min(usize::from(u16::MAX));
            if capacity <= output.len() {
                return Err(crate::RawOutputError::Invalid(
                    "reader inventory exceeds attempt ordinals",
                ));
            }
            output.reserve_capacity(capacity)?;
        }
        output
            .try_push(handle)
            .map_err(|_| crate::RawOutputError::Invalid("reader inventory capacity changed"))?;
        drop(output);
        self.raw_output_changed();
        Ok(())
    }

    pub(super) fn raw_control_owner(&self) -> Option<HostControlAllocationOwner> {
        self.raw_output
            .lock()
            .expect("task raw output poisoned")
            .allocation_owner()
            .cloned()
    }

    pub(crate) fn raw_output_reader(
        &self,
        index: usize,
    ) -> Option<crate::process_output::RawOutputReadHandle> {
        self.raw_output
            .lock()
            .expect("task raw output poisoned")
            .get(index)
            .cloned()
    }

    pub(crate) fn raw_output_changed(&self) {
        self.raw_output_wake.notify_waiters();
    }

    pub(crate) fn raw_output_notification(&self) -> tokio::sync::futures::Notified<'_> {
        self.raw_output_wake.notified()
    }

    pub(crate) fn stop_for_raw_output(&self, error: Arc<crate::process_output::RawOutputError>) {
        self.raw_output_failure
            .lock()
            .expect("task raw output poisoned")
            .get_or_insert(error);
        self.raw_output_changed();
    }

    pub(crate) fn should_stop(&self) -> bool {
        self.is_cancelled()
            || self.raw_output_failure().is_some()
            || self.output_delivery_stop.load(Ordering::Acquire)
    }

    pub(crate) fn stop_for_output_delivery(&self) {
        self.output_delivery_stop.store(true, Ordering::Release);
        self.raw_output_changed();
    }

    pub(crate) fn raw_output_failure(&self) -> Option<Arc<crate::process_output::RawOutputError>> {
        self.raw_output_failure
            .lock()
            .expect("task raw output poisoned")
            .clone()
    }

    pub(super) fn complete(&self, value: CompletedTask, checkpoint: NativeCheckpointEvidence) {
        let mut result = self.value.lock().expect("task completion poisoned");
        if result.is_none() {
            *result = Some(PersistedCompletion {
                original: value,
                checkpoint,
            });
            self.mark_completed();
            drop(result);
            self.terminal_wake.notify_waiters();
            self.raw_output_changed();
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
    let unresolved = evidence.attempts.iter().find_map(|attempt| {
        attempt
            .unverified_retirement
            .as_ref()
            .filter(|evidence| evidence.retired_at_driver_millis.is_none())
    });
    if let Some(uncertainty) = unresolved {
        return Some(CompletedTask {
            result: Err(TransferFailure::RetirementUnverified(
                observed
                    .err()
                    .map_or_else(|| uncertainty.diagnostic.clone(), |error| error.to_string()),
            )),
            evidence: Some(evidence),
        });
    }
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
            Err(failure) => failure,
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
