use std::sync::atomic::Ordering;
use std::sync::Arc;

use runmat_execution::state::{AttemptState, TaskState};
use runmat_execution_runner::driver::DriverEventKind;
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::task::TaskRecord;
use runmat_execution_runner::{
    AttemptFailureKind, AttemptReport, Driver, DriverAction, DriverCommand,
};

use super::RemotePoolDriver;
use crate::NativeProgramFailure;

type PendingTaskResolution = (
    tokio::sync::oneshot::Sender<(u64, super::CompletionResult)>,
    u64,
    super::CompletionResult,
);

impl RemotePoolDriver {
    pub(super) fn apply_report(self: &Arc<Self>, mut report: BackendReport) {
        if matches!(&report.report, AttemptReport::GraphRetiredCancelled) {
            // Only the local process owner can issue this certificate. A
            // remote worker cannot prove parent-owned process-tree cleanup.
            report.report = AttemptReport::RetirementUnverified {
                message: "remote Graph retirement certificate has no local process owner".into(),
            };
        }
        let task_id = report.task_id;
        let terminal_report = report.report.clone();
        let spmd_task = self.spmd_task_context(task_id);
        let (actions, outcome, committed_watermark) = {
            let mut driver = self.driver.lock().expect("remote driver poisoned");
            let previous_events = driver.events().len();
            let actions = match driver.handle(DriverCommand::BackendReport(report)) {
                Ok(actions) => actions,
                Err(error) => {
                    tracing::error!(%task_id, %error, "remote scheduler rejected backend report");
                    return;
                }
            };
            let outcome = driver.task_record(task_id).ok().and_then(|task| {
                if task.state == TaskState::Succeeded {
                    Some(match &task.committed {
                        Some(commit) => self
                            .execution_objects
                            .commit_results(commit.result.result_objects())
                            .map(|()| commit.result.clone())
                            .map_err(|error| {
                                NativeProgramFailure::Infrastructure(error.to_string())
                            }),
                        None => Err(NativeProgramFailure::Infrastructure(
                            "succeeded remote task has no committed result".into(),
                        )),
                    })
                } else if self.direct_bundle.is_none()
                    && task.state == TaskState::Indeterminate
                    && matches!(&terminal_report, AttemptReport::Succeeded { .. })
                {
                    Some(Err(NativeProgramFailure::WorkerLost(
                        "Graph attempt reported success without verified worker retirement".into(),
                    )))
                } else {
                    terminal_failure(&driver, task)
                        .map(|message| Err(report_failure(&terminal_report, task.state, message)))
                }
            });
            // Serialize committed-object protection and stale-result disposal with
            // scheduler admission. Otherwise a discarded overlapping result can
            // remove an accepted object's catalog entry before it is protected.
            // Lock order is driver -> object catalog; neither operation calls back
            // into the driver. Dispatch/worker calls remain outside both locks.
            let actions = actions
                .into_iter()
                .filter_map(|action| match action {
                    DriverAction::GarbageCollectResults { objects, .. } => {
                        self.execution_objects.discard_results(&objects);
                        None
                    }
                    action => Some(action),
                })
                .collect();
            let committed_watermark = driver.events()[previous_events..].iter().find_map(|event| {
                matches!(&event.kind, DriverEventKind::ResultCommitted { task_id: committed, .. }
                    if *committed == task_id)
                .then_some(event.sequence + 1)
            });
            (actions, outcome, committed_watermark)
        };
        self.dispatch(actions);
        if let Some(watermark) = committed_watermark {
            self.committed_event_watermark
                .fetch_max(watermark, Ordering::AcqRel);
            self.committed_task.notify_one();
        }
        if let Some(outcome) = outcome {
            self.publish_ordered(task_id, spmd_task, outcome);
        }
    }

    pub(super) fn resolve_non_success_terminals(&self) {
        let terminal = {
            let driver = self.driver.lock().expect("remote driver poisoned");
            driver
                .task_records()
                .filter_map(|task| {
                    terminal_failure(&driver, task)
                        .map(|message| (task.id(), state_failure(task.state, message)))
                })
                .collect::<Vec<_>>()
        };
        for (task_id, failure) in terminal {
            let spmd_task = self.spmd_task_context(task_id);
            self.publish_ordered(task_id, spmd_task, Err(failure));
        }
    }

    /// Reserve the originating task's deterministic completion order, then
    /// fence its SPMD peers before making that completion observable.
    /// Otherwise the gang waiter can cancel a blocked peer before the
    /// collective broker has released it.
    fn publish_ordered(
        &self,
        task_id: runmat_execution::TaskId,
        spmd_task: Option<runmat_execution::SpmdTaskContext>,
        outcome: super::CompletionResult,
    ) {
        let failed = outcome.is_err();
        let resolution = self.take_task_resolution(task_id, outcome);
        match spmd_task {
            Some(task) => {
                if failed {
                    // Publish the originating result before releasing peers
                    // blocked in this gang's collective rounds.
                    self.collectives
                        .fail_gang(&task.gang, "an SPMD peer terminated");
                }
                Self::publish_task_resolution(resolution);
                self.collectives.rank_terminated(&task.gang, task.rank);
            }
            None => Self::publish_task_resolution(resolution),
        }
    }

    fn spmd_task_context(
        &self,
        task_id: runmat_execution::TaskId,
    ) -> Option<runmat_execution::SpmdTaskContext> {
        self.programs
            .lock()
            .expect("remote program catalog poisoned")
            .get(&task_id)
            .and_then(|program| match program.as_ref() {
                runmat_execution_artifact::WorkerProgramRequest::Direct(direct) => {
                    match &direct.context {
                        runmat_execution::ProgramInvocationContext::SpmdTask { task } => {
                            Some(task.clone())
                        }
                        _ => None,
                    }
                }
                runmat_execution_artifact::WorkerProgramRequest::Graph(_) => None,
            })
    }

    fn take_task_resolution(
        &self,
        task_id: runmat_execution::TaskId,
        outcome: super::CompletionResult,
    ) -> Option<PendingTaskResolution> {
        let resolution = self
            .completions
            .lock()
            .expect("remote completion registry poisoned")
            .remove(&task_id)
            .map(|sender| {
                let order = self.completion_sequence.fetch_add(1, Ordering::AcqRel);
                (sender, order, outcome)
            });
        // Exact admitted programs and Graph transfer authority have the same
        // lifetime as scheduler task records: completed-task replay and durable
        // recovery still need them.
        self.progress
            .lock()
            .expect("remote progress registry poisoned")
            .remove(&task_id);
        resolution
    }

    fn publish_task_resolution(resolution: Option<PendingTaskResolution>) {
        if let Some((sender, order, outcome)) = resolution {
            let _ = sender.send((order, outcome));
        }
    }
}

/// Typed failure for the report that made a task terminal; durable diagnostics
/// from the scheduler record remain the message authority.
fn report_failure(
    report: &AttemptReport,
    state: TaskState,
    message: String,
) -> NativeProgramFailure {
    match report {
        AttemptReport::RuntimeFailed { failure } => NativeProgramFailure::Runtime(failure.clone()),
        AttemptReport::Failed {
            kind: AttemptFailureKind::Infrastructure,
            ..
        } => NativeProgramFailure::Infrastructure(message),
        AttemptReport::Lost { .. } | AttemptReport::RetirementUnverified { .. } => {
            NativeProgramFailure::WorkerLost(message)
        }
        AttemptReport::Cancelled | AttemptReport::GraphRetiredCancelled => {
            NativeProgramFailure::Cancelled
        }
        _ => state_failure(state, message),
    }
}

fn state_failure(state: TaskState, message: String) -> NativeProgramFailure {
    match state {
        TaskState::Cancelled => NativeProgramFailure::Cancelled,
        TaskState::Indeterminate => NativeProgramFailure::WorkerLost(message),
        _ => NativeProgramFailure::Execution(message),
    }
}

fn terminal_failure(driver: &Driver, task: &TaskRecord) -> Option<String> {
    let task_id = task.id();
    let message = match task.state {
        TaskState::Failed => "remote task failed",
        TaskState::Indeterminate => "remote worker was lost",
        TaskState::Cancelled
            if !driver
                .task_attempts(task_id)
                .expect("task is borrowed from this driver")
                .any(|attempt| {
                    matches!(
                        attempt.state,
                        AttemptState::Assigned | AttemptState::Starting | AttemptState::Running
                    )
                }) =>
        {
            "remote task was cancelled"
        }
        _ => return None,
    };
    Some(
        driver
            .task_attempts(task_id)
            .expect("task is borrowed from this driver")
            .max_by_key(|attempt| attempt.request.ordinal)
            .and_then(|attempt| attempt.terminal.as_ref())
            .and_then(|terminal| terminal.diagnostic.clone())
            .unwrap_or_else(|| message.into()),
    )
}

#[cfg(test)]
#[path = "../pool_tests/completion_authority.rs"]
mod tests;
