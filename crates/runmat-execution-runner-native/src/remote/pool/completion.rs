use std::sync::atomic::Ordering;
use std::sync::Arc;

use runmat_execution::state::TaskState;
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::{AttemptReport, DriverCommand};

use super::RemotePoolDriver;

type PendingTaskResolution = (
    tokio::sync::oneshot::Sender<(u64, super::CompletionResult)>,
    u64,
    super::CompletionResult,
);

impl RemotePoolDriver {
    pub(super) fn apply_report(self: &Arc<Self>, report: BackendReport) {
        let task_id = report.task_id;
        let spmd_task = self.spmd_task_context(task_id);
        let (actions, terminal, committed_results) = {
            let mut driver = self.driver.lock().expect("remote driver poisoned");
            let actions = match driver.handle(DriverCommand::BackendReport(report.clone())) {
                Ok(actions) => actions,
                Err(_) => return,
            };
            let snapshot = driver.snapshot();
            let committed_results = snapshot.tasks.get(&task_id).and_then(|task| {
                task.committed
                    .as_ref()
                    .filter(|commit| commit.attempt_id == report.attempt_id)
                    .map(|commit| commit.result.result_objects().to_vec())
            });
            let terminal = snapshot.tasks.get(&task_id).and_then(|task| {
                matches!(
                    task.state,
                    TaskState::Succeeded
                        | TaskState::Failed
                        | TaskState::Cancelled
                        | TaskState::Indeterminate
                )
                .then_some(task.state)
            });
            (actions, terminal, committed_results)
        };
        if let Some(results) = committed_results {
            if let Err(error) = self.execution_objects.commit_results(&results) {
                self.resolve_task(
                    task_id,
                    Err(crate::NativeProgramFailure::Infrastructure(
                        error.to_string(),
                    )),
                );
                return;
            }
        }
        self.dispatch(actions);
        if let Some(state) = terminal {
            let outcome = match report.report {
                AttemptReport::Succeeded { result } => Ok(result),
                AttemptReport::Failed { kind, message } => match kind {
                    runmat_execution_runner::AttemptFailureKind::Execution
                    | runmat_execution_runner::AttemptFailureKind::Rejected => {
                        Err(crate::NativeProgramFailure::Execution(message))
                    }
                    runmat_execution_runner::AttemptFailureKind::Infrastructure => {
                        Err(crate::NativeProgramFailure::Infrastructure(message))
                    }
                },
                AttemptReport::Lost { message } => {
                    Err(crate::NativeProgramFailure::WorkerLost(message))
                }
                AttemptReport::RuntimeFailed { failure } => {
                    Err(crate::NativeProgramFailure::Runtime(failure))
                }
                AttemptReport::Cancelled => Err(crate::NativeProgramFailure::Cancelled),
                AttemptReport::Started => {
                    Err(crate::NativeProgramFailure::Infrastructure(format!(
                        "remote task reached terminal state {state:?} without a terminal report"
                    )))
                }
            };
            // Reserve the originating task's deterministic completion order,
            // then fence its peers before making that completion observable.
            // Otherwise the gang waiter can cancel a blocked peer before the
            // collective broker has released it.
            let resolution = self.take_task_resolution(task_id, outcome);
            if let Some(task) = spmd_task {
                if matches!(
                    state,
                    TaskState::Failed | TaskState::Cancelled | TaskState::Indeterminate
                ) {
                    // Publish the originating result before releasing peers
                    // blocked in this gang's collective rounds.
                    self.collectives
                        .fail_gang(&task.gang, "an SPMD peer terminated");
                }
                Self::publish_task_resolution(resolution);
                self.collectives.rank_terminated(&task.gang, task.rank);
            } else {
                Self::publish_task_resolution(resolution);
            }
        }
    }

    pub(super) fn resolve_non_success_terminals(&self) {
        let terminal = {
            let snapshot = self
                .driver
                .lock()
                .expect("remote driver poisoned")
                .snapshot();
            snapshot
                .tasks
                .iter()
                .filter_map(|(task_id, task)| {
                    let message = match task.state {
                        TaskState::Failed => "remote task failed",
                        TaskState::Cancelled => "remote task was cancelled",
                        TaskState::Indeterminate => "remote worker was lost",
                        _ => return None,
                    };
                    Some((*task_id, task.state, message.to_string()))
                })
                .collect::<Vec<_>>()
        };
        for (task_id, state, message) in terminal {
            let spmd_task = self.spmd_task_context(task_id);
            let failure = match state {
                TaskState::Cancelled => crate::NativeProgramFailure::Cancelled,
                TaskState::Indeterminate => crate::NativeProgramFailure::WorkerLost(message),
                _ => crate::NativeProgramFailure::Execution(message),
            };
            let resolution = self.take_task_resolution(task_id, Err(failure));
            if let Some(task) = spmd_task {
                self.collectives
                    .fail_gang(&task.gang, "an SPMD peer terminated");
                Self::publish_task_resolution(resolution);
                self.collectives.rank_terminated(&task.gang, task.rank);
            } else {
                Self::publish_task_resolution(resolution);
            }
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
            .and_then(|program| match &program.context {
                runmat_execution::ProgramInvocationContext::SpmdTask { task } => Some(task.clone()),
                _ => None,
            })
    }

    fn resolve_task(&self, task_id: runmat_execution::TaskId, outcome: super::CompletionResult) {
        Self::publish_task_resolution(self.take_task_resolution(task_id, outcome));
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
        self.programs
            .lock()
            .expect("remote program catalog poisoned")
            .remove(&task_id);
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
