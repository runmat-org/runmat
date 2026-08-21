use std::sync::Arc;

use runmat_execution::state::TaskState;
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::{AttemptReport, DriverCommand};

use super::RemotePoolDriver;

impl RemotePoolDriver {
    pub(super) fn apply_report(self: &Arc<Self>, report: BackendReport) {
        let task_id = report.task_id;
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
                    .map(|commit| commit.result_objects.clone())
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
                self.resolve_task(task_id, Err(error.to_string()));
                return;
            }
        }
        self.dispatch(actions);
        if let Some(state) = terminal {
            let outcome = match report.report {
                AttemptReport::Succeeded { result } => Ok(result),
                AttemptReport::Failed { message, .. } | AttemptReport::Lost { message } => {
                    Err(message)
                }
                AttemptReport::Cancelled => Err("remote task was cancelled".into()),
                AttemptReport::Started => Err(format!(
                    "remote task reached terminal state {state:?} without a terminal report"
                )),
            };
            self.resolve_task(task_id, outcome);
        }
    }

    pub(super) fn resolve_non_success_terminals(&self) {
        let terminal = self
            .driver
            .lock()
            .expect("remote driver poisoned")
            .snapshot()
            .tasks
            .iter()
            .filter_map(|(task_id, task)| {
                let message = match task.state {
                    TaskState::Failed => "remote task failed",
                    TaskState::Cancelled => "remote task was cancelled",
                    TaskState::Indeterminate => "remote worker was lost",
                    _ => return None,
                };
                Some((*task_id, message.to_string()))
            })
            .collect::<Vec<_>>();
        for (task_id, message) in terminal {
            self.resolve_task(task_id, Err(message));
        }
    }

    fn resolve_task(&self, task_id: runmat_execution::TaskId, outcome: super::CompletionResult) {
        if let Some(sender) = self
            .completions
            .lock()
            .expect("remote completion registry poisoned")
            .remove(&task_id)
        {
            let _ = sender.send(outcome);
        }
        self.programs
            .lock()
            .expect("remote program catalog poisoned")
            .remove(&task_id);
        self.progress
            .lock()
            .expect("remote progress registry poisoned")
            .remove(&task_id);
    }
}
