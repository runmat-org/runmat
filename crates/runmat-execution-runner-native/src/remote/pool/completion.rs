use std::sync::atomic::Ordering;
use std::sync::Arc;

use runmat_execution::state::{AttemptState, TaskState};
use runmat_execution::TaskId;
use runmat_execution_runner::driver::DriverEventKind;
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::{AttemptSuccess, DriverAction, DriverCommand, DriverSnapshot};

use super::RemotePoolDriver;

impl RemotePoolDriver {
    pub(super) fn apply_report(self: &Arc<Self>, report: BackendReport) {
        let task_id = report.task_id;
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
            let snapshot = driver.snapshot();
            let outcome = snapshot.tasks.get(&task_id).and_then(|task| {
                if task.state == TaskState::Succeeded {
                    Some(match &task.committed {
                        Some(commit) => self
                            .execution_objects
                            .commit_results(&commit.result_objects)
                            .map(|()| AttemptSuccess {
                                outputs: commit.outputs.clone(),
                                result_objects: commit.result_objects.clone(),
                                resources: commit.resources.clone(),
                            })
                            .map_err(|error| error.to_string()),
                        None => Err("succeeded remote task has no committed result".into()),
                    })
                } else {
                    terminal_failure(&snapshot, task_id, task.state).map(Err)
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
            self.resolve_task(task_id, outcome);
        }
    }

    pub(super) fn resolve_non_success_terminals(&self) {
        let snapshot = self
            .driver
            .lock()
            .expect("remote driver poisoned")
            .snapshot();
        let terminal = snapshot
            .tasks
            .iter()
            .filter_map(|(task_id, task)| {
                terminal_failure(&snapshot, *task_id, task.state).map(|message| (*task_id, message))
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
        // Exact admitted programs have the same lifetime as scheduler task records:
        // completed-task replay and durable recovery still need their authority.
        self.progress
            .lock()
            .expect("remote progress registry poisoned")
            .remove(&task_id);
    }
}

fn terminal_failure(
    snapshot: &DriverSnapshot,
    task_id: TaskId,
    state: TaskState,
) -> Option<String> {
    let message = match state {
        TaskState::Failed => "remote task failed",
        TaskState::Indeterminate => "remote worker was lost",
        TaskState::Cancelled
            if !snapshot.attempts.values().any(|attempt| {
                attempt.request.task_id == task_id
                    && matches!(
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
        snapshot
            .attempts
            .values()
            .filter(|attempt| attempt.request.task_id == task_id)
            .max_by_key(|attempt| attempt.request.ordinal)
            .and_then(|attempt| attempt.terminal.as_ref())
            .and_then(|terminal| terminal.diagnostic.clone())
            .unwrap_or_else(|| message.into()),
    )
}

#[cfg(test)]
#[path = "../pool_tests/completion_authority.rs"]
mod tests;
