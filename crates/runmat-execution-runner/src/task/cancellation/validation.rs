use super::TaskCancellationCause;
use crate::{
    task::{AttemptDisposition, AttemptRecord, TaskRecord},
    RunnerError, RunnerResult,
};
use runmat_execution::state::TaskState;

pub(crate) fn validate_cancellation(
    task: &TaskRecord,
    history: &[&AttemptRecord],
    now_millis: u64,
) -> RunnerResult<()> {
    if (task.state == TaskState::Cancelled) != task.cancellation.is_some() {
        return Err(invalid(
            "task cancellation state lacks or contradicts its original cause",
        ));
    }
    let Some(cancellation) = task.cancellation else {
        return Ok(());
    };
    let at = cancellation.requested_at_driver_millis;
    if at > now_millis {
        return Err(invalid(
            "task cancellation is newer than its observed history",
        ));
    }
    match cancellation.cause {
        TaskCancellationCause::Task { task_id, .. } => {
            if task_id != task.submission.request.id {
                return Err(invalid("task cancellation references a foreign task"));
            }
        }
        TaskCancellationCause::Scope { .. } => {} // Scope ancestry belongs to the enclosing driver.
        TaskCancellationCause::Deadline => {
            if task
                .submission
                .request
                .deadline_unix_millis
                .is_none_or(|deadline| deadline > at)
            {
                return Err(invalid(
                    "task deadline cancellation contradicts its admitted deadline",
                ));
            }
        }
        TaskCancellationCause::WallTime { attempt_id } => {
            let attempt = history
                .iter()
                .find(|attempt| attempt.request.id == attempt_id)
                .ok_or_else(|| invalid("wall-time cancellation references a foreign attempt"))?;
            if attempt
                .assigned_at_millis
                .saturating_add(attempt.request.task.resources.max_wall_millis)
                > at
                || attempt.cancellation_requested_at != Some(at)
            {
                return Err(invalid(
                    "wall-time cancellation contradicts its original attempt interval",
                ));
            }
        }
        TaskCancellationCause::Provider { attempt_id } => {
            let attempt = history
                .iter()
                .find(|attempt| attempt.request.id == attempt_id)
                .ok_or_else(|| invalid("provider cancellation references a foreign attempt"))?;
            if attempt.cancellation_requested_at.is_some()
                || attempt.terminal.as_ref().is_none_or(|terminal| {
                    terminal.disposition != AttemptDisposition::Cancelled
                        || terminal.closed_at_driver_millis != at
                })
            {
                return Err(invalid(
                    "provider cancellation contradicts its original terminal report",
                ));
            }
        }
        TaskCancellationCause::Dependency { task_id } => {
            if !task.submission.dependencies.contains(&task_id) {
                return Err(invalid(
                    "dependency cancellation references an unadmitted dependency",
                ));
            }
        }
    }
    if history.iter().any(|attempt| {
        attempt
            .cancellation_requested_at
            .is_some_and(|requested| requested != at)
    }) {
        return Err(invalid(
            "task cancellation interval differs from its original attempts",
        ));
    }
    Ok(())
}
fn invalid(message: &str) -> RunnerError {
    RunnerError::Invalid(message.into())
}
