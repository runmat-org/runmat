use crate::task::{AttemptRecord, ResultCommit, TaskRecord};
use crate::{RunnerError, RunnerResult};
use runmat_execution::identity::AttemptId;
use runmat_execution::state::{AttemptState, TaskState};
use std::collections::BTreeSet;

pub(crate) fn validate_history(
    task: &TaskRecord,
    history: &[&AttemptRecord],
    driver_fence: u64,
    now_millis: u64,
) -> RunnerResult<()> {
    let retry = task.submission.retry_resource_envelope()?;
    if driver_fence == 0
        || task.attempt_count > retry.maximum_attempts
        || history.len() != usize::from(task.attempt_count)
    {
        return Err(RunnerError::Invalid(
            "task history differs from its admitted retry envelope".into(),
        ));
    }
    let mut ordinals = BTreeSet::new();
    for attempt in history {
        attempt
            .validate_terminal(now_millis)
            .map_err(RunnerError::Invalid)?;
        let request = &attempt.request;
        let expected_id = AttemptId::derive(&[
            task.id().bytes().as_slice(),
            request.ordinal.to_be_bytes().as_slice(),
            request.driver_fence.to_be_bytes().as_slice(),
        ]);
        if request.id != expected_id
            || request.task_id != task.id()
            || request.task != task.submission.request
            || request.scope_id != task.submission.request.scope_id
            || request.ordinal == 0
            || request.ordinal > task.attempt_count
            || request.driver_fence == 0
            || request.driver_fence > driver_fence
            || !ordinals.insert(request.ordinal)
        {
            return Err(RunnerError::Invalid(
                "attempt differs from its admitted task or fence".into(),
            ));
        }
    }
    match task.active_attempt {
        Some(id) => {
            let attempt = history
                .iter()
                .find(|attempt| attempt.request.id == id)
                .ok_or_else(|| {
                    RunnerError::Invalid("task references an unknown active attempt".into())
                })?;
            if attempt.terminal.is_some()
                || attempt.request.ordinal != task.attempt_count
                || !(matches!(
                    task.state,
                    TaskState::Assigned | TaskState::Running | TaskState::Committing
                ) || (task.state == TaskState::Indeterminate
                    && attempt
                        .unverified_retirement
                        .as_ref()
                        .is_some_and(|evidence| evidence.retired_at_driver_millis.is_none())))
            {
                return Err(RunnerError::Invalid(
                    "active attempt differs from the original task lifecycle".into(),
                ));
            }
        }
        None if matches!(
            task.state,
            TaskState::Assigned | TaskState::Running | TaskState::Committing
        ) =>
        {
            return Err(RunnerError::Invalid(
                "active task lacks its original attempt".into(),
            ));
        }
        None => {}
    }
    if history.iter().any(|attempt| {
        attempt.terminal.is_none() && task.active_attempt != Some(attempt.request.id)
    }) {
        return Err(RunnerError::Invalid(
            "live physical attempt lacks its original task reservation".into(),
        ));
    }
    super::super::cancellation::validate_cancellation(task, history, now_millis)?;
    validate_commit(task, history)
}

pub(crate) fn validate_commit(task: &TaskRecord, history: &[&AttemptRecord]) -> RunnerResult<()> {
    if (task.state == TaskState::Succeeded) != task.committed.is_some() {
        return Err(RunnerError::Invalid(
            "snapshot task success and committed result disagree".into(),
        ));
    }
    let Some(commit) = &task.committed else {
        return Ok(());
    };
    let attempt = history
        .iter()
        .copied()
        .find(|a| a.request.id == commit.attempt_id)
        .ok_or_else(|| {
            RunnerError::Invalid("snapshot commit references an unknown attempt".into())
        })?;
    if task.active_attempt.is_some()
        || attempt.state != AttemptState::Completed
        || attempt.request.task_id != task.id()
        || attempt.request.ordinal != task.attempt_count
        || commit.driver_fence != attempt.request.driver_fence
    {
        return Err(RunnerError::Invalid(
            "snapshot commit does not belong to a completed task attempt".into(),
        ));
    }
    if attempt.terminal.as_ref().is_none_or(|terminal| {
        terminal.disposition != crate::task::AttemptDisposition::Succeeded
            || terminal.resources.as_deref() != commit.result.resources()
    }) {
        return Err(RunnerError::Invalid(
            "commit differs from original terminal attempt evidence".into(),
        ));
    }
    let success = commit.result.clone();
    success
        .validate_against(&attempt.request)
        .map_err(RunnerError::Invalid)?;
    if *commit != ResultCommit::from_success(commit.attempt_id, commit.driver_fence, success) {
        return Err(RunnerError::Invalid(
            "snapshot commit identity is invalid".into(),
        ));
    }
    Ok(())
}
