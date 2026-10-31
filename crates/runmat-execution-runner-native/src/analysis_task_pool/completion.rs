//! Original scheduler evidence and delivery outcome are separate authorities.
use super::*;
use runmat_execution::state::TaskState;

pub(crate) struct AnalysisTaskCompletion {
    pub response: ProgramExecutionResponse,
    pub attempt: Option<AttemptSuccess>,
    pub evidence: TaskExecutionEvidence,
    pub failure: Option<crate::task_outcome::NativeTaskFailure>,
    pub original_graph_wall_millis: Option<u64>,
}

impl AnalysisTaskCompletion {
    pub(crate) fn from_native(
        completion: crate::NativeProgramCompletion,
    ) -> NativeExecutionResult<Self> {
        let (result, evidence, checkpoint) = completion.into_parts();
        let checkpoint_failure =
            crate::task_outcome::NativeTaskFailure::from_checkpoint(&checkpoint);
        let evidence = evidence.ok_or_else(|| {
            checkpoint_failure.as_ref().map_or_else(
                || protocol("local completion lacks original scheduler evidence"),
                |failure| failure.error(),
            )
        })?;
        let mut completed = Self::new(evidence, result)?;
        if let Some(failure) = checkpoint_failure {
            completed.response = ProgramExecutionResponse::Failure {
                message: checkpoint
                    .delivery_error()
                    .expect("failed checkpoint has a diagnostic"),
            };
            completed.attempt = None;
            completed.failure = Some(failure);
        }
        Ok(completed)
    }
    pub(crate) fn new(
        evidence: TaskExecutionEvidence,
        result: Result<AttemptSuccess, String>,
    ) -> NativeExecutionResult<Self> {
        evidence.validate().map_err(protocol)?;
        if !matches!(
            evidence.task.state,
            TaskState::Succeeded
                | TaskState::Failed
                | TaskState::Cancelled
                | TaskState::Indeterminate
        ) || evidence.attempts.iter().any(|attempt| {
            attempt.terminal.is_none()
                && !(result.is_err()
                    && evidence.task.state == TaskState::Indeterminate
                    && attempt
                        .unverified_retirement
                        .as_ref()
                        .is_some_and(|evidence| evidence.retired_at_driver_millis.is_none()))
        }) {
            return Err(protocol(
                "analysis completion still has active scheduler work",
            ));
        }
        let failure = crate::task_outcome::NativeTaskFailure::from_scheduler(&evidence, &result);
        let (response, attempt) = match result {
            Ok(success) => {
                let commit = evidence
                    .task
                    .committed
                    .as_ref()
                    .ok_or_else(|| protocol("delivered success lacks a scheduler commit"))?;
                if success != commit.result {
                    return Err(protocol(
                        "delivered success differs from the original scheduler commit",
                    ));
                }
                (success_response(success.clone()), Some(success))
            }
            Err(message) => (ProgramExecutionResponse::Failure { message }, None),
        };
        Ok(Self {
            response,
            attempt,
            evidence,
            failure,
            original_graph_wall_millis: None,
        })
    }
}

/// Surface the original delivery failure before asking for successful accounting.
pub(crate) fn delivered_attempt<'a>(
    response: &ProgramExecutionResponse,
    attempt: Option<&'a AttemptSuccess>,
    failure: Option<&crate::task_outcome::NativeTaskFailure>,
) -> NativeExecutionResult<&'a AttemptSuccess> {
    if let Some(failure) = failure {
        return Err(failure.error());
    }
    if let ProgramExecutionResponse::Failure { message } = response {
        return Err(protocol(message));
    }
    attempt.ok_or_else(|| protocol("analysis task lacks scheduler attempt evidence"))
}
