//! Original scheduler evidence and delivery outcome are separate authorities.
use super::*;
use runmat_execution::state::TaskState;

pub(crate) struct AnalysisTaskCompletion {
    pub response: ProgramExecutionResponse,
    pub attempt: Option<AttemptSuccess>,
    pub evidence: TaskExecutionEvidence,
}

impl AnalysisTaskCompletion {
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
        ) || evidence
            .attempts
            .iter()
            .any(|attempt| attempt.terminal.is_none())
        {
            return Err(protocol(
                "analysis completion still has active scheduler work",
            ));
        }
        let (response, attempt) = match result {
            Ok(success) => {
                let commit = evidence
                    .task
                    .committed
                    .as_ref()
                    .ok_or_else(|| protocol("delivered success lacks a scheduler commit"))?;
                if success.outputs != commit.outputs
                    || success.result_objects != commit.result_objects
                    || success.resources != commit.resources
                {
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
        })
    }
}

/// Surface the original delivery failure before asking for successful accounting.
pub(crate) fn delivered_attempt<'a>(
    response: &ProgramExecutionResponse,
    attempt: Option<&'a AttemptSuccess>,
) -> NativeExecutionResult<&'a AttemptSuccess> {
    if let ProgramExecutionResponse::Failure { message } = response {
        return Err(protocol(message));
    }
    attempt.ok_or_else(|| protocol("analysis task lacks scheduler attempt evidence"))
}
