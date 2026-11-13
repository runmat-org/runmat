//! Project a validated supervisor attachment without erasing runtime failures.
use runmat_execution_artifact::ProgramExecutionResponse;
use runmat_runtime::execution::ExecutionServiceError;

use crate::supervisor::LocalJobState;

pub(super) fn terminal_response(
    state: LocalJobState,
    message: Option<String>,
    response: Option<ProgramExecutionResponse>,
) -> Result<Option<ProgramExecutionResponse>, ExecutionServiceError> {
    match (state, response) {
        (LocalJobState::Succeeded, Some(response)) => Ok(Some(response)),
        (LocalJobState::Succeeded, None) => Err(ExecutionServiceError::Failed(
            "durable program completed without a result".into(),
        )),
        (
            LocalJobState::Failed,
            Some(response @ ProgramExecutionResponse::RuntimeFailure { .. }),
        ) => Ok(Some(response)),
        // A cancelled or indeterminate job cannot publish a late successful
        // worker response. The scheduler's adopted disposition remains final.
        (LocalJobState::Failed | LocalJobState::Indeterminate, _) => {
            Err(ExecutionServiceError::Failed(
                message.unwrap_or_else(|| "durable program failed".into()),
            ))
        }
        (LocalJobState::Cancelled, _) => Err(ExecutionServiceError::Cancelled),
        (
            LocalJobState::Queued
            | LocalJobState::Starting
            | LocalJobState::Running
            | LocalJobState::Cancelling,
            _,
        ) => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adopted_state_preserves_typed_failure_without_resurrecting_worker_success() {
        let failure = ProgramExecutionResponse::RuntimeFailure {
            failure: runmat_execution::ProgramRuntimeFailure {
                message: "original runtime failure".into(),
                identifier: Some("RunMat:IndexOutOfBounds".into()),
                span: None,
                builtin: None,
                task_id: None,
                call_frames: Vec::new(),
                call_frames_elided: 0,
                call_stack: Vec::new(),
                phase: None,
            },
        };
        assert_eq!(
            terminal_response(
                LocalJobState::Failed,
                Some("coarse status".into()),
                Some(failure.clone())
            )
            .unwrap(),
            Some(failure.clone())
        );
        for state in [
            LocalJobState::Queued,
            LocalJobState::Starting,
            LocalJobState::Running,
            LocalJobState::Cancelling,
        ] {
            assert_eq!(
                terminal_response(state, None, Some(failure.clone())).unwrap(),
                None
            );
        }
        for state in [
            LocalJobState::Failed,
            LocalJobState::Cancelled,
            LocalJobState::Indeterminate,
        ] {
            assert!(terminal_response(
                state,
                Some("original supervisor disposition".into()),
                Some(ProgramExecutionResponse::Success {
                    outputs: Vec::new()
                })
            )
            .is_err());
        }
        assert_eq!(
            terminal_response(
                LocalJobState::Succeeded,
                None,
                Some(ProgramExecutionResponse::Success {
                    outputs: Vec::new()
                })
            )
            .unwrap(),
            Some(ProgramExecutionResponse::Success {
                outputs: Vec::new()
            })
        );
        assert!(terminal_response(LocalJobState::Succeeded, None, None).is_err());
    }
}
