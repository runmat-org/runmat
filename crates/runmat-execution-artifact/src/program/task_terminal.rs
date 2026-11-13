//! Selected-program task control, separate from literal worker-process exit.
use super::ProgramExecutionResponse;
use crate::{ArtifactError, ArtifactResult};
use runmat_execution::{CancellationReason, ProgramExecutionAssignment, ProgramRuntimeFailure};
use serde::{Deserialize, Serialize};

pub const PROGRAM_TASK_TERMINAL_SCHEMA_VERSION: u16 = 1;
pub const PROGRAM_TASK_CONTROL_SCHEMA_VERSION: u16 = 1;

/// A wire report, not authority on its own. Only the actual invocation owner
/// can issue Retired after its roots, child work and external obligations end.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum ProgramTaskRetirement {
    Retired,
    Unverified {
        diagnostic: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cleanup_failure: Option<Box<ProgramRuntimeFailure>>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramTaskTerminal {
    pub schema_version: u16,
    /// The actual scheduler assignment, never a task-id-only reconstruction.
    pub assignment: ProgramExecutionAssignment,
    /// None is allowed only for a same-invocation already-retired or
    /// never-started acknowledgment. Launch consumers require a real response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response: Option<ProgramExecutionResponse>,
    pub retirement: ProgramTaskRetirement,
}
impl ProgramTaskTerminal {
    pub fn validate_assignment(&self, expected: &ProgramExecutionAssignment) -> ArtifactResult<()> {
        if self.schema_version != PROGRAM_TASK_TERMINAL_SCHEMA_VERSION {
            return Err(ArtifactError::Invalid(
                "unsupported program task terminal schema".into(),
            ));
        }
        self.assignment
            .validate()
            .map_err(crate::error::contract_error)?;
        if &self.assignment != expected {
            return Err(ArtifactError::Identity(
                "program task terminal belongs to another exact assignment".into(),
            ));
        }
        if let ProgramTaskRetirement::Unverified {
            diagnostic,
            cleanup_failure,
        } = &self.retirement
        {
            if diagnostic.trim().is_empty() {
                return Err(ArtifactError::Invalid(
                    "unverified retirement requires a diagnostic".into(),
                ));
            }
            if let Some(failure) = cleanup_failure {
                failure.validate().map_err(crate::error::contract_error)?;
            }
        }
        Ok(())
    }
}

/// Cancellation requests carry the same actual assignment and original
/// reason. Returning from cancel requests a stop; it never proves retirement.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramTaskCancellation {
    pub schema_version: u16,
    pub assignment: ProgramExecutionAssignment,
    pub reason: CancellationReason,
}
impl ProgramTaskCancellation {
    pub fn validate_assignment(&self, expected: &ProgramExecutionAssignment) -> ArtifactResult<()> {
        if self.schema_version != PROGRAM_TASK_CONTROL_SCHEMA_VERSION {
            return Err(ArtifactError::Invalid(
                "unsupported program task control schema".into(),
            ));
        }
        self.assignment
            .validate()
            .map_err(crate::error::contract_error)?;
        if &self.assignment != expected {
            return Err(ArtifactError::Identity(
                "program task cancellation belongs to another exact assignment".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::identity::{AttemptId, WorkerId};
    use runmat_execution::{ExecutionScopeId, PoolBackend, PoolId, TaskId};

    fn assignment() -> ProgramExecutionAssignment {
        ProgramExecutionAssignment {
            scope_id: ExecutionScopeId::derive(&[b"scope"]),
            pool_id: PoolId::derive(&[b"pool"]),
            task_id: TaskId::derive(&[b"task"]),
            attempt_id: AttemptId::derive(&[b"attempt"]),
            worker_id: WorkerId::derive(&[b"worker"]),
            backend: PoolBackend::BrowserWorkers,
            resources: Default::default(),
        }
    }

    #[test]
    fn program_retirement_and_cancellation_preserve_the_full_attempt_fence() {
        let expected = assignment();
        let terminal = ProgramTaskTerminal {
            schema_version: PROGRAM_TASK_TERMINAL_SCHEMA_VERSION,
            assignment: expected.clone(),
            response: Some(ProgramExecutionResponse::Failure {
                message: "original operation failed".into(),
            }),
            retirement: ProgramTaskRetirement::Unverified {
                diagnostic: "child retirement not proved".into(),
                cleanup_failure: None,
            },
        };
        let encoded = serde_json::to_vec(&terminal).expect("encode");
        let decoded: ProgramTaskTerminal = serde_json::from_slice(&encoded).expect("decode");
        assert_eq!(decoded, terminal);
        decoded
            .validate_assignment(&expected)
            .expect("exact assignment");
        let mut other = expected.clone();
        other.attempt_id = AttemptId::derive(&[b"another-attempt"]);
        assert!(decoded.validate_assignment(&other).is_err());
        let cancellation = ProgramTaskCancellation {
            schema_version: PROGRAM_TASK_CONTROL_SCHEMA_VERSION,
            assignment: expected.clone(),
            reason: CancellationReason::DependencyFailed,
        };
        cancellation
            .validate_assignment(&expected)
            .expect("exact control");
        assert!(cancellation.validate_assignment(&other).is_err());
        assert_eq!(
            serde_json::from_slice::<ProgramTaskCancellation>(
                &serde_json::to_vec(&cancellation).expect("encode control")
            )
            .expect("decode control"),
            cancellation
        );
    }
}
