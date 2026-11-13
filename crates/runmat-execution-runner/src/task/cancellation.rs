//! Original driver cancellation decisions; imported metadata is not provider attestation.
use runmat_execution::{identity::AttemptId, CancellationReason, ExecutionScopeId, GangId, TaskId};
use serde::{Deserialize, Deserializer, Serialize};

mod validation;
pub(crate) use validation::validate_cancellation;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "cause", rename_all = "snake_case", deny_unknown_fields)]
pub enum TaskCancellationCause {
    Task {
        task_id: TaskId,
        reason: CancellationReason,
    },
    Scope {
        scope_id: ExecutionScopeId,
        reason: CancellationReason,
    },
    Gang {
        gang_id: GangId,
        generation: u64,
        reason: CancellationReason,
    },
    Deadline,
    WallTime {
        attempt_id: AttemptId,
    },
    Provider {
        attempt_id: AttemptId,
    },
    Dependency {
        task_id: TaskId,
    },
}

impl TaskCancellationCause {
    /// A provider observation is not authority to manufacture a user stop.
    /// Preserve the scheduler's original cause and its actual attempt fence.
    pub fn reason_for_attempt(
        self,
        request: &super::AttemptRequest,
    ) -> Result<CancellationReason, String> {
        match self {
            Self::Task { task_id, reason } if task_id == request.task_id => Ok(reason),
            Self::Scope { reason, .. } => Ok(reason),
            Self::Gang {
                gang_id,
                generation,
                reason,
            } if matches!(&request.task.invocation_context,
                    runmat_execution::ProgramInvocationContext::SpmdTask { task }
                        if task.gang.id == gang_id && task.gang.generation == generation) =>
            {
                Ok(reason)
            }
            Self::Deadline if request.task.deadline_unix_millis.is_some() => {
                Ok(CancellationReason::Deadline)
            }
            Self::WallTime { attempt_id } if attempt_id == request.id => {
                Ok(CancellationReason::Deadline)
            }
            Self::Dependency { .. } => Ok(CancellationReason::DependencyFailed),
            Self::Provider { .. } => {
                Err("provider observation is not a scheduler cancellation command".into())
            }
            _ => Err("scheduler cancellation cause contradicts its fenced attempt".into()),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskCancellation {
    pub cause: TaskCancellationCause,
    pub requested_at_driver_millis: u64,
}

// An explicit null belongs to a noncancelled task; absence is an older/incomplete format.
pub(super) fn deserialize_cancellation<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<TaskCancellation>, D::Error> {
    Option::<TaskCancellation>::deserialize(deserializer)
}
