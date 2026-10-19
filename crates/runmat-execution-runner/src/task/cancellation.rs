//! Original driver cancellation decisions; imported metadata is not provider attestation.
use runmat_execution::{identity::AttemptId, CancellationReason, ExecutionScopeId, TaskId};
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
