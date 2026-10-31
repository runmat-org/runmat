use std::collections::BTreeSet;

use runmat_execution::identity::AttemptId;
use runmat_execution::state::TaskState;
use runmat_execution::task::TaskRequest;
use runmat_execution::TaskId;
use serde::{Deserialize, Serialize};

use super::ResultCommit;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TaskSubmission {
    pub request: TaskRequest,
    pub dependencies: BTreeSet<TaskId>,
    pub priority: i16,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TaskRecord {
    pub submission: TaskSubmission,
    pub state: TaskState,
    #[serde(deserialize_with = "super::cancellation::deserialize_cancellation")]
    pub cancellation: Option<super::TaskCancellation>,
    pub attempt_count: u16,
    pub active_attempt: Option<AttemptId>,
    pub committed: Option<ResultCommit>,
    /// Exact dependency that propagated unresolved physical ownership here.
    /// Absent for this task's own uncertainty or an independent terminal cause.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub indeterminate_dependency: Option<TaskId>,
    pub enqueued_sequence: u64,
}

impl TaskRecord {
    pub fn new(submission: TaskSubmission, sequence: u64) -> Self {
        let state = if submission.dependencies.is_empty() {
            TaskState::Ready
        } else {
            TaskState::Deferred
        };
        Self {
            submission,
            state,
            cancellation: None,
            attempt_count: 0,
            active_attempt: None,
            committed: None,
            indeterminate_dependency: None,
            enqueued_sequence: sequence,
        }
    }

    pub fn id(&self) -> TaskId {
        self.submission.request.id
    }
}
