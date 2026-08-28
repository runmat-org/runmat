use runmat_execution::identity::{AttemptId, WorkerId};
use runmat_execution::{PoolId, TaskId};

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum RunnerError {
    #[error("invalid execution driver request: {0}")]
    Invalid(String),
    #[error("task {0} does not exist")]
    UnknownTask(TaskId),
    #[error("pool {0} does not exist")]
    UnknownPool(PoolId),
    #[error("worker {0} does not exist")]
    UnknownWorker(WorkerId),
    #[error("attempt {0} does not exist")]
    UnknownAttempt(AttemptId),
    #[error("task graph contains a dependency cycle")]
    DependencyCycle,
    #[error("execution backend failed: {0}")]
    Backend(String),
    #[error("checkpoint failed: {0}")]
    Checkpoint(String),
}

pub type RunnerResult<T> = Result<T, RunnerError>;
