use runmat_execution::identity::{AttemptId, WorkerId};
use runmat_execution::{PoolId, TaskId};

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum RunnerError {
    #[error("invalid execution driver request: {0}")]
    Invalid(String),
    /// The task's resource request exceeds what the target pool or every one of its workers
    /// offers.
    #[error("task resource request cannot be satisfied by the target pool: {0}")]
    PoolCapacity(Box<PoolCapacityRefusal>),
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

/// The request and the pool limits behind a [`RunnerError::PoolCapacity`] refusal.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PoolCapacityRefusal {
    pub requested_cpu_millicores: u32,
    pub requested_memory_bytes: u64,
    pub requested_scratch_bytes: u64,
    pub pool_cpu_millicores: u32,
    pub pool_memory_bytes: u64,
    pub pool_scratch_bytes: u64,
    pub missing_capabilities: Vec<String>,
    /// Up to two workers' offers, for the diagnostic.
    pub workers: Vec<String>,
}

impl PoolCapacityRefusal {
    /// Whether the request needs more memory or scratch than the pool offers.
    pub fn exceeds_memory(&self) -> bool {
        self.requested_memory_bytes > self.pool_memory_bytes
            || self.requested_scratch_bytes > self.pool_scratch_bytes
    }
}

impl std::fmt::Display for PoolCapacityRefusal {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "requested {} millicores, {} memory bytes, {} scratch bytes; pool limit {} millicores, \
             {} memory bytes, {} scratch bytes, missing capabilities {:?}; {:?}",
            self.requested_cpu_millicores,
            self.requested_memory_bytes,
            self.requested_scratch_bytes,
            self.pool_cpu_millicores,
            self.pool_memory_bytes,
            self.pool_scratch_bytes,
            self.missing_capabilities,
            self.workers,
        )
    }
}
