#[derive(Debug, thiserror::Error)]
pub enum NativeExecutionError {
    #[error("native process output failed: {0}")]
    ProcessOutput(#[source] std::sync::Arc<crate::RawOutputError>),
    #[error("native metadata {operation:?} at {} failed: {source}", path.display())]
    MetadataIo {
        operation: MetadataIoOperation,
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("native execution was cancelled")]
    Cancelled,
    #[error("native task {task_id} was cancelled: {cancellation:?}; quiescence={quiescence:?}")]
    TaskCancelled {
        task_id: runmat_execution::TaskId,
        cancellation: runmat_execution_runner::TaskCancellation,
        quiescence: crate::NativeTaskQuiescence,
    },
    #[error("{diagnostic}")]
    CheckpointDelivery {
        diagnostic: String,
        evidence: Box<crate::NativeCheckpointEvidence>,
    },
    #[error("invalid native execution configuration: {0}")]
    Configuration(String),
    #[error("native process host failed: {0}")]
    Host(#[from] runmat_process_host::ProcessHostError),
    #[error("remote worker channel failed: {source}")]
    RemoteChannel {
        #[source]
        source: std::sync::Arc<NativeExecutionError>,
    },
    #[error("local supervisor transport failed: {source}")]
    SupervisorTransport {
        #[source]
        source: std::io::Error,
    },
    #[error("local supervisor startup failed: {cause}; exit={exit:?}; stderr={stderr}")]
    SupervisorStartup {
        #[source]
        cause: Box<NativeExecutionError>,
        exit: Option<runmat_process_host::ProcessExit>,
        stderr: Box<runmat_process_host::ProcessOutputSnapshot>,
    },
    #[error("local supervisor rejected the request ({code}): {message}")]
    SupervisorRejected { code: String, message: String },
    #[error("native bounded encoding failed: {0}")]
    Encoding(#[from] runmat_canonical_codec::CanonicalCodecError),
    #[error("native execution protocol failed: {0}")]
    Protocol(String),
    #[error("native simulation operation accounting failed: {0}")]
    Usage(#[from] runmat_simulation_usage::LedgerError),
    #[error(
        "cooperative rank solving has no parent-owned operation grants for an admitted FEA run"
    )]
    UnsupportedMeteredCooperative,
    #[error("portable execution driver failed: {0}")]
    Driver(runmat_execution_runner::RunnerError),
    /// A typed resource refusal: a plan that fits no admitted path, or a task whose memory or
    /// scratch request exceeds what the target pool offers.
    #[error("FEA resource planning failed: {0}")]
    Resource(runmat_analysis_fea::planning::resource::ResourcePlanningError),
}

impl From<runmat_execution_runner::RunnerError> for NativeExecutionError {
    /// A pool refusal for memory or scratch is the typed memory-budget error, with the bytes
    /// the task needs and the bytes the pool offers; every other driver failure stays a driver
    /// error.
    fn from(error: runmat_execution_runner::RunnerError) -> Self {
        match error {
            runmat_execution_runner::RunnerError::PoolCapacity(refusal)
                if refusal.exceeds_memory() =>
            {
                Self::Resource(
                    runmat_analysis_fea::planning::resource::ResourcePlanningError::MemoryBudget {
                        required_memory_bytes: refusal.requested_memory_bytes,
                        available_memory_bytes: refusal.pool_memory_bytes,
                        required_scratch_bytes: refusal.requested_scratch_bytes,
                        available_scratch_bytes: refusal.pool_scratch_bytes,
                    },
                )
            }
            error => Self::Driver(error),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetadataIoOperation {
    PreparePrivateDirectory,
    AtomicReplacement,
}

pub type NativeExecutionResult<T> = Result<T, NativeExecutionError>;

impl From<crate::RawOutputError> for NativeExecutionError {
    fn from(error: crate::RawOutputError) -> Self {
        Self::ProcessOutput(std::sync::Arc::new(error))
    }
}
