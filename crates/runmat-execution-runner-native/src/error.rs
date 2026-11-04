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
    Driver(#[from] runmat_execution_runner::RunnerError),
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
