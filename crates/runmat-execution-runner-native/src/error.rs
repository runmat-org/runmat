#[derive(Debug, thiserror::Error)]
pub enum NativeExecutionError {
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
    #[error("portable execution driver failed: {0}")]
    Driver(#[from] runmat_execution_runner::RunnerError),
}

pub type NativeExecutionResult<T> = Result<T, NativeExecutionError>;
