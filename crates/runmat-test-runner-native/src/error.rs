#[derive(Debug, thiserror::Error)]
pub enum NativeRunnerError {
    #[error("native test-runner I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("native test-runner {peer} closed its control stream: {source}")]
    ControlClosed {
        peer: ControlPeer,
        #[source]
        source: std::io::Error,
    },
    #[error("native test-runner protocol failed: {0}")]
    Protocol(String),
    #[error("native test-runner configuration is invalid: {0}")]
    Configuration(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ControlPeer {
    Worker,
    Coordinator,
}

impl std::fmt::Display for ControlPeer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Worker => "worker",
            Self::Coordinator => "coordinator",
        })
    }
}

pub type NativeRunnerResult<T> = Result<T, NativeRunnerError>;
