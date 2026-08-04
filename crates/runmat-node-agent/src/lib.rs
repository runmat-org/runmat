pub mod allocation;
pub mod cli;
pub mod config;
pub mod enrollment;
pub mod inventory;
pub mod platform;
pub mod service;
pub mod service_install;

pub use config::{AgentConfig, AgentFileConfig};

#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("agent configuration is invalid: {0}")]
    Configuration(String),
    #[error("node is not enrolled")]
    NotEnrolled,
    #[error("node credential file is unsafe: {0}")]
    UnsafeCredential(String),
    #[error("allocation was rejected: {0}")]
    AllocationRejected(String),
    #[error("agent transport failed: {0}")]
    Transport(#[from] runmat_execution_transport_native::TransportError),
    #[error("agent process failed: {0}")]
    Process(#[from] runmat_process_host::ProcessHostError),
    #[error("agent I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("agent state is malformed: {0}")]
    State(#[from] serde_json::Error),
}

pub type AgentResult<T> = Result<T, AgentError>;
