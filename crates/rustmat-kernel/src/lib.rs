//! RustMat Jupyter Kernel
//!
//! A high-performance, V8-inspired Jupyter kernel for MATLAB/Octave code.
//! Implements the Jupyter messaging protocol over ZMQ with async execution.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub mod connection;
pub mod execution;
pub mod jupyter_plotting;
pub mod protocol;
pub mod server;

pub use connection::ConnectionInfo;
pub use execution::ExecutionEngine;
pub use jupyter_plotting::{JupyterPlottingManager, JupyterPlottingConfig, DisplayData, JupyterPlottingExtension};
pub use protocol::{ExecuteReply, ExecuteRequest, JupyterMessage, MessageType};
pub use server::KernelServer;

/// Kernel configuration and runtime state
#[derive(Debug, Clone)]
pub struct KernelConfig {
    /// Connection information for ZMQ sockets
    pub connection: ConnectionInfo,
    /// Kernel session identifier
    pub session_id: String,
    /// Whether to enable debug logging
    pub debug: bool,
    /// Maximum execution timeout in seconds
    pub execution_timeout: Option<u64>,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            connection: ConnectionInfo::default(),
            session_id: Uuid::new_v4().to_string(),
            debug: false,
            execution_timeout: Some(300), // 5 minutes
        }
    }
}

/// Kernel capability information reported to Jupyter
#[derive(Debug, Serialize, Deserialize)]
pub struct KernelInfo {
    pub protocol_version: String,
    pub implementation: String,
    pub implementation_version: String,
    pub language_info: LanguageInfo,
    pub banner: String,
    pub help_links: Vec<HelpLink>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LanguageInfo {
    pub name: String,
    pub version: String,
    pub mimetype: String,
    pub file_extension: String,
    pub pygments_lexer: String,
    pub codemirror_mode: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HelpLink {
    pub text: String,
    pub url: String,
}

impl Default for KernelInfo {
    fn default() -> Self {
        Self {
            protocol_version: "5.3".to_string(),
            implementation: "rustmat".to_string(),
            implementation_version: "0.0.1".to_string(),
            language_info: LanguageInfo {
                name: "matlab".to_string(),
                version: "R2025a-compatible".to_string(),
                mimetype: "text/x-matlab".to_string(),
                file_extension: ".m".to_string(),
                pygments_lexer: "matlab".to_string(),
                codemirror_mode: "octave".to_string(),
            },
            banner: "RustMat - High-performance MATLAB/Octave runtime".to_string(),
            help_links: vec![HelpLink {
                text: "RustMat Documentation".to_string(),
                url: "https://github.com/rustmat/rustmat".to_string(),
            }],
        }
    }
}

/// Error types for kernel operations
#[derive(Debug, thiserror::Error)]
pub enum KernelError {
    #[error("ZMQ error: {0}")]
    Zmq(#[from] zmq::Error),
    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Execution error: {0}")]
    Execution(String),
    #[error("Protocol error: {0}")]
    Protocol(String),
    #[error("Connection error: {0}")]
    Connection(String),
    #[error("Internal error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, KernelError>;
