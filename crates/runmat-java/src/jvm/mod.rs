mod config;
#[cfg(not(target_family = "wasm"))]
mod discover;
#[cfg(not(target_family = "wasm"))]
mod lifecycle;

pub use config::{JvmConfig, JvmVersion};
#[cfg(not(target_family = "wasm"))]
pub use discover::{discover_jvm, JavaDiscoveryRequest, JvmInstallation};
#[cfg(not(target_family = "wasm"))]
pub use lifecycle::JvmProcess;

#[derive(Debug, thiserror::Error)]
pub enum JvmError {
    #[error("no compatible Java runtime was found{detail}")]
    NotFound { detail: String },
    #[error("Java runtime path {path} is invalid: {reason}")]
    InvalidInstallation { path: String, reason: String },
    #[error("Java runtime version {found} does not satisfy requested range {required}")]
    UnsupportedVersion { found: String, required: String },
    #[error("Java runtime configuration is invalid: {0}")]
    InvalidConfiguration(String),
    #[error("the process JVM is already running with incompatible configuration: {0}")]
    ConfigurationConflict(String),
    #[error("Java runtime startup failed: {0}")]
    Startup(String),
    #[error("Java thread attachment failed: {0}")]
    ThreadAttachment(String),
}
