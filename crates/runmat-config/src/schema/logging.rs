use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level
    #[serde(default)]
    pub level: LogLevel,
    /// Enable debug logging
    #[serde(default)]
    pub debug: bool,
    /// Log file path
    pub file: Option<PathBuf>,
}

/// Log level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Warn,
            debug: false,
            file: None,
        }
    }
}

impl Default for LogLevel {
    fn default() -> Self {
        Self::Info
    }
}
