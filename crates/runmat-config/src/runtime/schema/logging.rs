use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoggingConfig {
    /// Log level
    #[serde(default)]
    pub level: LogLevel,
    /// Enable debug logging
    #[serde(default)]
    pub debug: bool,
    /// Log file path
    pub file: Option<PathBuf>,
    /// User-controlled collector export, independent of product telemetry.
    #[serde(default)]
    pub otel: OtelConfig,
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
            otel: OtelConfig::default(),
        }
    }
}

/// Serializable collector policy. Resolved header secrets must never be stored here.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OtelConfig {
    pub enabled: bool,
    pub endpoint: String,
    pub protocol: OtelProtocol,
    pub signals: Vec<OtelSignal>,
    pub headers_from_env: Option<String>,
    pub service_name: String,
    pub resource_attributes: BTreeMap<String, String>,
}

impl Default for OtelConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoint: "http://localhost:4318".into(),
            protocol: OtelProtocol::HttpProtobuf,
            signals: vec![OtelSignal::Traces, OtelSignal::Logs],
            headers_from_env: Some("RUNMAT_OTEL_HEADERS".into()),
            service_name: "runmat".into(),
            resource_attributes: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OtelProtocol {
    #[serde(rename = "http/protobuf")]
    HttpProtobuf,
    #[serde(rename = "grpc")]
    Grpc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OtelSignal {
    Traces,
    Logs,
    Metrics,
}

impl Default for LogLevel {
    fn default() -> Self {
        Self::Warn
    }
}
