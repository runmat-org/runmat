use clap::ValueEnum;
use serde::{Deserialize, Serialize};

use super::defaults::default_true;

/// Telemetry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    /// Enable runtime telemetry
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Echo each payload to stdout for transparency
    #[serde(default)]
    pub show_payloads: bool,
    /// Optional HTTP endpoint override
    pub http_endpoint: Option<String>,
    /// Optional UDP endpoint override (host:port)
    pub udp_endpoint: Option<String>,
    /// Bounded queue size for async delivery
    #[serde(default = "default_telemetry_queue")]
    pub queue_size: usize,
    /// Deliver telemetry synchronously on the caller thread
    #[serde(default)]
    pub sync_mode: bool,
    /// Drain policy used when the process exits
    #[serde(default)]
    pub drain_mode: TelemetryDrainMode,
    /// Maximum time to wait for telemetry drain on shutdown
    #[serde(default = "default_telemetry_drain_timeout_ms")]
    pub drain_timeout_ms: u64,
    /// Require ingestion key (self-built binaries default to false)
    #[serde(default = "default_true")]
    pub require_ingestion_key: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum, Default)]
#[serde(rename_all = "kebab-case")]
pub enum TelemetryDrainMode {
    None,
    #[default]
    All,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            show_payloads: false,
            http_endpoint: None,
            udp_endpoint: Some("udp.telemetry.runmat.com:7846".to_string()),
            queue_size: default_telemetry_queue(),
            sync_mode: false,
            drain_mode: TelemetryDrainMode::All,
            drain_timeout_ms: default_telemetry_drain_timeout_ms(),
            require_ingestion_key: true,
        }
    }
}

pub(crate) fn default_telemetry_queue() -> usize {
    256
}

pub(crate) fn default_telemetry_drain_timeout_ms() -> u64 {
    50
}
