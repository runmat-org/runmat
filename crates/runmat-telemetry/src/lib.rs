use serde::Serialize;

use runmat_accelerate_api::{ApiDeviceInfo, ProviderTelemetry};

pub const EVENT_RUNTIME_STARTED: &str = "runtime_started";
pub const EVENT_RUNTIME_FINISHED: &str = "runtime_finished";

#[derive(Debug, Clone, Copy)]
pub enum TelemetryEventKind {
    RuntimeStarted,
    RuntimeFinished,
}

impl TelemetryEventKind {
    pub fn label(&self) -> &'static str {
        match self {
            TelemetryEventKind::RuntimeStarted => EVENT_RUNTIME_STARTED,
            TelemetryEventKind::RuntimeFinished => EVENT_RUNTIME_FINISHED,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TelemetryRunKind {
    Script,
    Repl,
    Benchmark,
    Install,
}

impl TelemetryRunKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            TelemetryRunKind::Script => "script",
            TelemetryRunKind::Repl => "repl",
            TelemetryRunKind::Benchmark => "benchmark",
            TelemetryRunKind::Install => "install",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ProviderSnapshot {
    pub device: ApiDeviceInfo,
    pub telemetry: ProviderTelemetry,
}

impl ProviderSnapshot {
    pub fn gpu_wall_ns(&self) -> u64 {
        self.telemetry.fused_elementwise.total_wall_time_ns
            + self.telemetry.fused_reduction.total_wall_time_ns
            + self.telemetry.matmul.total_wall_time_ns
    }

    pub fn gpu_dispatches(&self) -> u64 {
        self.telemetry.fused_elementwise.count
            + self.telemetry.fused_reduction.count
            + self.telemetry.matmul.count
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeTelemetryEnvelope<P: Serialize> {
    #[serde(rename = "event_label")]
    pub event_label: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cid: Option<String>,
    pub session_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub os: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arch: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub release: Option<String>,
    pub run_kind: String,
    pub payload: P,
}

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeStartedPayload {
    pub jit_enabled: bool,
    pub accelerate_enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp_ms: Option<u64>,
}

pub type RuntimeStartedEnvelope = RuntimeTelemetryEnvelope<RuntimeStartedPayload>;

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeExecutionCounters {
    pub total_executions: u64,
    pub jit_compiled: u64,
    pub interpreter_fallback: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeFinishedPayload {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_us: Option<u64>,
    pub success: bool,
    pub jit_enabled: bool,
    pub jit_used: bool,
    pub accelerate_enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub counters: Option<RuntimeExecutionCounters>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderSnapshot>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_wall_ns: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_ratio: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_dispatches: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_upload_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_download_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fusion_cache_hits: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fusion_cache_misses: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fusion_hit_ratio: Option<f64>,
}

pub type RuntimeFinishedEnvelope = RuntimeTelemetryEnvelope<RuntimeFinishedPayload>;

pub fn serialize_envelope<P: Serialize>(envelope: &RuntimeTelemetryEnvelope<P>) -> Option<String> {
    serde_json::to_string(envelope).ok()
}
