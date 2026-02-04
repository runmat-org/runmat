use crate::RunMatSession;
use runmat_telemetry::{
    serialize_envelope, ProviderSnapshot, RuntimeExecutionCounters, RuntimeFinishedEnvelope,
    RuntimeFinishedPayload, RuntimeStartedEnvelope, RuntimeStartedPayload, TelemetryRunKind,
    EVENT_RUNTIME_FINISHED, EVENT_RUNTIME_STARTED,
};
use runmat_time::{unix_timestamp_ms, Instant};
use std::sync::Arc;
use std::time::Duration;
use uuid::Uuid;

/// Host-provided transport for runtime telemetry events.
///
/// The sink is intentionally synchronous and best-effort: core constructs the JSON payload and
/// hands it off to the host implementation (CLI HTTP/UDP, Desktop fetch proxy, etc.).
pub trait TelemetrySink: Send + Sync {
    fn emit(&self, payload_json: String);
}

#[derive(Debug, Clone, Default)]
pub struct TelemetryPlatformInfo {
    pub os: Option<String>,
    pub arch: Option<String>,
}

#[derive(Debug, Clone)]
pub struct TelemetryRunConfig {
    pub kind: TelemetryRunKind,
    pub jit_enabled: bool,
    pub accelerate_enabled: bool,
}

#[derive(Debug, Clone, Default)]
pub struct TelemetryRunFinish {
    /// If not set, we use wall-clock elapsed time from the run guard.
    pub duration: Option<Duration>,
    pub success: bool,
    pub jit_used: bool,
    /// A short, privacy-safe error class/identifier (no source snippets).
    pub error: Option<String>,
    pub counters: Option<RuntimeExecutionCounters>,
    pub provider: Option<ProviderSnapshot>,
}

pub struct TelemetryRunGuard {
    sink: Arc<dyn TelemetrySink>,
    cid: Option<String>,
    platform: TelemetryPlatformInfo,
    release: String,
    session_id: String,
    run_kind: String,
    started_at: Instant,
    started_payload: RuntimeStartedPayload,
}

impl TelemetryRunGuard {
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn finish(self, mut finish: TelemetryRunFinish) {
        let duration = finish.duration.take().or_else(|| Some(self.started_at.elapsed()));
        let duration_us = duration.map(|d| (d.as_micros().min(u64::MAX as u128)) as u64);

        let (gpu_wall_ns, gpu_dispatches, upload_bytes, download_bytes, fusion_hits, fusion_misses) =
            finish
                .provider
                .as_ref()
                .map_or((None, None, None, None, None, None), |snapshot| {
                    (
                        Some(snapshot.gpu_wall_ns()),
                        Some(snapshot.gpu_dispatches()),
                        Some(snapshot.telemetry.upload_bytes),
                        Some(snapshot.telemetry.download_bytes),
                        Some(snapshot.telemetry.fusion_cache_hits),
                        Some(snapshot.telemetry.fusion_cache_misses),
                    )
                });

        let gpu_ratio = match (gpu_wall_ns, duration_us) {
            (Some(wall_ns), Some(us)) if us > 0 => {
                Some(clamp_ratio(wall_ns as f64 / (us as f64 * 1000.0)))
            }
            _ => None,
        };
        let fusion_hit_ratio = match (fusion_hits, fusion_misses) {
            (Some(h), Some(m)) if h + m > 0 => Some(h as f64 / (h + m) as f64),
            _ => None,
        };

        let error = finish.error.map(|mut e| {
            if e.len() > 256 {
                e.truncate(256);
            }
            e
        });

        let envelope: RuntimeFinishedEnvelope = RuntimeFinishedEnvelope {
            event_label: EVENT_RUNTIME_FINISHED,
            cid: self.cid.clone(),
            session_id: self.session_id.clone(),
            os: self.platform.os.clone(),
            arch: self.platform.arch.clone(),
            release: Some(self.release.clone()),
            run_kind: self.run_kind.clone(),
            payload: RuntimeFinishedPayload {
                duration_us,
                success: finish.success,
                jit_enabled: self.started_payload.jit_enabled,
                jit_used: finish.jit_used,
                accelerate_enabled: self.started_payload.accelerate_enabled,
                timestamp_ms: Some(unix_timestamp_ms().min(u64::MAX as u128) as u64),
                error,
                counters: finish.counters,
                provider: finish.provider,
                gpu_wall_ns,
                gpu_ratio,
                gpu_dispatches,
                gpu_upload_bytes: upload_bytes,
                gpu_download_bytes: download_bytes,
                fusion_cache_hits: fusion_hits,
                fusion_cache_misses: fusion_misses,
                fusion_hit_ratio,
            },
        };

        if let Some(serialized) = serialize_envelope(&envelope) {
            self.sink.emit(serialized);
        }
    }
}

fn clamp_ratio(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

impl RunMatSession {
    pub fn set_telemetry_sink(&mut self, sink: Option<Arc<dyn TelemetrySink>>) {
        self.telemetry_sink = sink;
    }

    pub fn set_telemetry_platform_info(&mut self, platform: TelemetryPlatformInfo) {
        self.telemetry_platform = platform;
    }

    pub fn telemetry_platform_info(&self) -> &TelemetryPlatformInfo {
        &self.telemetry_platform
    }

    pub fn telemetry_run(&self, config: TelemetryRunConfig) -> Option<TelemetryRunGuard> {
        if !self.telemetry_consent {
            return None;
        }
        let sink = self.telemetry_sink.as_ref()?.clone();

        let platform = TelemetryPlatformInfo {
            os: self
                .telemetry_platform
                .os
                .clone()
                .or_else(|| Some(std::env::consts::OS.to_string())),
            arch: self
                .telemetry_platform
                .arch
                .clone()
                .or_else(|| Some(std::env::consts::ARCH.to_string())),
        };

        let session_id = Uuid::new_v4().to_string();
        let started_payload = RuntimeStartedPayload {
            jit_enabled: config.jit_enabled,
            accelerate_enabled: config.accelerate_enabled,
            timestamp_ms: Some(unix_timestamp_ms().min(u64::MAX as u128) as u64),
        };
        let envelope: RuntimeStartedEnvelope = RuntimeStartedEnvelope {
            event_label: EVENT_RUNTIME_STARTED,
            cid: self.telemetry_client_id.clone(),
            session_id: session_id.clone(),
            os: platform.os.clone(),
            arch: platform.arch.clone(),
            release: Some(env!("CARGO_PKG_VERSION").to_string()),
            run_kind: config.kind.as_str().to_string(),
            payload: started_payload.clone(),
        };

        if let Some(serialized) = serialize_envelope(&envelope) {
            sink.emit(serialized);
        }

        Some(TelemetryRunGuard {
            sink,
            cid: self.telemetry_client_id.clone(),
            platform,
            release: env!("CARGO_PKG_VERSION").to_string(),
            session_id,
            run_kind: config.kind.as_str().to_string(),
            started_at: Instant::now(),
            started_payload,
        })
    }
}

