use crate::config::TelemetryConfig as RuntimeTelemetryConfig;
use crossbeam_channel::{bounded, Receiver, Sender};
use once_cell::sync::OnceCell;
use runmat_accelerate_api::{provider, ApiDeviceInfo, ProviderTelemetry};
use serde::Serialize;
use std::fs;
use std::net::{SocketAddr, ToSocketAddrs, UdpSocket};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

const DEFAULT_HTTP_ENDPOINT: &str = "https://telemetry.runmat.org/ingest";
const MIN_QUEUE_SIZE: usize = 8;

static CLIENT: OnceCell<TelemetryClient> = OnceCell::new();

fn parse_env_bool(key: &str) -> Option<bool> {
    match std::env::var(key) {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        },
        Err(_) => None,
    }
}
/// Initialize the telemetry client using the runtime configuration.
pub fn init(config: &RuntimeTelemetryConfig) {
    if !config.enabled {
        return;
    }
    let options = TelemetryOptions::from(config);
    if !options.enabled {
        return;
    }
    let client = TelemetryClient::new(options);
    let _ = CLIENT.set(client);
}

/// Emit a session start event (one per CLI mode).
pub fn emit_session_start(event: TelemetrySessionEvent) {
    if let Some(client) = CLIENT.get() {
        client.emit_session_start(event);
    }
}

/// Emit a runtime value-delivery event when a script/REPL/benchmark completes.
pub fn emit_runtime_value(record: RuntimeTelemetryRecord) {
    if let Some(client) = CLIENT.get() {
        client.emit_runtime_value(record);
    }
}

/// Capture the current acceleration provider snapshot, if one is registered.
pub fn capture_provider_snapshot() -> Option<ProviderSnapshot> {
    provider().map(|p| ProviderSnapshot {
        device: p.device_info_struct(),
        telemetry: p.telemetry_snapshot(),
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeExecutionCounters {
    pub total_executions: u64,
    pub jit_compiled: u64,
    pub interpreter_fallback: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProviderSnapshot {
    pub device: ApiDeviceInfo,
    pub telemetry: ProviderTelemetry,
}

impl ProviderSnapshot {
    fn gpu_wall_ns(&self) -> u64 {
        self.telemetry.fused_elementwise.total_wall_time_ns
            + self.telemetry.fused_reduction.total_wall_time_ns
            + self.telemetry.matmul.total_wall_time_ns
    }

    fn gpu_dispatches(&self) -> u64 {
        self.telemetry.fused_elementwise.count
            + self.telemetry.fused_reduction.count
            + self.telemetry.matmul.count
    }
}

#[derive(Debug, Clone, Copy)]
pub enum TelemetryRunKind {
    Script,
    Repl,
    Benchmark,
}

impl TelemetryRunKind {
    fn as_str(&self) -> &'static str {
        match self {
            TelemetryRunKind::Script => "script",
            TelemetryRunKind::Repl => "repl",
            TelemetryRunKind::Benchmark => "benchmark",
        }
    }
}

pub struct TelemetrySessionEvent {
    pub kind: TelemetryRunKind,
    pub jit_enabled: bool,
    pub accelerate_enabled: bool,
}

pub struct RuntimeTelemetryRecord {
    pub kind: TelemetryRunKind,
    pub duration: Option<Duration>,
    pub success: bool,
    pub error: Option<String>,
    pub jit_enabled: bool,
    pub jit_used: bool,
    pub accelerate_enabled: bool,
    pub counters: Option<RuntimeExecutionCounters>,
    pub provider: Option<ProviderSnapshot>,
}

struct TelemetryClient {
    enabled: bool,
    show_payloads: bool,
    sender: Option<Sender<String>>,
    transport: TelemetryTransport,
    sync_mode: bool,
    context: TelemetryContext,
}

#[derive(Clone)]
struct TelemetryContext {
    cid: Option<String>,
    session_id: String,
}

struct TelemetryOptions {
    enabled: bool,
    show_payloads: bool,
    queue_size: usize,
    http_endpoint: Option<String>,
    udp_endpoint: Option<String>,
    sync_mode: bool,
    ingestion_key: Option<String>,
    require_ingestion_key: bool,
}

impl From<&RuntimeTelemetryConfig> for TelemetryOptions {
    fn from(cfg: &RuntimeTelemetryConfig) -> Self {
        Self {
            enabled: cfg.enabled,
            show_payloads: cfg.show_payloads,
            queue_size: cfg.queue_size.max(MIN_QUEUE_SIZE),
            http_endpoint: cfg
                .http_endpoint
                .as_ref()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .or_else(|| Some(DEFAULT_HTTP_ENDPOINT.to_string())),
            udp_endpoint: cfg
                .udp_endpoint
                .as_ref()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty()),
            sync_mode: parse_env_bool("RUNMAT_TELEMETRY_SYNC").unwrap_or(false),
            ingestion_key: resolve_ingestion_key(),
            require_ingestion_key: cfg.require_ingestion_key,
        }
    }
}

impl TelemetryClient {
    fn new(options: TelemetryOptions) -> Self {
        let transport = TelemetryTransport::from_options(&options);
        let context = TelemetryContext {
            cid: stable_client_id(),
            session_id: Uuid::new_v4().to_string(),
        };

        if !options.enabled || (options.require_ingestion_key && options.ingestion_key.is_none()) {
            return Self {
                enabled: false,
                show_payloads: options.show_payloads,
                sender: None,
                transport,
                sync_mode: options.sync_mode,
                context,
            };
        }

        let sender = if options.sync_mode {
            None
        } else {
            let (tx, rx) = bounded(options.queue_size);
            spawn_worker(rx, transport.clone());
            Some(tx)
        };

        Self {
            enabled: true,
            show_payloads: options.show_payloads,
            sender,
            transport,
            sync_mode: options.sync_mode,
            context,
        }
    }

    fn emit_session_start(&self, event: TelemetrySessionEvent) {
        if !self.enabled {
            return;
        }
        let payload = SessionStartEnvelope::new(&self.context, event);
        self.enqueue(payload);
    }

    fn emit_runtime_value(&self, record: RuntimeTelemetryRecord) {
        if !self.enabled {
            return;
        }
        let payload = RuntimeValueEnvelope::new(&self.context, record);
        self.enqueue(payload);
    }

    fn enqueue<T: Serialize>(&self, value: T) {
        if !self.enabled {
            return;
        }
        match serde_json::to_string(&value) {
            Ok(serialized) => {
                if self.show_payloads {
                    println!("[runmat telemetry] {serialized}");
                }
                if self.sync_mode {
                    self.transport.send(&serialized);
                } else if let Some(sender) = &self.sender {
                    let _ = sender.try_send(serialized);
                }
            }
            Err(err) => {
                log::debug!("failed to serialize telemetry payload: {err}");
            }
        }
    }
}

#[derive(Clone, Default)]
struct TelemetryTransport {
    udp_target: Option<SocketAddr>,
    http_endpoint: Option<String>,
    ingestion_key: Option<String>,
}

impl TelemetryTransport {
    fn from_options(options: &TelemetryOptions) -> Self {
        let udp_target = options
            .udp_endpoint
            .as_ref()
            .and_then(|target| resolve_udp_target(target));
        Self {
            udp_target,
            http_endpoint: options.http_endpoint.clone(),
            ingestion_key: options.ingestion_key.clone(),
        }
    }

    fn send(&self, payload: &str) {
        if let Some(target) = self.udp_target {
            if let Ok(socket) = UdpSocket::bind("0.0.0.0:0") {
                let _ = socket.send_to(payload.as_bytes(), target);
            }
        }

        if let Some(endpoint) = &self.http_endpoint {
            let mut request = ureq::post(endpoint)
                .set("Content-Type", "application/json")
                .timeout(Duration::from_secs(2));
            if let Some(key) = &self.ingestion_key {
                request = request.set("x-telemetry-key", key);
            }
            if let Err(err) = request.send_string(payload) {
                log::debug!("telemetry http error: {err}");
            }
        }
    }
}

fn spawn_worker(rx: Receiver<String>, transport: TelemetryTransport) {
    thread::Builder::new()
        .name("runmat-telemetry".to_string())
        .spawn(move || {
            while let Ok(payload) = rx.recv() {
                transport.send(&payload);
            }
        })
        .expect("failed to spawn telemetry worker");
}

fn resolve_udp_target(endpoint: &str) -> Option<SocketAddr> {
    endpoint
        .to_socket_addrs()
        .ok()
        .and_then(|mut iter| iter.next())
}

fn stable_client_id() -> Option<String> {
    let home = dirs::home_dir()?;
    let path = home.join(".runmat").join("telemetry_id");

    if let Ok(existing) = fs::read_to_string(&path) {
        let trimmed = existing.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }

    let new_id = Uuid::new_v4().to_string();
    if let Some(parent) = path.parent() {
        if let Err(err) = fs::create_dir_all(parent) {
            log::debug!("failed to create telemetry dir: {err}");
        }
    }
    if let Err(err) = fs::write(&path, &new_id) {
        log::debug!("failed to persist telemetry id: {err}");
    }
    Some(new_id)
}

fn resolve_ingestion_key() -> Option<String> {
    if let Ok(value) = std::env::var("RUNMAT_TELEMETRY_KEY") {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    option_env!("RUNMAT_TELEMETRY_KEY")
        .map(str::trim)
        .and_then(|value| {
            if value.is_empty() {
                None
            } else {
                Some(value.to_string())
            }
        })
}

fn now_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_millis() as u64
}

fn duration_to_micros(duration: Duration) -> u64 {
    duration.as_micros().min(u64::MAX as u128) as u64
}

fn clamp_ratio(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[derive(Serialize)]
struct SessionStartEnvelope<'a> {
    #[serde(rename = "event_label")]
    event_label: &'static str,
    cid: Option<&'a str>,
    session_id: &'a str,
    os: &'static str,
    arch: &'static str,
    release: &'static str,
    run_kind: &'a str,
    payload: SessionStartPayload,
}

#[derive(Serialize)]
struct SessionStartPayload {
    jit_enabled: bool,
    accelerate_enabled: bool,
    timestamp_ms: u64,
}

impl<'a> SessionStartEnvelope<'a> {
    fn new(context: &'a TelemetryContext, event: TelemetrySessionEvent) -> Self {
        Self {
            event_label: "runtime_started",
            cid: context.cid.as_deref(),
            session_id: &context.session_id,
            os: std::env::consts::OS,
            arch: std::env::consts::ARCH,
            release: env!("CARGO_PKG_VERSION"),
            run_kind: event.kind.as_str(),
            payload: SessionStartPayload {
                jit_enabled: event.jit_enabled,
                accelerate_enabled: event.accelerate_enabled,
                timestamp_ms: now_timestamp_ms(),
            },
        }
    }
}

#[derive(Serialize)]
struct RuntimeValueEnvelope<'a> {
    #[serde(rename = "event_label")]
    event_label: &'static str,
    cid: Option<&'a str>,
    session_id: &'a str,
    os: &'static str,
    arch: &'static str,
    release: &'static str,
    run_kind: &'a str,
    payload: RuntimeValuePayload,
}

#[derive(Serialize)]
struct RuntimeValuePayload {
    duration_us: Option<u64>,
    success: bool,
    jit_enabled: bool,
    jit_used: bool,
    accelerate_enabled: bool,
    timestamp_ms: u64,
    error: Option<String>,
    counters: Option<RuntimeExecutionCounters>,
    provider: Option<ProviderSnapshot>,
    gpu_wall_ns: Option<u64>,
    gpu_ratio: Option<f64>,
    gpu_dispatches: Option<u64>,
    gpu_upload_bytes: Option<u64>,
    gpu_download_bytes: Option<u64>,
    fusion_cache_hits: Option<u64>,
    fusion_cache_misses: Option<u64>,
    fusion_hit_ratio: Option<f64>,
}

impl<'a> RuntimeValueEnvelope<'a> {
    fn new(context: &'a TelemetryContext, record: RuntimeTelemetryRecord) -> Self {
        let duration_us = record.duration.map(duration_to_micros);
        let (gpu_wall_ns, gpu_dispatches, upload_bytes, download_bytes, fusion_hits, fusion_misses) =
            record
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

        let error = record.error.map(|mut e| {
            if e.len() > 256 {
                e.truncate(256);
            }
            e
        });

        Self {
            event_label: "runtime_finished",
            cid: context.cid.as_deref(),
            session_id: &context.session_id,
            os: std::env::consts::OS,
            arch: std::env::consts::ARCH,
            release: env!("CARGO_PKG_VERSION"),
            run_kind: record.kind.as_str(),
            payload: RuntimeValuePayload {
                duration_us,
                success: record.success,
                jit_enabled: record.jit_enabled,
                jit_used: record.jit_used,
                accelerate_enabled: record.accelerate_enabled,
                timestamp_ms: now_timestamp_ms(),
                error,
                counters: record.counters,
                provider: record.provider,
                gpu_wall_ns,
                gpu_ratio,
                gpu_dispatches,
                gpu_upload_bytes: upload_bytes,
                gpu_download_bytes: download_bytes,
                fusion_cache_hits: fusion_hits,
                fusion_cache_misses: fusion_misses,
                fusion_hit_ratio: match (fusion_hits, fusion_misses) {
                    (Some(h), Some(m)) if h + m > 0 => Some(h as f64 / (h + m) as f64),
                    _ => None,
                },
            },
        }
    }
}
