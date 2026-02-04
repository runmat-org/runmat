use crate::config::TelemetryConfig as RuntimeTelemetryConfig;
use crossbeam_channel::{bounded, Receiver, Sender};
use once_cell::sync::OnceCell;
use runmat_accelerate_api::provider;
use runmat_core::TelemetrySink;
use runmat_time::Instant;
use std::fs;
use std::net::{SocketAddr, ToSocketAddrs, UdpSocket};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use uuid::Uuid;

const DEFAULT_HTTP_ENDPOINT: &str = "https://telemetry.runmat.org/ingest";
const MIN_QUEUE_SIZE: usize = 8;
const DEFAULT_DRAIN_TIMEOUT_MS: u64 = 50;

static CLIENT: OnceCell<Arc<TelemetryClient>> = OnceCell::new();

pub use runmat_telemetry::{ProviderSnapshot, RuntimeExecutionCounters, TelemetryRunKind};

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

/// Initialize the CLI telemetry transport sink (host-only).
pub fn init(config: &RuntimeTelemetryConfig) {
    if !config.enabled {
        return;
    }
    let options = TelemetryOptions::from(config);
    if !options.enabled {
        return;
    }
    let client = Arc::new(TelemetryClient::new(options));
    let _ = CLIENT.set(client);
}

/// Returns the configured telemetry sink for `runmat-core`, if enabled.
pub fn sink() -> Option<Arc<dyn TelemetrySink>> {
    CLIENT.get().map(|client| client.clone() as Arc<dyn TelemetrySink>)
}

/// Capture the current acceleration provider snapshot, if one is registered.
pub fn capture_provider_snapshot() -> Option<ProviderSnapshot> {
    provider().map(|p| ProviderSnapshot {
        device: p.device_info_struct(),
        telemetry: p.telemetry_snapshot(),
    })
}

/// Surface the stable client id used for analytics so other components can reuse it.
pub fn telemetry_client_id() -> Option<String> {
    stable_client_id()
}

struct TelemetryClient {
    enabled: bool,
    show_payloads: bool,
    sender: Option<Sender<TelemetryJob>>,
    transport: TelemetryTransport,
    sync_mode: bool,
    pending_total: Arc<AtomicUsize>,
    drain_mode: TelemetryDrainMode,
    drain_timeout: Duration,
}

impl TelemetrySink for TelemetryClient {
    fn emit(&self, payload_json: String) {
        if !self.enabled {
            return;
        }
        self.enqueue_serialized(payload_json);
    }
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
    drain_mode: TelemetryDrainMode,
    drain_timeout: Duration,
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
            drain_mode: resolve_drain_mode(),
            drain_timeout: resolve_drain_timeout(),
        }
    }
}

impl TelemetryClient {
    fn new(options: TelemetryOptions) -> Self {
        let transport = TelemetryTransport::from_options(&options);
        let pending_total = Arc::new(AtomicUsize::new(0));

        if !options.enabled || (options.require_ingestion_key && options.ingestion_key.is_none()) {
            return Self {
                enabled: false,
                show_payloads: options.show_payloads,
                sender: None,
                transport,
                sync_mode: options.sync_mode,
                pending_total,
                drain_mode: options.drain_mode,
                drain_timeout: options.drain_timeout,
            };
        }

        let sender = if options.sync_mode {
            None
        } else {
            let (tx, rx) = bounded(options.queue_size);
            spawn_worker(rx, transport.clone(), pending_total.clone());
            Some(tx)
        };

        Self {
            enabled: true,
            show_payloads: options.show_payloads,
            sender,
            transport,
            sync_mode: options.sync_mode,
            pending_total,
            drain_mode: options.drain_mode,
            drain_timeout: options.drain_timeout,
        }
    }

    fn enqueue_serialized(&self, serialized: String) {
        if !self.enabled {
            return;
        }
        if self.show_payloads {
            println!("[runmat telemetry] {serialized}");
        }
        if self.sync_mode {
            self.transport.send(&serialized);
            return;
        }
        if let Some(sender) = &self.sender {
            self.increment_counters();
            if sender
                .try_send(TelemetryJob {
                    payload: serialized,
                })
                .is_err()
            {
                self.decrement_counters();
            }
        }
    }

    fn increment_counters(&self) {
        self.pending_total.fetch_add(1, Ordering::SeqCst);
    }

    fn decrement_counters(&self) {
        self.pending_total.fetch_sub(1, Ordering::SeqCst);
    }
}

impl Drop for TelemetryClient {
    fn drop(&mut self) {
        self.sender.take();
        match self.drain_mode {
            TelemetryDrainMode::None => {}
            TelemetryDrainMode::All => self.wait_for(&self.pending_total),
        }
    }
}

impl TelemetryClient {
    fn wait_for(&self, counter: &AtomicUsize) {
        let start = Instant::now();
        while counter.load(Ordering::SeqCst) > 0 {
            if start.elapsed() >= self.drain_timeout {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
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

fn spawn_worker(
    rx: Receiver<TelemetryJob>,
    transport: TelemetryTransport,
    pending_total: Arc<AtomicUsize>,
) {
    thread::Builder::new()
        .name("runmat-telemetry".to_string())
        .spawn(move || {
            while let Ok(job) = rx.recv() {
                transport.send(&job.payload);
                pending_total.fetch_sub(1, Ordering::SeqCst);
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

fn resolve_drain_mode() -> TelemetryDrainMode {
    match std::env::var("RUNMAT_TELEMETRY_DRAIN") {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "started" | "start" | "session" => TelemetryDrainMode::All,
            "all" | "full" | "both" | "runtime" => TelemetryDrainMode::All,
            "none" | "off" | "" => TelemetryDrainMode::None,
            _ => TelemetryDrainMode::All,
        },
        Err(_) => TelemetryDrainMode::All,
    }
}

fn resolve_drain_timeout() -> Duration {
    const MAX_TIMEOUT_MS: u64 = 5_000;
    match std::env::var("RUNMAT_TELEMETRY_DRAIN_TIMEOUT_MS") {
        Ok(value) => value
            .trim()
            .parse::<u64>()
            .map(|ms| Duration::from_millis(ms.min(MAX_TIMEOUT_MS)))
            .unwrap_or_else(|_| Duration::from_millis(DEFAULT_DRAIN_TIMEOUT_MS)),
        Err(_) => Duration::from_millis(DEFAULT_DRAIN_TIMEOUT_MS),
    }
}

struct TelemetryJob {
    payload: String,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TelemetryDrainMode {
    None,
    All,
}

