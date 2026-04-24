use crossbeam_channel::Receiver;
use std::fs;
use std::net::{SocketAddr, ToSocketAddrs, UdpSocket};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use uuid::Uuid;

const DEFAULT_HTTP_ENDPOINT: &str = "https://api.runmat.com/v1/t";

#[derive(Clone, Default)]
pub(crate) struct TelemetryTransport {
    udp_target: Option<SocketAddr>,
    http_endpoint: Option<String>,
    ingestion_key: Option<String>,
}

impl TelemetryTransport {
    pub(crate) fn from_options(options: &crate::telemetry::client::TelemetryOptions) -> Self {
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

    pub(crate) fn send(&self, payload: &str) {
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

pub(crate) fn spawn_worker(
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

pub(crate) fn stable_client_id() -> Option<String> {
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

pub(crate) fn resolve_ingestion_key() -> Option<String> {
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

pub(crate) struct TelemetryJob {
    pub(crate) payload: String,
}

pub(crate) const MIN_QUEUE_SIZE: usize = 8;
pub(crate) const DEFAULT_HTTP_ENDPOINT_VALUE: &str = DEFAULT_HTTP_ENDPOINT;
pub(crate) const MAX_DRAIN_TIMEOUT_MS: u64 = 5_000;
