use crossbeam_channel::{bounded, Sender};
use runmat_config::TelemetryConfig as RuntimeTelemetryConfig;
use runmat_config::TelemetryDrainMode as RuntimeTelemetryDrainMode;
use runmat_core::TelemetrySink;
use runmat_time::Instant;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::telemetry::transport::{
    resolve_ingestion_key, spawn_worker, TelemetryJob, TelemetryTransport,
    DEFAULT_HTTP_ENDPOINT_VALUE, MAX_DRAIN_TIMEOUT_MS, MIN_QUEUE_SIZE,
};

pub(crate) struct TelemetryClient {
    enabled: bool,
    show_payloads: bool,
    sender: Option<Sender<TelemetryJob>>,
    transport: TelemetryTransport,
    sync_mode: bool,
    pending_total: Arc<AtomicUsize>,
    drain_mode: RuntimeTelemetryDrainMode,
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

pub(crate) struct TelemetryOptions {
    pub(crate) enabled: bool,
    show_payloads: bool,
    queue_size: usize,
    pub(crate) http_endpoint: Option<String>,
    pub(crate) udp_endpoint: Option<String>,
    sync_mode: bool,
    pub(crate) ingestion_key: Option<String>,
    require_ingestion_key: bool,
    drain_mode: RuntimeTelemetryDrainMode,
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
                .or_else(|| Some(DEFAULT_HTTP_ENDPOINT_VALUE.to_string())),
            udp_endpoint: cfg
                .udp_endpoint
                .as_ref()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty()),
            sync_mode: cfg.sync_mode,
            ingestion_key: resolve_ingestion_key(),
            require_ingestion_key: cfg.require_ingestion_key,
            drain_mode: cfg.drain_mode,
            drain_timeout: Duration::from_millis(cfg.drain_timeout_ms.min(MAX_DRAIN_TIMEOUT_MS)),
        }
    }
}

impl TelemetryClient {
    pub(crate) fn new(options: TelemetryOptions) -> Self {
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

impl Drop for TelemetryClient {
    fn drop(&mut self) {
        self.sender.take();
        match self.drain_mode {
            RuntimeTelemetryDrainMode::None => {}
            RuntimeTelemetryDrainMode::All => self.wait_for(&self.pending_total),
        }
    }
}
