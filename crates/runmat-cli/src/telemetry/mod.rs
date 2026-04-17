mod client;
mod transport;

use once_cell::sync::OnceCell;
use runmat_config::TelemetryConfig as RuntimeTelemetryConfig;
use runmat_core::TelemetrySink;
use std::sync::Arc;

pub use runmat_telemetry::{ProviderSnapshot, RuntimeExecutionCounters, TelemetryRunKind};

use self::client::TelemetryClient;

static CLIENT: OnceCell<Arc<TelemetryClient>> = OnceCell::new();

/// Initialize the CLI telemetry transport sink (host-only).
pub fn init(config: &RuntimeTelemetryConfig) {
    if !config.enabled {
        return;
    }
    let options = client::TelemetryOptions::from(config);
    if !options.enabled {
        return;
    }
    let client = Arc::new(TelemetryClient::new(options));
    let _ = CLIENT.set(client);
}

/// Returns the configured telemetry sink for `runmat-core`, if enabled.
pub fn sink() -> Option<Arc<dyn TelemetrySink>> {
    CLIENT
        .get()
        .map(|client| client.clone() as Arc<dyn TelemetrySink>)
}

/// Capture the current acceleration provider snapshot, if one is registered.
pub fn capture_provider_snapshot() -> Option<ProviderSnapshot> {
    runmat_accelerate_api::provider().map(|provider| ProviderSnapshot {
        device: provider.device_info_struct(),
        telemetry: provider.telemetry_snapshot(),
    })
}

/// Surface the stable client id used for analytics so other components can reuse it.
pub fn telemetry_client_id() -> Option<String> {
    transport::stable_client_id()
}
