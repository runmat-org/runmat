//! Optional OTLP tracing for CLI and hidden native-host processes.

/// Keeps the exporter alive for the surrounding command and flushes it on process exit.
pub struct OtelGuard;

impl Drop for OtelGuard {
    fn drop(&mut self) {
        runmat_logging::shutdown_otel();
    }
}

/// Enables OTLP tracing when `RUNMAT_OTEL_ENDPOINT` names a non-empty OTLP HTTP base endpoint.
///
/// The exporter appends `/v1/traces`. The explicit opt-in prevents ordinary CLI invocations from
/// attempting a network connection. Native workers inherit the variable, so their stage and
/// progress spans use the same collector without writing diagnostics into the framed stdout
/// protocol.
pub fn init_from_env() -> Option<OtelGuard> {
    let endpoint = std::env::var("RUNMAT_OTEL_ENDPOINT").ok()?;
    if endpoint.trim().is_empty() {
        return None;
    }
    let _ = runmat_logging::init_logging(runmat_logging::LoggingOptions {
        enable_otlp: true,
        enable_traces: false,
        pid: i64::from(std::process::id()),
        default_filter: Some("info".into()),
    });
    Some(OtelGuard)
}
