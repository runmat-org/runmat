//! User collector export, independent of RunMat product telemetry.
use runmat_logging::otel::{OtelOptions, OtelProtocol, OtelSignal, StartupHeaders};

/// Keeps thread-local subscribers alive and flushes both signals on exit.
pub struct OtelGuard {
    _logging: runmat_logging::LoggingGuard,
    exporting: bool,
}

impl Drop for OtelGuard {
    fn drop(&mut self) {
        if self.exporting {
            runmat_logging::shutdown_otel();
        }
    }
}

/// Explicit conversion keeps settings serialization independent of exporter credentials.
pub fn options_from_config(config: &runmat_config::runtime::OtelConfig) -> OtelOptions {
    use runmat_config::runtime::{OtelProtocol as ConfigProtocol, OtelSignal as ConfigSignal};
    OtelOptions {
        enabled: config.enabled,
        endpoint: config.endpoint.clone(),
        protocol: match config.protocol {
            ConfigProtocol::HttpProtobuf => OtelProtocol::HttpProtobuf,
            ConfigProtocol::Grpc => OtelProtocol::Grpc,
        },
        signals: config
            .signals
            .iter()
            .map(|signal| match signal {
                ConfigSignal::Traces => OtelSignal::Traces,
                ConfigSignal::Logs => OtelSignal::Logs,
                ConfigSignal::Metrics => OtelSignal::Metrics,
            })
            .collect(),
        headers_from_env: config.headers_from_env.clone(),
        service_name: config.service_name.clone(),
        resource_attributes: config.resource_attributes.clone(),
        ..OtelOptions::default()
    }
}

/// Regular CLI commands use their already-layered effective configuration.
pub fn init_from_config(
    config: &runmat_config::runtime::LoggingConfig,
) -> Result<Option<OtelGuard>, String> {
    init(
        options_from_config(&config.otel),
        Some(format!("{:?}", config.level).to_ascii_lowercase()),
    )
}

/// One log owner preserves env_logger's terminal formatting and independently forwards
/// original records into the tracing subscriber. Machine-readable stdout is untouched.
struct CliLogger {
    terminal: env_logger::Logger,
    bridge_warning: std::sync::atomic::AtomicBool,
}

impl log::Log for CliLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        log::Log::enabled(&self.terminal, metadata)
            || runmat_logging::log_metadata_enabled(metadata)
    }
    fn log(&self, record: &log::Record<'_>) {
        log::Log::log(&self.terminal, record);
        if runmat_logging::forward_log_record(record).is_err()
            && !self
                .bridge_warning
                .swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            eprintln!("RunMat: the host log bridge could not forward a record");
        }
    }
    fn flush(&self) {
        log::Log::flush(&self.terminal);
    }
}

static CLI_LOGGER: std::sync::OnceLock<CliLogger> = std::sync::OnceLock::new();

/// Initialize the actual single global log owner and subscriber, even when OTLP is off.
pub fn init_with_terminal(
    config: &runmat_config::runtime::LoggingConfig,
    terminal: env_logger::Logger,
) -> Result<OtelGuard, String> {
    if CLI_LOGGER.get().is_some() {
        return Err("CLI logging is already initialized".into());
    }
    let options = options_from_config(&config.otel);
    let exporting = options.enabled;
    runmat_logging::configure_otel(options, StartupHeaders::default())?;
    let logging = runmat_logging::init_logging_with_host_logger(runmat_logging::LoggingOptions {
        enable_otlp: exporting,
        enable_traces: false,
        pid: i64::from(std::process::id()),
        default_filter: Some(if config.debug {
            "debug".into()
        } else {
            format!("{:?}", config.level).to_ascii_lowercase()
        }),
    });
    CLI_LOGGER
        .set(CliLogger {
            terminal,
            bridge_warning: std::sync::atomic::AtomicBool::new(false),
        })
        .map_err(|_| "CLI logging is already initialized")?;
    log::set_logger(
        CLI_LOGGER
            .get()
            .ok_or("CLI logging initialization failed")?,
    )
    .map_err(|_| "another host logger is already installed; CLI logging was not activated")?;
    // The terminal and subscriber apply their own independent filters.
    log::set_max_level(log::LevelFilter::Trace);
    Ok(OtelGuard {
        _logging: logging,
        exporting,
    })
}

fn init(options: OtelOptions, filter: Option<String>) -> Result<Option<OtelGuard>, String> {
    if !options.enabled {
        return Ok(None);
    }
    runmat_logging::configure_otel(options, StartupHeaders::default())?;
    let logging = runmat_logging::init_logging(runmat_logging::LoggingOptions {
        enable_otlp: true,
        enable_traces: false,
        pid: i64::from(std::process::id()),
        default_filter: filter,
    });
    Ok(Some(OtelGuard {
        _logging: logging,
        exporting: true,
    }))
}

/// Hidden local workers inherit startup environment; remote workers should receive only their
/// administrator's collector policy, never submitter keychain secrets or personal user settings.
pub fn init_from_env() -> Option<OtelGuard> {
    let result = OtelOptions::default()
        .resolve_environment()
        .and_then(|options| init(options, Some("info".into())));
    match result {
        Ok(guard) => guard,
        Err(error) => {
            eprintln!("RunMat: OpenTelemetry startup failed: {error}");
            None
        }
    }
}
