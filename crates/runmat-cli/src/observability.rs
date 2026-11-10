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

impl OtelGuard {
    fn finish(mut self) -> Result<(), String> {
        if self.exporting {
            runmat_logging::shutdown_otel();
            self.exporting = false;
        }
        self._logging.flush_file_result()
    }
}

#[derive(Debug)]
struct LoggingCompletionError {
    primary: anyhow::Error,
    logging: String,
}
impl std::fmt::Display for LoggingCompletionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{}; additionally, {}",
            self.primary, self.logging
        )
    }
}
impl std::error::Error for LoggingCompletionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.primary.as_ref())
    }
}

/// Never turn a successful command into success after its configured sink failed,
/// and retain an existing user-code error as the primary diagnostic/cause.
pub fn complete_command(result: anyhow::Result<()>, guard: OtelGuard) -> anyhow::Result<()> {
    combine_command_result(result, guard.finish())
}

fn combine_command_result(
    result: anyhow::Result<()>,
    logging: Result<(), String>,
) -> anyhow::Result<()> {
    match (result, logging) {
        (result, Ok(())) => result,
        (Ok(()), Err(logging)) => Err(anyhow::anyhow!(logging)),
        (Err(primary), Err(logging)) => Err(LoggingCompletionError { primary, logging }.into()),
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

fn filter_from_config(config: &runmat_config::runtime::LoggingConfig) -> &'static str {
    use runmat_config::runtime::LogLevel;
    if config.debug {
        return "debug";
    }
    match config.level {
        LogLevel::Error => "error",
        LogLevel::Warn => "warn",
        LogLevel::Info => "info",
        LogLevel::Debug => "debug",
        LogLevel::Trace => "trace",
    }
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
    // Resolve/open before collector selection or any global logger/subscriber
    // installation. No directory creation, truncation or machine-stdout output.
    let file = config
        .file
        .as_ref()
        .map(|path| {
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .map_err(|error| format!("Cannot open configured CLI log file: {error}"))
        })
        .transpose()?;
    let options = options_from_config(&config.otel);
    let exporting = options.enabled;
    runmat_logging::configure_otel(options, StartupHeaders::default())?;
    let logging_options = runmat_logging::LoggingOptions {
        enable_otlp: exporting,
        enable_traces: false,
        pid: i64::from(std::process::id()),
        default_filter: Some(filter_from_config(config).into()),
    };
    let logging = match file {
        Some(file) => {
            runmat_logging::init_logging_with_host_logger_and_file(logging_options, file)?
        }
        None => runmat_logging::init_logging_with_host_logger(logging_options),
    };
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
/// Environment variable naming a file that a local hidden worker appends its log to when no
/// OpenTelemetry collector is configured. A worker's own stderr is a bounded diagnostic capture,
/// so solver progress (`runmat_fea::step` lines, for example) reaches a run log only this way.
/// The filter is `RUST_LOG`, else `RUNMAT_LOG`, else `info`.
pub const WORKER_LOG_FILE_ENV: &str = "RUNMAT_WORKER_LOG_FILE";

pub fn init_from_env() -> Option<OtelGuard> {
    let result = OtelOptions::default()
        .resolve_environment()
        .and_then(|options| init(options, Some("info".into())));
    match result {
        Ok(Some(guard)) => Some(guard),
        Ok(None) => init_worker_log_file(),
        Err(error) => {
            eprintln!("RunMat: OpenTelemetry startup failed: {error}");
            None
        }
    }
}

fn init_worker_log_file() -> Option<OtelGuard> {
    let path = std::env::var_os(WORKER_LOG_FILE_ENV)?;
    let file = match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        Ok(file) => file,
        Err(error) => {
            eprintln!("RunMat: cannot open the worker log file: {error}");
            return None;
        }
    };
    let options = runmat_logging::LoggingOptions {
        enable_otlp: false,
        enable_traces: false,
        pid: i64::from(std::process::id()),
        default_filter: Some("info".into()),
    };
    match runmat_logging::init_logging_with_host_logger_and_file(options, file) {
        Ok(logging) => Some(OtelGuard {
            _logging: logging,
            exporting: false,
        }),
        Err(error) => {
            eprintln!("RunMat: worker file logging failed: {error}");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn typed_filter_and_secondary_file_error_keep_original_cause() {
        use runmat_config::runtime::LogLevel;
        let mut config = runmat_config::runtime::LoggingConfig::default();
        for (level, expected) in [
            (LogLevel::Error, "error"),
            (LogLevel::Warn, "warn"),
            (LogLevel::Info, "info"),
            (LogLevel::Debug, "debug"),
            (LogLevel::Trace, "trace"),
        ] {
            config.level = level;
            assert_eq!(filter_from_config(&config), expected);
        }
        config.debug = true;
        assert_eq!(filter_from_config(&config), "debug");
        assert!(combine_command_result(Ok(()), Err("file failed".into())).is_err());
        let combined = combine_command_result(
            Err(anyhow::anyhow!("original user failure")),
            Err("file failed".into()),
        )
        .unwrap_err();
        assert!(combined.to_string().starts_with("original user failure"));
        assert!(combined.to_string().contains("file failed"));
        assert_eq!(combined.root_cause().to_string(), "original user failure");
        assert!(combine_command_result(Ok(()), Ok(())).is_ok());
    }
}
