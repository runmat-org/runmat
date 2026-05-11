use anyhow::Result;
use std::env;
use std::path::PathBuf;

use crate::{
    AccelerateProviderPreference, GcPreset, JitOptLevel, LogLevel, PlotBackend, PlotMode,
    RunMatConfig,
};

use super::parse::{
    parse_auto_offload_log_level, parse_bool, parse_power_preference, parse_provider_preference,
    parse_telemetry_drain_mode,
};

const MIN_QUEUE_SIZE: usize = 8;

/// Apply environment variable overrides.
pub(crate) fn apply_environment_variables(config: &mut RunMatConfig) -> Result<()> {
    // Runtime settings
    if let Some(timeout) = env_value("RUNMAT_TIMEOUT", &[]) {
        if let Ok(timeout) = timeout.parse() {
            config.runtime.timeout = timeout;
        }
    }

    if let Some(limit) = env_value("RUNMAT_CALLSTACK_LIMIT", &[]) {
        if let Ok(limit) = limit.parse() {
            config.runtime.callstack_limit = limit;
        }
    }

    if let Some(namespace) = env_value("RUNMAT_ERROR_NAMESPACE", &[]) {
        let trimmed = namespace.trim();
        if !trimmed.is_empty() {
            config.runtime.error_namespace = trimmed.to_string();
        }
    }

    if let Some(verbose) = env_bool("RUNMAT_VERBOSE", &[]) {
        config.runtime.verbose = verbose;
    }

    if let Some(snapshot) = env_value("RUNMAT_SNAPSHOT_PATH", &[]) {
        config.runtime.snapshot_path = Some(PathBuf::from(snapshot));
    }

    // Telemetry settings
    if let Some(flag) = env_bool("RUNMAT_TELEMETRY", &[]) {
        config.telemetry.enabled = flag;
    }
    if let Some(flag) = env_bool("RUNMAT_NO_TELEMETRY", &[]) {
        if flag {
            config.telemetry.enabled = false;
        }
    }
    if let Some(show) = env_bool("RUNMAT_TELEMETRY_SHOW", &[]) {
        config.telemetry.show_payloads = show;
    }
    if let Some(endpoint) = env_value(
        "RUNMAT_TELEMETRY_ENDPOINT",
        &["RUNMAT_TELEMETRY_HTTP_ENDPOINT"],
    ) {
        let trimmed = endpoint.trim();
        if trimmed.is_empty() {
            config.telemetry.http_endpoint = None;
        } else {
            config.telemetry.http_endpoint = Some(trimmed.to_string());
        }
    }
    if let Some(udp) = env_value("RUNMAT_TELEMETRY_UDP_ENDPOINT", &[]) {
        let trimmed = udp.trim();
        if trimmed.is_empty() || trimmed == "0" || trimmed.eq_ignore_ascii_case("off") {
            config.telemetry.udp_endpoint = None;
        } else {
            config.telemetry.udp_endpoint = Some(trimmed.to_string());
        }
    }
    if let Some(queue) = env_value("RUNMAT_TELEMETRY_QUEUE_SIZE", &[]) {
        if let Ok(parsed) = queue.parse::<usize>() {
            config.telemetry.queue_size = parsed.max(MIN_QUEUE_SIZE);
        }
    }
    if let Some(sync_mode) = env_bool("RUNMAT_TELEMETRY_SYNC", &[]) {
        config.telemetry.sync_mode = sync_mode;
    }
    if let Some(drain_mode) = env_value("RUNMAT_TELEMETRY_DRAIN", &[]) {
        if let Some(parsed) = parse_telemetry_drain_mode(&drain_mode) {
            config.telemetry.drain_mode = parsed;
        }
    }
    if let Some(drain_timeout) = env_value("RUNMAT_TELEMETRY_DRAIN_TIMEOUT_MS", &[]) {
        if let Ok(parsed) = drain_timeout.trim().parse::<u64>() {
            config.telemetry.drain_timeout_ms = parsed;
        }
    }

    // Acceleration settings
    if let Some(accel) = env_value("RUNMAT_ACCEL_ENABLE", &[]) {
        if let Some(flag) = parse_bool(&accel) {
            config.accelerate.enabled = flag;
        }
    }

    if let Some(provider) = env_value("RUNMAT_ACCEL_PROVIDER", &[]) {
        if let Some(pref) = parse_provider_preference(&provider) {
            config.accelerate.provider = pref;
        }
    }

    if let Some(force_inprocess) = env_bool("RUNMAT_ACCEL_FORCE_INPROCESS", &[]) {
        if force_inprocess {
            config.accelerate.provider = AccelerateProviderPreference::InProcess;
        }
    }

    if let Some(wgpu_toggle) = env_bool("RUNMAT_ACCEL_WGPU", &[]) {
        config.accelerate.provider = if wgpu_toggle {
            AccelerateProviderPreference::Wgpu
        } else {
            AccelerateProviderPreference::InProcess
        };
    }

    if let Some(fallback) = env_bool("RUNMAT_ACCEL_DISABLE_FALLBACK", &[]) {
        config.accelerate.allow_inprocess_fallback = !fallback;
    }

    if let Some(force_fallback) = env_bool("RUNMAT_ACCEL_WGPU_FORCE_FALLBACK", &[]) {
        config.accelerate.wgpu_force_fallback_adapter = force_fallback;
    }

    if let Some(power) = env_value("RUNMAT_ACCEL_WGPU_POWER", &[]) {
        if let Some(pref) = parse_power_preference(&power) {
            config.accelerate.wgpu_power_preference = pref;
        }
    }

    if let Some(auto_enabled) = env_bool("RUNMAT_ACCEL_AUTO_OFFLOAD", &[]) {
        config.accelerate.auto_offload.enabled = auto_enabled;
    }

    if let Some(auto_calibrate) = env_bool("RUNMAT_ACCEL_CALIBRATE", &[]) {
        config.accelerate.auto_offload.calibrate = auto_calibrate;
    }

    if let Some(profile_path) = env_value("RUNMAT_ACCEL_PROFILE", &[]) {
        config.accelerate.auto_offload.profile_path = Some(PathBuf::from(profile_path));
    }

    if let Some(auto_log) = env_value("RUNMAT_ACCEL_AUTO_LOG", &[]) {
        if let Some(level) = parse_auto_offload_log_level(&auto_log) {
            config.accelerate.auto_offload.log_level = level;
        }
    }

    // JIT settings
    if let Some(jit_enabled) = env_bool("RUNMAT_JIT_ENABLE", &[]) {
        config.jit.enabled = jit_enabled;
    }

    if let Some(jit_disabled) = env_bool("RUNMAT_JIT_DISABLE", &[]) {
        if jit_disabled {
            config.jit.enabled = false;
        }
    }

    if let Some(threshold) = env_value("RUNMAT_JIT_THRESHOLD", &[]) {
        if let Ok(threshold) = threshold.parse() {
            config.jit.threshold = threshold;
        }
    }

    if let Some(opt_level) = env_value("RUNMAT_JIT_OPT_LEVEL", &[]) {
        config.jit.optimization_level = match opt_level.to_lowercase().as_str() {
            "none" => JitOptLevel::None,
            "size" => JitOptLevel::Size,
            "speed" => JitOptLevel::Speed,
            "aggressive" => JitOptLevel::Aggressive,
            _ => config.jit.optimization_level,
        };
    }

    // GC settings
    if let Some(preset) = env_value("RUNMAT_GC_PRESET", &[]) {
        config.gc.preset = match preset.to_lowercase().as_str() {
            "low-latency" => Some(GcPreset::LowLatency),
            "high-throughput" => Some(GcPreset::HighThroughput),
            "low-memory" => Some(GcPreset::LowMemory),
            "debug" => Some(GcPreset::Debug),
            _ => config.gc.preset,
        };
    }

    if let Some(young_size) = env_value("RUNMAT_GC_YOUNG_SIZE", &[]) {
        if let Ok(young_size) = young_size.parse() {
            config.gc.young_size_mb = Some(young_size);
        }
    }

    if let Some(threads) = env_value("RUNMAT_GC_THREADS", &[]) {
        if let Ok(threads) = threads.parse() {
            config.gc.threads = Some(threads);
        }
    }

    if let Some(stats) = env_bool("RUNMAT_GC_STATS", &[]) {
        config.gc.collect_stats = stats;
    }

    // Plotting settings
    if let Some(plot_mode) = env_value("RUNMAT_PLOT_MODE", &[]) {
        config.plotting.mode = match plot_mode.to_lowercase().as_str() {
            "auto" => PlotMode::Auto,
            "gui" => PlotMode::Gui,
            "headless" => PlotMode::Headless,
            "jupyter" => PlotMode::Jupyter,
            _ => config.plotting.mode,
        };
    }

    if let Some(headless) = env_bool("RUNMAT_PLOT_HEADLESS", &[]) {
        config.plotting.force_headless = headless;
    }

    if let Some(backend) = env_value("RUNMAT_PLOT_BACKEND", &[]) {
        config.plotting.backend = match backend.to_lowercase().as_str() {
            "auto" => PlotBackend::Auto,
            "wgpu" => PlotBackend::Wgpu,
            "static" => PlotBackend::Static,
            "web" => PlotBackend::Web,
            _ => config.plotting.backend,
        };
    }

    // Logging settings
    if let Some(debug) = env_bool("RUNMAT_DEBUG", &[]) {
        config.logging.debug = debug;
    }

    if let Some(log_level) = env_value("RUNMAT_LOG_LEVEL", &[]) {
        config.logging.level = match log_level.to_lowercase().as_str() {
            "error" => LogLevel::Error,
            "warn" => LogLevel::Warn,
            "info" => LogLevel::Info,
            "debug" => LogLevel::Debug,
            "trace" => LogLevel::Trace,
            _ => config.logging.level,
        };
    }

    // Kernel settings
    if let Some(ip) = env_value("RUNMAT_KERNEL_IP", &[]) {
        config.kernel.ip = ip;
    }

    if let Some(key) = env_value("RUNMAT_KERNEL_KEY", &[]) {
        config.kernel.key = Some(key);
    }

    Ok(())
}

pub(crate) fn env_value(primary: &str, aliases: &[&str]) -> Option<String> {
    env::var(primary)
        .ok()
        .or_else(|| aliases.iter().find_map(|alias| env::var(alias).ok()))
}

fn env_bool(primary: &str, aliases: &[&str]) -> Option<bool> {
    env_value(primary, aliases).and_then(|value| parse_bool(&value))
}
