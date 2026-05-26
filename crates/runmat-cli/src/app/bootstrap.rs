use anyhow::{Context, Result};
use env_logger::Env;
use log::{debug, error, info, warn};
use runmat_accelerate::AccelerateInitOptions;
use runmat_config::{self as config, ConfigLoader, PlotMode, RunMatConfig};

use crate::app::dispatch;
use crate::cli::{Cli, CliOverrideSources, GcPreset, LogLevel, OptLevel};
use crate::logging::format_log_record;
use crate::telemetry;

pub async fn run_cli(cli: Cli, sources: CliOverrideSources) -> Result<()> {
    if cli.generate_config {
        let sample_config = ConfigLoader::generate_sample_config();
        println!("{sample_config}");
        return Ok(());
    }

    let mut config = match load_configuration(&cli) {
        Ok(c) => c,
        Err(e) => return Err(e),
    };
    apply_cli_overrides(&mut config, &cli, &sources);

    telemetry::init(&config.telemetry);

    let log_level = if config.logging.debug || cli.debug {
        log::LevelFilter::Debug
    } else {
        match config.logging.level {
            config::LogLevel::Error => log::LevelFilter::Error,
            config::LogLevel::Warn => log::LevelFilter::Warn,
            config::LogLevel::Info => log::LevelFilter::Info,
            config::LogLevel::Debug => log::LevelFilter::Debug,
            config::LogLevel::Trace => log::LevelFilter::Trace,
        }
    };

    env_logger::Builder::from_env(Env::default().default_filter_or("info"))
        .filter_level(log_level)
        .format(format_log_record)
        .init();

    let accel_options: AccelerateInitOptions = (&config.accelerate).into();
    runmat_accelerate::initialize_acceleration_provider_with(&accel_options);

    info!("RunMat v{} starting", env!("CARGO_PKG_VERSION"));
    debug!("Configuration loaded: {config:?}");

    configure_gc_from_config(&config)?;
    configure_plotting_from_config(&config);
    report_plot_context_status(&config);

    let wants_gui = match config.plotting.mode {
        PlotMode::Gui => true,
        PlotMode::Auto => !config.plotting.force_headless && is_gui_available(),
        PlotMode::Headless => false,
    };

    let _gui_initialized = if wants_gui {
        info!("Initializing GUI plotting system");

        runmat_plot::register_main_thread();

        match runmat_plot::gui::initialize_native_window() {
            Ok(()) => {
                info!("Native window system initialized successfully");
            }
            Err(e) => {
                info!("Native window initialization failed: {e}, using thread manager");
            }
        }

        match runmat_plot::initialize_gui_manager() {
            Ok(()) => {
                info!("GUI thread manager initialized successfully");

                match runmat_plot::health_check_global() {
                    Ok(result) => {
                        info!("GUI system health check: {result}");
                        true
                    }
                    Err(e) => {
                        error!("GUI system health check failed: {e}");
                        true
                    }
                }
            }
            Err(e) => {
                error!("Failed to initialize GUI thread manager: {e}");
                false
            }
        }
    } else {
        false
    };

    dispatch::dispatch(&cli, &config).await
}

/// Load configuration from files and environment
fn load_configuration(cli: &Cli) -> Result<RunMatConfig> {
    if let Some(config_file) = &cli.config {
        if config_file.is_dir() {
            error!(
                "Specified config path is a directory, expected a file: {}",
                config_file.display()
            );
            std::process::exit(1);
        }
        if !config_file.is_file() {
            error!(
                "Specified config file does not exist: {}",
                config_file.display()
            );
            std::process::exit(1);
        }
        info!("Loading configuration from: {}", config_file.display());
        return ConfigLoader::load_from_file(config_file);
    }

    ConfigLoader::load()
}

/// Apply CLI argument overrides to configuration
fn apply_cli_overrides(config: &mut RunMatConfig, cli: &Cli, sources: &CliOverrideSources) {
    if cli.no_jit {
        config.jit.enabled = false;
    }
    if sources.jit_threshold {
        config.jit.threshold = cli.jit_threshold;
    }
    if sources.jit_opt_level {
        config.jit.optimization_level = match cli.jit_opt_level {
            OptLevel::None => config::JitOptLevel::None,
            OptLevel::Size => config::JitOptLevel::Size,
            OptLevel::Speed => config::JitOptLevel::Speed,
            OptLevel::Aggressive => config::JitOptLevel::Aggressive,
        };
    }

    if sources.callstack_limit {
        config.runtime.callstack_limit = cli.callstack_limit;
    }
    if let Some(error_namespace) = &cli.error_namespace {
        config.runtime.error_namespace = error_namespace.clone();
    }
    if sources.verbose {
        config.runtime.verbose = cli.verbose;
    }
    if let Some(snapshot) = &cli.snapshot {
        config.runtime.snapshot_path = Some(snapshot.clone());
    }
    if config.runtime.error_namespace.trim().is_empty() {
        config.runtime.error_namespace =
            config::error_namespace_for_language_compat(config.language.compat).to_string();
    }

    if let Some(preset) = &cli.gc_preset {
        config.gc.preset = Some(match preset {
            GcPreset::LowLatency => config::GcPreset::LowLatency,
            GcPreset::HighThroughput => config::GcPreset::HighThroughput,
            GcPreset::LowMemory => config::GcPreset::LowMemory,
            GcPreset::Debug => config::GcPreset::Debug,
        });
    }
    if let Some(young_size) = cli.gc_young_size {
        config.gc.young_size_mb = Some(young_size);
    }
    if let Some(threads) = cli.gc_threads {
        config.gc.threads = Some(threads);
    }
    if sources.gc_stats {
        config.gc.collect_stats = cli.gc_stats;
    }

    if let Some(plot_mode) = &cli.plot_mode {
        config.plotting.mode = *plot_mode;
    }
    if cli.plot_headless {
        config.plotting.force_headless = true;
    }
    if let Some(backend) = &cli.plot_backend {
        config.plotting.backend = *backend;
    }
    if let Some(target) = cli.plot_scatter_target {
        config.plotting.scatter_target_points = Some(target);
    }
    if let Some(budget) = cli.plot_surface_vertex_budget {
        config.plotting.surface_vertex_budget = Some(budget);
    }

    if sources.debug {
        config.logging.debug = cli.debug;
    }
    if sources.log_level {
        config.logging.level = match cli.log_level {
            LogLevel::Error => config::LogLevel::Error,
            LogLevel::Warn => config::LogLevel::Warn,
            LogLevel::Info => config::LogLevel::Info,
            LogLevel::Debug => config::LogLevel::Debug,
            LogLevel::Trace => config::LogLevel::Trace,
        };
    }
}

/// Configure GC from the loaded configuration
fn configure_gc_from_config(config: &RunMatConfig) -> Result<()> {
    let mut gc_config = if let Some(preset) = config.gc.preset {
        gc_config_from_preset(preset)
    } else {
        runmat_gc::GcConfig::default()
    };

    if let Some(young_size) = config.gc.young_size_mb {
        gc_config.young_generation_size = young_size * 1024 * 1024;
    }

    if let Some(threads) = config.gc.threads {
        gc_config.max_gc_threads = threads;
    }

    gc_config.collect_statistics = config.gc.collect_stats;
    gc_config.verbose_logging = config.logging.debug || config.runtime.verbose;

    info!(
        "Configuring GC with preset: {:?}",
        config
            .gc
            .preset
            .map(|p| format!("{p:?}"))
            .unwrap_or_else(|| "default".to_string())
    );
    debug!(
        "GC Configuration: young_gen={}MB, threads={}, stats={}",
        gc_config.young_generation_size / 1024 / 1024,
        gc_config.max_gc_threads,
        gc_config.collect_statistics
    );

    runmat_gc::gc_configure(gc_config).context("Failed to configure garbage collector")?;
    Ok(())
}

fn gc_config_from_preset(preset: config::GcPreset) -> runmat_gc::GcConfig {
    match preset {
        config::GcPreset::LowLatency => runmat_gc::GcConfig::low_latency(),
        config::GcPreset::HighThroughput => runmat_gc::GcConfig::high_throughput(),
        config::GcPreset::LowMemory => runmat_gc::GcConfig::low_memory(),
        config::GcPreset::Debug => runmat_gc::GcConfig::debug(),
    }
}

fn configure_plotting_from_config(config: &RunMatConfig) {
    use runmat_runtime::builtins::plotting::{
        set_runtime_plotting_mode, set_scatter_target_points, set_surface_vertex_budget,
        RuntimePlottingMode,
    };

    let runtime_mode = if config.plotting.force_headless {
        RuntimePlottingMode::Static
    } else {
        match config.plotting.mode {
            PlotMode::Auto => RuntimePlottingMode::Auto,
            PlotMode::Gui => RuntimePlottingMode::Interactive,
            PlotMode::Headless => RuntimePlottingMode::Static,
        }
    };
    set_runtime_plotting_mode(runtime_mode);

    if let Some(points) = config.plotting.scatter_target_points {
        set_scatter_target_points(points);
    }
    if let Some(budget) = config.plotting.surface_vertex_budget {
        set_surface_vertex_budget(budget);
    }
}

fn report_plot_context_status(config: &RunMatConfig) {
    if let Err(err) = runmat_runtime::builtins::plotting::context::ensure_context_from_provider() {
        if config.accelerate.enabled {
            warn!("Shared plotting context unavailable: {err}");
        } else {
            debug!("Plotting context unavailable (GPU disabled): {err}");
        }
    }
}

fn is_gui_available() -> bool {
    if std::env::var("CI").is_ok()
        || std::env::var("GITHUB_ACTIONS").is_ok()
        || std::env::var("HEADLESS").is_ok()
        || std::env::var("NO_GUI").is_ok()
    {
        return false;
    }

    if std::env::var("SSH_CLIENT").is_ok() && std::env::var("DISPLAY").is_err() {
        return false;
    }

    atty::is(atty::Stream::Stdout)
}
