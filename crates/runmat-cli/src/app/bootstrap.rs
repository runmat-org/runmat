use anyhow::{Context, Result};
use env_logger::Env;
use log::{debug, error, info, warn};
use runmat_accelerate::AccelerateInitOptions;
use runmat_config::{self as config, ConfigLoader, PlotMode, RunMatConfig};

use crate::app::dispatch;
use crate::cli::{Cli, Commands, GcPreset, LogLevel, OptLevel};
use crate::commands::jupyter;
use crate::logging::format_log_record;
use crate::telemetry;

pub async fn run_cli(cli: Cli) -> Result<()> {
    if cli.generate_config {
        let sample_config = ConfigLoader::generate_sample_config();
        println!("{sample_config}");
        return Ok(());
    }

    if cli.install_kernel {
        return jupyter::install_jupyter_kernel().await;
    }

    let mut config = match load_configuration(&cli) {
        Ok(c) => c,
        Err(e) => {
            if matches!(cli.command, Some(Commands::Pkg { .. })) {
                eprintln!("Warning: {e}. Using default configuration for pkg command.");
                RunMatConfig::default()
            } else {
                return Err(e);
            }
        }
    };
    apply_cli_overrides(&mut config, &cli);

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
        PlotMode::Headless | PlotMode::Jupyter => false,
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
    let config_from_env = std::env::var("RUNMAT_CONFIG")
        .ok()
        .map(std::path::PathBuf::from);

    if let Some(config_file) = &cli.config {
        let is_from_env = config_from_env.as_ref() == Some(config_file);

        if config_file.exists() {
            if config_file.is_dir() {
                info!(
                    "Config path is a directory, ignoring: {}",
                    config_file.display()
                );
            } else {
                info!("Loading configuration from: {}", config_file.display());
                return ConfigLoader::load_from_file(config_file);
            }
        } else if !is_from_env {
            error!(
                "Specified config file does not exist: {}",
                config_file.display()
            );
            std::process::exit(1);
        }
    }

    match ConfigLoader::load() {
        Ok(c) => Ok(c),
        Err(e) => {
            if let Ok(conf_env) = std::env::var("RUNMAT_CONFIG") {
                let p = std::path::PathBuf::from(conf_env);
                if p.is_dir() {
                    info!(
                        "Config path from env is a directory, ignoring: {}",
                        p.display()
                    );
                    return Ok(RunMatConfig::default());
                }
            }

            if let Some(home) = dirs::home_dir() {
                let dir = home.join(".runmat");
                if dir.is_dir() {
                    info!(
                        "Home config path is a directory, ignoring: {}",
                        dir.display()
                    );
                    return Ok(RunMatConfig::default());
                }
            }

            Err(e)
        }
    }
}

/// Apply CLI argument overrides to configuration
fn apply_cli_overrides(config: &mut RunMatConfig, cli: &Cli) {
    if cli.no_jit {
        config.jit.enabled = false;
    }
    config.jit.threshold = cli.jit_threshold;
    config.jit.optimization_level = match cli.jit_opt_level {
        OptLevel::None => config::JitOptLevel::None,
        OptLevel::Size => config::JitOptLevel::Size,
        OptLevel::Speed => config::JitOptLevel::Speed,
        OptLevel::Aggressive => config::JitOptLevel::Aggressive,
    };

    config.runtime.timeout = cli.timeout;
    config.runtime.callstack_limit = cli.callstack_limit;
    if let Some(error_namespace) = &cli.error_namespace {
        config.runtime.error_namespace = error_namespace.clone();
    }
    config.runtime.verbose = cli.verbose;
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
    config.gc.collect_stats = cli.gc_stats;

    if let Some(plot_mode) = &cli.plot_mode {
        config.plotting.mode = *plot_mode;
        let env_value = match plot_mode {
            PlotMode::Auto => "auto",
            PlotMode::Gui => "gui",
            PlotMode::Headless => "headless",
            PlotMode::Jupyter => "jupyter",
        };
        std::env::set_var("RUNMAT_PLOT_MODE", env_value);
    }
    if cli.plot_headless {
        config.plotting.force_headless = true;
        std::env::set_var("RUNMAT_PLOT_MODE", "headless");
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

    config.logging.debug = cli.debug;
    config.logging.level = match cli.log_level {
        LogLevel::Error => config::LogLevel::Error,
        LogLevel::Warn => config::LogLevel::Warn,
        LogLevel::Info => config::LogLevel::Info,
        LogLevel::Debug => config::LogLevel::Debug,
        LogLevel::Trace => config::LogLevel::Trace,
    };
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
        set_scatter_target_points, set_surface_vertex_budget,
    };

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
