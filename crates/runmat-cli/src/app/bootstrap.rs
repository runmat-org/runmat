use anyhow::{Context, Result};
use env_logger::Env;
use log::{debug, error, info, warn};
use runmat_accelerate::AccelerateInitOptions;
use runmat_config::runtime::{
    self as config, ConfigLoadOptions, ConfigLoader, ConfigSource, FeaArtifactStoreMode, PlotMode,
    ResolvedRuntimeConfig, RunMatRuntimeConfig,
};

use crate::app::dispatch;
use crate::cli::{Cli, CliOverrideSources, Commands, GcPreset, LogLevel, OptLevel};
use crate::logging::format_log_record;
use crate::telemetry;

/// Install a lazy local host port for CLI-owned worker entry points. This does
/// not read credentials or contact Server until an FEA root asks for admission.
pub fn install_local_fea_admission(
    org: Option<uuid::Uuid>,
) -> Result<runmat_runtime::analysis::fea_run_admission::FeaRunAdmissionGuard> {
    runmat_runtime::analysis::fea_run_admission::replace_fea_run_admission(Some(
        std::sync::Arc::new(super::fea_admission::CliFeaAdmission::new(org)),
    ))
    .map_err(anyhow::Error::msg)
}

pub async fn run_cli(cli: Cli, sources: CliOverrideSources) -> Result<()> {
    crate::presentation::initialize(cli.color, crate::presentation::cli_output_mode(&cli));
    if cli.generate_config {
        let sample_config = ConfigLoader::generate_sample_config();
        println!("{sample_config}");
        return Ok(());
    }

    let mut resolved = load_configuration(&cli)?;
    apply_cli_overrides(&mut resolved.config, &cli, &sources);
    mark_cli_sources(&mut resolved, &cli, &sources);
    let config = resolved.config;

    // Configuration inspection/editing must not initialize execution, GPU or
    // collectors. The values and sources here include explicit CLI overrides.
    if let Some(Commands::Config { config_command }) = &cli.command {
        if matches!(
            config_command,
            crate::cli::ConfigCommand::Show {
                provenance: true,
                ..
            }
        ) {
            let canonical: serde_json::Value =
                serde_json::from_str(&ConfigLoader::render_runtime_config(
                    &config,
                    std::path::Path::new("resolved.json"),
                )?)?;
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &serde_json::json!({"config":canonical,"sources":resolved.sources})
                )?
            );
            return Ok(());
        }
        return crate::commands::config::execute_config_command(config_command.clone(), &config)
            .await;
    }

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

    let terminal_logger = env_logger::Builder::from_env(Env::default().default_filter_or("info"))
        .filter_level(log_level)
        .format(format_log_record)
        .build();
    let logging = crate::observability::init_with_terminal(&config.logging, terminal_logger)
        .map_err(|error| anyhow::anyhow!("failed to initialize CLI logging: {error}"))?;

    // Installing this port performs no credential reads, transport setup or
    // network requests. Only an admitted local FEA root calls the provider.
    let _fea_admission = install_local_fea_admission(super::fea_admission::organization(&cli))?;

    let result = async move {
        telemetry::init(&config.telemetry);
        info!("RunMat v{} starting", env!("CARGO_PKG_VERSION"));
        debug!("Configuration loaded: {config:?}");

        let compilation_only = matches!(
            cli.command,
            Some(
                Commands::Compile { .. } | Commands::Mesh { .. } | Commands::NativeInterface { .. }
            )
        );
        let wants_gui = match config.plotting.mode {
            PlotMode::Gui => !compilation_only && !config.plotting.force_headless,
            PlotMode::Auto => {
                !compilation_only && !config.plotting.force_headless && is_gui_available()
            }
            PlotMode::Headless => false,
        };

        #[cfg(all(feature = "gui", not(target_arch = "wasm32")))]
        if wants_gui && super::native_graphics::applies(&cli) {
            return super::native_graphics::run(cli, config);
        }

        initialize_execution_lane(&config, compilation_only)?;

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

        let _meshing_execution_provider =
            runmat_runtime::meshing::replace_meshing_execution_provider(Some(
                crate::commands::mesh::meshing_execution_provider(),
            ))
            .map_err(|error| {
                anyhow::anyhow!("failed to install native meshing execution: {error}")
            })?;

        dispatch::dispatch(&cli, &config).await
    }
    .await;
    crate::observability::complete_command(result, logging)
}

pub(super) fn initialize_execution_lane(
    config: &RunMatRuntimeConfig,
    compilation_only: bool,
) -> Result<()> {
    configure_gc_from_config(config)?;
    if !compilation_only {
        let accel_options: AccelerateInitOptions = (&config.accelerate).into();
        runmat_accelerate::initialize_acceleration_provider_with(&accel_options);
        configure_plotting_from_config(config);
        configure_analysis_from_config(config)?;
        report_plot_context_status(config);
    }
    Ok(())
}

/// Load configuration from files and environment
fn load_configuration(cli: &Cli) -> Result<ResolvedRuntimeConfig> {
    ConfigLoader::load_with_options(&ConfigLoadOptions {
        explicit_file: cli.config.clone(),
        ..ConfigLoadOptions::native()
    })
}

fn mark_cli_sources(resolved: &mut ResolvedRuntimeConfig, cli: &Cli, sources: &CliOverrideSources) {
    for (field, provided, argument) in [
        ("jit.enabled", cli.no_jit, "--no-jit"),
        ("jit.threshold", sources.jit_threshold, "--jit-threshold"),
        (
            "jit.optimization_level",
            sources.jit_opt_level,
            "--jit-opt-level",
        ),
        (
            "callstack_limit",
            sources.callstack_limit,
            "--callstack-limit",
        ),
        (
            "error_namespace",
            cli.error_namespace.is_some(),
            "--error-namespace",
        ),
        ("verbose", sources.verbose, "--verbose"),
        ("gc.preset", cli.gc_preset.is_some(), "--gc-preset"),
        (
            "gc.young_size_mb",
            cli.gc_young_size.is_some(),
            "--gc-young-size",
        ),
        ("gc.threads", cli.gc_threads.is_some(), "--gc-threads"),
        ("gc.collect_stats", sources.gc_stats, "--gc-stats"),
        ("plotting.mode", cli.plot_mode.is_some(), "--plot-mode"),
        (
            "plotting.force_headless",
            cli.plot_headless,
            "--plot-headless",
        ),
        (
            "plotting.backend",
            cli.plot_backend.is_some(),
            "--plot-backend",
        ),
        (
            "plotting.scatter_target_points",
            cli.plot_scatter_target.is_some(),
            "--plot-scatter-target",
        ),
        (
            "plotting.surface_vertex_budget",
            cli.plot_surface_vertex_budget.is_some(),
            "--plot-surface-vertex-budget",
        ),
        ("logging.debug", sources.debug, "--debug"),
        ("logging.level", sources.log_level, "--log-level"),
    ] {
        if provided {
            resolved.sources.insert(
                format!("runtime.{field}"),
                ConfigSource::CommandLine {
                    argument: argument.into(),
                },
            );
        }
    }
}

/// Apply CLI argument overrides to configuration
fn apply_cli_overrides(config: &mut RunMatRuntimeConfig, cli: &Cli, sources: &CliOverrideSources) {
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
fn configure_gc_from_config(config: &RunMatRuntimeConfig) -> Result<()> {
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

fn configure_plotting_from_config(config: &RunMatRuntimeConfig) {
    use runmat_runtime::builtins::plotting::{
        set_runtime_plotting_mode, set_scatter_target_points, set_scene_export_budget_bytes,
        set_surface_vertex_budget, RuntimePlottingMode,
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
    if let Some(export) = config.plotting.export.as_ref() {
        set_scene_export_budget_bytes(export.scene_budget_bytes);
    }
}

fn configure_analysis_from_config(config: &RunMatRuntimeConfig) -> Result<()> {
    use runmat_runtime::analysis::{self, storage};

    let artifact_root = config
        .fea
        .artifact_root
        .clone()
        .unwrap_or_else(analysis::default_fea_artifact_root);
    let artifact_store = match config.fea.artifact_store {
        Some(FeaArtifactStoreMode::InMemory) => storage::AnalysisArtifactStoreConfig::InMemory,
        Some(FeaArtifactStoreMode::Filesystem) | None => {
            storage::AnalysisArtifactStoreConfig::Filesystem {
                root: artifact_root.clone(),
            }
        }
    };
    storage::configure_artifact_store(artifact_store)
        .map_err(|err| anyhow::anyhow!("Failed to configure FEA artifact store: {err}"))?;
    storage::configure_artifact_retention(storage::AnalysisArtifactRetentionConfig {
        max_runs: config.fea.artifact_max_runs,
        max_runs_per_kind: config.fea.artifact_max_runs_per_kind,
    })
    .map_err(|err| anyhow::anyhow!("Failed to configure FEA artifact retention: {err}"))?;
    analysis::configure_fea_runtime(analysis::FeaRuntimeConfig {
        artifact_root: Some(artifact_root.clone()),
        study_artifact_root: config
            .fea
            .study_artifact_root
            .clone()
            .or_else(|| Some(artifact_root.join("studies"))),
        // Sized from this host's admitted memory (the shared machine-capacity authority).
        publication_staging_bytes: runmat_execution_runner_native::publication_staging_bytes(),
    })
    .map_err(|err| anyhow::anyhow!("Failed to configure FEA runtime: {err}"))?;
    Ok(())
}

fn report_plot_context_status(config: &RunMatRuntimeConfig) {
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
