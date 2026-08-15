use clap::parser::ValueSource;
use clap::{ArgMatches, Parser, Subcommand};
use runmat_config::runtime::{PlotBackend, PlotMode};
use runmat_server_client::auth::CredentialStoreMode;
use std::path::PathBuf;
use uuid::Uuid;

use crate::cli::parse::{parse_bool_env, parse_figure_size, parse_log_level_env};
use crate::cli::remote::{FsCommand, OrgCommand, ProjectCommand, RemoteCommand};
use crate::cli::value_types::{
    AotOptLevel, CaptureFiguresMode, FigureSize, GcPreset, LogLevel, OptLevel,
};
use crate::cli::ColorMode;
use crate::cli::TestArgs;
use crate::cli::{BatchCommand, ClusterCommand, JobCommand, PackageCommand};

#[derive(Parser, Clone)]
#[command(
    name = "runmat",
    version = env!("CARGO_PKG_VERSION"),
    about = "High-performance MATLAB/Octave code runtime",
    long_about = r#"
RunMat is a modern, high-performance runtime for MATLAB/Octave.

It is built in Rust, and features a V8-inspired tiered execution model with a 
baseline interpreter feeding an optimizing JIT compiler built on Cranelift.

Key features:
• JIT compilation with Cranelift for optimal performance
• Generational garbage collection with configurable policies
• High-performance BLAS/LAPACK operations
• World-class error messages and debugging
• Compatible with MATLAB/Octave syntax and semantics

Performance Features:
• Multi-tier execution: interpreter + JIT compiler
• Adaptive optimization based on hotspot profiling
• Generational GC with write barriers and concurrent collection
• SIMD-optimized mathematical operations
• Zero-copy memory management where possible

Examples:
  runmat                                   # Start interactive REPL with JIT
  runmat --no-jit                          # Start REPL with interpreter only
  runmat --gc-preset low-latency           # Optimize GC for low latency
  runmat script.m                          # Execute MATLAB/Octave script
  runmat --emit-bytecode script.m           # Emit bytecode disassembly
  runmat version --detailed                # Show detailed version information
"#,
    after_help = r#"
Environment Variables:
  RUNMAT_CONFIG=<path>        Explicit path to runmat.toml/runmat.json
  RUNMAT_API_KEY=<token>      Remote API token (remote commands)
  RUNMAT_SERVER_URL=<url>     Remote server URL (remote commands)
  RUNMAT_ORG_ID=<uuid>        Remote org override (remote commands)
  RUNMAT_PROJECT_ID=<uuid>    Remote project override (remote commands)
  RUNMAT_PACKAGE_CACHE_DIR=<path>  Shared package cache override
  NO_COLOR=<non-empty>        Disable ANSI color by default
  CLICOLOR=0                  Disable automatic ANSI color
  CLICOLOR_FORCE=<non-zero>   Force ANSI color for human output
  FORCE_COLOR=<1..3>          Force a specific color level

For more information, visit: https://github.com/runmat-org/runmat
"#
)]
#[command(propagate_version = true)]
pub struct Cli {
    /// Control ANSI styling for human-readable output
    #[arg(long, value_enum, default_value = "auto", global = true)]
    pub color: ColorMode,

    /// Resolve only from locally available package content
    #[arg(long, global = true)]
    pub offline: bool,

    /// Require an up-to-date lockfile
    #[arg(long, global = true)]
    pub locked: bool,

    /// Require the lockfile and prohibit network access or lock mutation
    #[arg(long, global = true)]
    pub frozen: bool,

    /// Enable debug logging
    #[arg(short, long, value_parser = parse_bool_env)]
    pub debug: bool,

    /// Set log level
    #[arg(long, value_enum, default_value = "warn", value_parser = parse_log_level_env)]
    pub log_level: LogLevel,

    /// Maximum number of call stack frames to record
    #[arg(long, default_value = "200")]
    pub callstack_limit: usize,

    /// Emit bytecode disassembly for a script (stdout if omitted path)
    #[arg(long, value_name = "PATH", num_args = 0..=1, default_missing_value = "-")]
    pub emit_bytecode: Option<PathBuf>,

    /// Error identifier namespace prefix
    #[arg(long)]
    pub error_namespace: Option<String>,

    /// Configuration file path
    #[arg(long, env = "RUNMAT_CONFIG")]
    pub config: Option<PathBuf>,

    /// Disable JIT compilation (use interpreter only)
    #[arg(long, value_parser = parse_bool_env)]
    pub no_jit: bool,

    /// JIT compilation threshold (number of executions before JIT)
    #[arg(long, default_value = "10")]
    pub jit_threshold: u32,

    /// JIT optimization level (none, size, speed, aggressive)
    #[arg(long, value_enum, default_value = "speed")]
    pub jit_opt_level: OptLevel,

    /// GC configuration preset
    #[arg(long, value_enum)]
    pub gc_preset: Option<GcPreset>,

    /// Young generation size in MB
    #[arg(long)]
    pub gc_young_size: Option<usize>,

    /// Maximum number of GC threads
    #[arg(long)]
    pub gc_threads: Option<usize>,

    /// Enable GC statistics collection
    #[arg(long, value_parser = parse_bool_env)]
    pub gc_stats: bool,

    /// Verbose output for REPL and execution
    #[arg(short, long)]
    pub verbose: bool,

    /// Plotting mode
    #[arg(long, value_enum)]
    pub plot_mode: Option<PlotMode>,

    /// Force headless plotting mode
    #[arg(long, value_parser = parse_bool_env)]
    pub plot_headless: bool,

    /// Plotting backend
    #[arg(long, value_enum)]
    pub plot_backend: Option<PlotBackend>,

    /// Override scatter target points for GPU decimation
    #[arg(long)]
    pub plot_scatter_target: Option<u32>,

    /// Override surface vertex budget for GPU LOD
    #[arg(long)]
    pub plot_surface_vertex_budget: Option<u64>,

    /// Directory where run artifacts are written
    #[arg(long)]
    pub artifacts_dir: Option<PathBuf>,

    /// Path to write artifact manifest JSON
    #[arg(long)]
    pub artifacts_manifest: Option<PathBuf>,

    /// Figure capture mode when artifact output is enabled
    #[arg(long, value_enum, default_value = "auto")]
    pub capture_figures: CaptureFiguresMode,

    /// Figure export size (WIDTHxHEIGHT)
    #[arg(long, default_value = "1280x720", value_parser = parse_figure_size)]
    pub figure_size: FigureSize,

    /// Maximum number of figures to export
    #[arg(long, default_value = "8")]
    pub max_figures: usize,

    /// Generate sample configuration file
    #[arg(long)]
    pub generate_config: bool,

    /// Command to execute
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// MATLAB script file to execute (alternative to subcommands)
    pub script: Option<PathBuf>,
}

#[derive(Clone, Debug, Default)]
pub struct CliOverrideSources {
    pub debug: bool,
    pub log_level: bool,
    pub callstack_limit: bool,
    pub jit_threshold: bool,
    pub jit_opt_level: bool,
    pub gc_stats: bool,
    pub verbose: bool,
}

impl CliOverrideSources {
    pub fn from_matches(matches: &ArgMatches) -> Self {
        Self {
            debug: Self::was_provided(matches, "debug"),
            log_level: Self::was_provided(matches, "log_level"),
            callstack_limit: Self::was_provided(matches, "callstack_limit"),
            jit_threshold: Self::was_provided(matches, "jit_threshold"),
            jit_opt_level: Self::was_provided(matches, "jit_opt_level"),
            gc_stats: Self::was_provided(matches, "gc_stats"),
            verbose: Self::was_provided(matches, "verbose"),
        }
    }

    fn was_provided(matches: &ArgMatches, id: &str) -> bool {
        matches
            .value_source(id)
            .is_some_and(|source| source != ValueSource::DefaultValue)
    }
}

#[derive(Subcommand, Clone)]
pub enum Commands {
    /// Start interactive REPL
    Repl {
        /// Enable verbose output
        #[arg(short, long)]
        verbose: bool,
    },
    /// Execute MATLAB script file
    Run {
        /// Script file to execute
        file: PathBuf,
        /// Emit structured JSON for .fea studies and sweeps
        #[arg(long)]
        json: bool,
        /// Arguments to pass to script
        #[arg(last = true)]
        args: Vec<String>,
    },
    /// Compile a MATLAB program into a standalone native executable
    Compile {
        /// MATLAB script or function entrypoint
        file: PathBuf,
        /// Output executable path
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Native optimization policy
        #[arg(long, value_enum, default_value = "speed")]
        optimization: AotOptLevel,
        /// Explicit system linker driver
        #[arg(long)]
        linker: Option<PathBuf>,
        /// Retain temporary object, archive, and response-file inputs
        #[arg(long)]
        keep_temps: bool,
        /// Replace an existing output after preserving it until publication succeeds
        #[arg(short, long)]
        force: bool,
    },
    /// Check a MATLAB script or FEA document without running it
    Check {
        /// .m or .fea file to check
        file: PathBuf,
        /// Emit structured JSON
        #[arg(long)]
        json: bool,
        /// Print completed analysis domains as well as diagnostics
        #[arg(short, long)]
        verbose: bool,
        /// Treat a diagnostic level as an error (currently: warnings)
        #[arg(short = 'D', value_name = "LEVEL")]
        deny: Vec<String>,
        /// Add a MATLAB lookup root while checking this source
        #[arg(long = "path", value_name = "DIRECTORY")]
        search_paths: Vec<PathBuf>,
    },
    /// Show version information
    Version {
        /// Show detailed version information
        #[arg(long)]
        detailed: bool,
    },
    /// Show system information
    Info,
    /// Show acceleration provider information
    AccelInfo {
        /// Output provider information and telemetry as JSON
        #[arg(long)]
        json: bool,
        /// Reset provider telemetry counters after printing
        #[arg(long)]
        reset: bool,
    },
    /// Apply auto-offload calibration from suite telemetry results
    #[cfg(feature = "wgpu")]
    AccelCalibrate {
        /// Path to suite results JSON produced by the benchmark harness
        input: PathBuf,
        /// Preview updates without persisting the calibration cache
        #[arg(long)]
        dry_run: bool,
        /// Emit calibration outcome as JSON
        #[arg(long)]
        json: bool,
    },
    /// Garbage collection utilities
    Gc {
        #[command(subcommand)]
        gc_command: GcCommand,
    },
    /// Performance benchmarking
    Benchmark {
        /// Script file to benchmark
        file: PathBuf,
        /// Number of iterations
        #[arg(short, long, default_value = "10")]
        iterations: u32,
        /// Enable JIT for benchmark
        #[arg(long)]
        jit: bool,
    },
    /// Configuration management
    Config {
        #[command(subcommand)]
        config_command: ConfigCommand,
    },
    /// Run against remote filesystem
    Remote {
        #[command(subcommand)]
        remote_command: RemoteCommand,
    },
    /// Authenticate with RunMat server
    Login {
        /// Server URL
        #[arg(long)]
        server: Option<String>,
        /// API key or access token
        #[arg(long)]
        api_key: Option<String>,
        /// Email for interactive login
        #[arg(long)]
        email: Option<String>,
        /// Credential storage mode: auto, secure, file, memory
        #[arg(long, default_value = "file")]
        credential_store: CredentialStoreMode,
        /// Default organization id
        #[arg(long)]
        org: Option<Uuid>,
        /// Default project id
        #[arg(long)]
        project: Option<Uuid>,
    },
    /// Organization management
    Org {
        #[command(subcommand)]
        org_command: OrgCommand,
    },
    /// Project management
    Project {
        #[command(subcommand)]
        project_command: ProjectCommand,
    },
    /// Remote filesystem commands
    Fs {
        #[command(subcommand)]
        fs_command: FsCommand,
    },
    /// Package resolution and shared-cache operations
    Package {
        #[command(subcommand)]
        package_command: PackageCommand,
    },
    /// Discover and run MATLAB-compatible tests
    Test(Box<TestArgs>),
    /// Submit and manage durable local batch jobs
    Batch {
        #[command(subcommand)]
        batch_command: BatchCommand,
    },
    /// Manage remote execution clusters and nodes
    Cluster {
        #[command(subcommand)]
        cluster_command: ClusterCommand,
    },
    /// Submit and manage durable encrypted remote jobs
    Job {
        #[command(subcommand)]
        job_command: JobCommand,
    },
}

#[derive(Subcommand, Clone)]
pub enum GcCommand {
    /// Show GC statistics
    Stats,
    /// Force minor collection
    Minor,
    /// Force major collection
    Major,
    /// Show current configuration
    Config,
    /// Test GC under stress
    Stress {
        /// Number of allocations
        #[arg(short, long, default_value = "10000")]
        allocations: usize,
    },
}

#[derive(Subcommand, Clone)]
pub enum ConfigCommand {
    /// Show resolved runtime configuration
    Show {
        /// Output format
        #[arg(long, value_enum, default_value = "toml")]
        format: ConfigFormat,
    },
    /// Generate a starter runmat config file (project + runtime sections)
    Generate {
        /// Output file path
        #[arg(short, long, default_value = "runmat.toml")]
        output: PathBuf,
        /// Output format (overrides file extension when set)
        #[arg(long, value_enum)]
        format: Option<ConfigFormat>,
    },
    /// Validate configuration file
    Validate {
        /// Config file to validate
        config_file: PathBuf,
    },
    /// Show configuration file locations
    Paths,
}

#[cfg(test)]
mod tests {
    use super::{Cli, Commands};
    use crate::cli::{BatchCommand, ClusterCommand, ClusterStateArg};
    use clap::Parser;

    #[test]
    fn removed_startup_snapshot_flag_is_rejected() {
        let error = match Cli::try_parse_from(["runmat", "--snapshot", "stdlib.snapshot", "main.m"])
        {
            Ok(_) => panic!("the removed startup snapshot flag must not parse"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("--snapshot"));
    }

    #[test]
    fn snapshot_is_no_longer_a_cli_command() {
        let cli = Cli::try_parse_from(["runmat", "snapshot"])
            .expect("ordinary positional input must continue to parse as a script path");
        assert!(cli.command.is_none());
        assert_eq!(
            cli.script.as_deref(),
            Some(std::path::Path::new("snapshot"))
        );
    }

    #[test]
    fn durable_batch_commands_parse_without_entering_script_mode() {
        let cli = Cli::try_parse_from([
            "runmat",
            "batch",
            "submit",
            "job.m",
            "--idempotency-key",
            "stable",
            "--retention-hours",
            "24",
            "--json",
            "--",
            "input",
        ])
        .unwrap();
        let Some(Commands::Batch {
            batch_command:
                BatchCommand::Submit {
                    file,
                    idempotency_key,
                    retention_hours,
                    json,
                    args,
                },
        }) = cli.command
        else {
            panic!("expected durable batch submit");
        };
        assert_eq!(file, std::path::Path::new("job.m"));
        assert_eq!(idempotency_key.as_deref(), Some("stable"));
        assert_eq!(retention_hours, 24);
        assert!(json);
        assert_eq!(args, ["input"]);
    }

    #[test]
    fn cluster_management_commands_parse_without_entering_script_mode() {
        let cli = Cli::try_parse_from([
            "runmat",
            "cluster",
            "state",
            "--org",
            "01234567-89ab-cdef-0123-456789abcdef",
            "clu_0123456789abcdef0123456789abcdef",
            "draining",
            "--json",
        ])
        .unwrap();
        let Some(Commands::Cluster {
            cluster_command:
                ClusterCommand::State {
                    cluster,
                    state,
                    json,
                    ..
                },
        }) = cli.command
        else {
            panic!("expected cluster state command");
        };
        assert_eq!(cluster, "clu_0123456789abcdef0123456789abcdef");
        assert!(matches!(state, ClusterStateArg::Draining));
        assert!(json);
    }
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum ConfigFormat {
    Toml,
    Json,
}
