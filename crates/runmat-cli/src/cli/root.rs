use clap::parser::ValueSource;
use clap::{ArgMatches, Parser, Subcommand};
use runmat_config::{PlotBackend, PlotMode};
use runmat_server_client::auth::CredentialStoreMode;
use std::path::PathBuf;
use uuid::Uuid;

use crate::cli::parse::{parse_bool_env, parse_figure_size, parse_log_level_env};
use crate::cli::remote::{FsCommand, OrgCommand, ProjectCommand, RemoteCommand};
use crate::cli::value_types::{
    CaptureFiguresMode, CompressionAlg, FigureSize, GcPreset, LogLevel, OptLevel,
};

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
• Jupyter kernel protocol support with async execution
• Fast startup with snapshotting capabilities
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
  runmat --install-kernel                  # Install as Jupyter kernel
  runmat kernel                            # Start Jupyter kernel
  runmat kernel-connection connection.json # Start with connection file
  runmat version --detailed                # Show detailed version information
"#,
    after_help = r#"
Environment Variables:
  RUNMAT_DEBUG=1              Enable debug logging
  RUNMAT_LOG_LEVEL=debug      Set log level (error, warn, info, debug, trace)
  RUNMAT_KERNEL_IP=127.0.0.1  Kernel IP address  
  RUNMAT_KERNEL_KEY=<key>     Kernel authentication key
  RUNMAT_TIMEOUT=300          Execution timeout in seconds
  RUNMAT_CALLSTACK_LIMIT=200  Maximum call stack frames to record
  RUNMAT_ERROR_NAMESPACE=RunMat Error identifier namespace prefix override
  RUNMAT_CONFIG=<path>        Path to configuration file
  RUNMAT_SNAPSHOT_PATH=<path> Snapshot file to preload standard library
  
  Garbage Collector:
  RUNMAT_GC_PRESET=<preset>   GC preset (low-latency, high-throughput, low-memory, debug)
  RUNMAT_GC_YOUNG_SIZE=<mb>   Young generation size in MB
  RUNMAT_GC_THREADS=<n>       Number of GC threads
  
  JIT Compiler:
  RUNMAT_JIT_ENABLE=1         Enable JIT compilation (default: true)
  RUNMAT_JIT_THRESHOLD=<n>    JIT compilation threshold (default: 10)
  RUNMAT_JIT_OPT_LEVEL=<level> JIT optimization level (none, size, speed, aggressive; default: speed)

For more information, visit: https://github.com/runmat-org/runmat
"#
)]
#[command(propagate_version = true)]
pub struct Cli {
    /// Enable debug logging
    #[arg(short, long, env = "RUNMAT_DEBUG", value_parser = parse_bool_env)]
    pub debug: bool,

    /// Set log level
    #[arg(long, value_enum, env = "RUNMAT_LOG_LEVEL", default_value = "warn", value_parser = parse_log_level_env)]
    pub log_level: LogLevel,

    /// Execution timeout in seconds
    #[arg(long, env = "RUNMAT_TIMEOUT", default_value = "300")]
    pub timeout: u64,

    /// Maximum number of call stack frames to record
    #[arg(long, env = "RUNMAT_CALLSTACK_LIMIT", default_value = "200")]
    pub callstack_limit: usize,

    /// Emit bytecode disassembly for a script (stdout if omitted path)
    #[arg(long, value_name = "PATH", num_args = 0..=1, default_missing_value = "-")]
    pub emit_bytecode: Option<PathBuf>,

    /// Error identifier namespace prefix
    #[arg(long, env = "RUNMAT_ERROR_NAMESPACE")]
    pub error_namespace: Option<String>,

    /// Configuration file path
    #[arg(long, env = "RUNMAT_CONFIG")]
    pub config: Option<PathBuf>,

    /// Disable JIT compilation (use interpreter only)
    #[arg(long, env = "RUNMAT_JIT_DISABLE", value_parser = parse_bool_env)]
    pub no_jit: bool,

    /// JIT compilation threshold (number of executions before JIT)
    #[arg(long, env = "RUNMAT_JIT_THRESHOLD", default_value = "10")]
    pub jit_threshold: u32,

    /// JIT optimization level (none, size, speed, aggressive)
    #[arg(
        long,
        value_enum,
        env = "RUNMAT_JIT_OPT_LEVEL",
        default_value = "speed"
    )]
    pub jit_opt_level: OptLevel,

    /// GC configuration preset
    #[arg(long, value_enum, env = "RUNMAT_GC_PRESET")]
    pub gc_preset: Option<GcPreset>,

    /// Young generation size in MB
    #[arg(long, env = "RUNMAT_GC_YOUNG_SIZE")]
    pub gc_young_size: Option<usize>,

    /// Maximum number of GC threads
    #[arg(long, env = "RUNMAT_GC_THREADS")]
    pub gc_threads: Option<usize>,

    /// Enable GC statistics collection
    #[arg(long, env = "RUNMAT_GC_STATS", value_parser = parse_bool_env)]
    pub gc_stats: bool,

    /// Verbose output for REPL and execution
    #[arg(short, long)]
    pub verbose: bool,

    /// Snapshot file to preload standard library
    #[arg(long, env = "RUNMAT_SNAPSHOT_PATH")]
    pub snapshot: Option<PathBuf>,

    /// Plotting mode
    #[arg(long, value_enum, env = "RUNMAT_PLOT_MODE")]
    pub plot_mode: Option<PlotMode>,

    /// Force headless plotting mode
    #[arg(long, env = "RUNMAT_PLOT_HEADLESS", value_parser = parse_bool_env)]
    pub plot_headless: bool,

    /// Plotting backend
    #[arg(long, value_enum, env = "RUNMAT_PLOT_BACKEND")]
    pub plot_backend: Option<PlotBackend>,

    /// Override scatter target points for GPU decimation
    #[arg(long)]
    pub plot_scatter_target: Option<u32>,

    /// Override surface vertex budget for GPU LOD
    #[arg(long)]
    pub plot_surface_vertex_budget: Option<u64>,

    /// Directory where run artifacts are written
    #[arg(long, env = "RUNMAT_ARTIFACTS_DIR")]
    pub artifacts_dir: Option<PathBuf>,

    /// Path to write artifact manifest JSON
    #[arg(long, env = "RUNMAT_ARTIFACTS_MANIFEST")]
    pub artifacts_manifest: Option<PathBuf>,

    /// Figure capture mode when artifact output is enabled
    #[arg(
        long,
        value_enum,
        env = "RUNMAT_CAPTURE_FIGURES",
        default_value = "auto"
    )]
    pub capture_figures: CaptureFiguresMode,

    /// Figure export size (WIDTHxHEIGHT)
    #[arg(long, env = "RUNMAT_FIGURE_SIZE", default_value = "1280x720", value_parser = parse_figure_size)]
    pub figure_size: FigureSize,

    /// Maximum number of figures to export
    #[arg(long, env = "RUNMAT_MAX_FIGURES", default_value = "8")]
    pub max_figures: usize,

    /// Generate sample configuration file
    #[arg(long)]
    pub generate_config: bool,

    /// Install RunMat as a Jupyter kernel
    #[arg(long)]
    pub install_kernel: bool,

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
    pub timeout: bool,
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
            timeout: Self::was_provided(matches, "timeout"),
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
    /// Start Jupyter kernel
    Kernel {
        /// Kernel IP address
        #[arg(long, env = "RUNMAT_KERNEL_IP", default_value = "127.0.0.1")]
        ip: String,
        /// Kernel authentication key
        #[arg(long, env = "RUNMAT_KERNEL_KEY")]
        key: Option<String>,
        /// Transport protocol
        #[arg(long, default_value = "tcp")]
        transport: String,
        /// Signature scheme
        #[arg(long, default_value = "hmac-sha256")]
        signature_scheme: String,
        /// Shell socket port (0 for auto-assign)
        #[arg(long, env = "RUNMAT_SHELL_PORT", default_value = "0")]
        shell_port: u16,
        /// IOPub socket port (0 for auto-assign)
        #[arg(long, env = "RUNMAT_IOPUB_PORT", default_value = "0")]
        iopub_port: u16,
        /// Stdin socket port (0 for auto-assign)
        #[arg(long, env = "RUNMAT_STDIN_PORT", default_value = "0")]
        stdin_port: u16,
        /// Control socket port (0 for auto-assign)
        #[arg(long, env = "RUNMAT_CONTROL_PORT", default_value = "0")]
        control_port: u16,
        /// Heartbeat socket port (0 for auto-assign)
        #[arg(long, env = "RUNMAT_HB_PORT", default_value = "0")]
        hb_port: u16,
        /// Write connection file to path
        #[arg(long)]
        connection_file: Option<PathBuf>,
    },
    /// Start kernel with connection file
    KernelConnection {
        /// Path to Jupyter connection file
        connection_file: PathBuf,
    },
    /// Execute MATLAB script file
    Run {
        /// Script file to execute
        file: PathBuf,
        /// Arguments to pass to script
        #[arg(last = true)]
        args: Vec<String>,
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
    /// Snapshot management
    Snapshot {
        #[command(subcommand)]
        snapshot_command: SnapshotCommand,
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
pub enum SnapshotCommand {
    /// Create a new snapshot
    Create {
        /// Output snapshot file
        #[arg(short, long)]
        output: PathBuf,
        /// Optimization level
        #[arg(short = 'O', long, value_enum, default_value = "speed")]
        optimization: OptLevel,
        /// Compression algorithm
        #[arg(short, long, value_enum)]
        compression: Option<CompressionAlg>,
    },
    /// Load and inspect a snapshot
    Info {
        /// Snapshot file to inspect
        snapshot: PathBuf,
    },
    /// List available presets
    Presets,
    /// Validate a snapshot file
    Validate {
        /// Snapshot file to validate
        snapshot: PathBuf,
    },
}

#[derive(Subcommand, Clone)]
pub enum ConfigCommand {
    /// Show current configuration
    Show,
    /// Generate sample configuration file
    Generate {
        /// Output file path
        #[arg(short, long, default_value = ".runmat.yaml")]
        output: PathBuf,
    },
    /// Validate configuration file
    Validate {
        /// Config file to validate
        config_file: PathBuf,
    },
    /// Show configuration file locations
    Paths,
}
