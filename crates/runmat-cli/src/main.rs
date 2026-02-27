//! RunMat - High-performance MATLAB/Octave code runtime
//!
//! A modern, V8-inspired MATLAB code runtime with Jupyter kernel support,
//! JIT compilation, generational garbage collection, and excellent developer ergonomics.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use env_logger::Env;
use log::{debug, error, info, warn};
use miette::{SourceOffset, SourceSpan};
use std::env;
use uuid::Uuid;

mod public_api;
mod remote;
mod telemetry_sink;
use runmat_accelerate::AccelerateInitOptions;
use runmat_builtins::Value;
use runmat_config::{self as config, ConfigLoader, PlotBackend, PlotMode, RunMatConfig};
use runmat_core::{
    ExecutionStreamEntry, ExecutionStreamKind, RunError, RunMatSession, TelemetryRunConfig,
    TelemetryRunFinish,
};
use runmat_gc::{
    gc_allocate, gc_collect_major, gc_collect_minor, gc_get_config, gc_stats, GcConfig,
};
use runmat_hir::LoweringContext;
use runmat_ignition::instr::Instr;
use runmat_kernel::{ConnectionInfo, KernelConfig, KernelServer};
use runmat_parser::ParserOptions;
use runmat_runtime::build_runtime_error;
use runmat_snapshot::presets::SnapshotPreset;
use runmat_snapshot::{SnapshotBuilder, SnapshotConfig, SnapshotLoader};
use runmat_time::Instant;
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::time::Duration;
use telemetry_sink::{
    capture_provider_snapshot, sink as telemetry_sink, telemetry_client_id,
    RuntimeExecutionCounters, TelemetryRunKind,
};

fn parser_compat(mode: config::LanguageCompatMode) -> runmat_parser::CompatMode {
    match mode {
        config::LanguageCompatMode::Matlab => runmat_parser::CompatMode::Matlab,
        config::LanguageCompatMode::Strict => runmat_parser::CompatMode::Strict,
    }
}

fn format_frontend_error(err: &RunError, source_name: &str, source: &str) -> Option<String> {
    match err {
        RunError::Syntax(err) => {
            let mut message = err.message.clone();
            if let Some(expected) = &err.expected {
                message = format!("{message} (expected {expected})");
            }
            if let Some(found) = &err.found_token {
                message = format!("{message} (found '{found}')");
            }
            let span = SourceSpan::new(SourceOffset::from(err.position), 1);
            Some(format_diagnostic(
                &message,
                Some("RunMat:SyntaxError"),
                Some(span),
                source_name,
                source,
            ))
        }
        RunError::Semantic(err) => {
            let span = err.span.map(|span| {
                SourceSpan::new(
                    SourceOffset::from(span.start),
                    span.end.saturating_sub(span.start).max(1),
                )
            });
            let identifier = err.identifier.as_deref().or(Some("RunMat:SemanticError"));
            Some(format_diagnostic(
                &err.message,
                identifier,
                span,
                source_name,
                source,
            ))
        }
        RunError::Compile(err) => {
            let span = err.span.map(|span| {
                SourceSpan::new(
                    SourceOffset::from(span.start),
                    span.end.saturating_sub(span.start).max(1),
                )
            });
            let identifier = err.identifier.as_deref().or(Some("RunMat:CompileError"));
            Some(format_diagnostic(
                &err.message,
                identifier,
                span,
                source_name,
                source,
            ))
        }
        RunError::Runtime(err) => {
            Some(err.format_diagnostic_with_source(Some(source_name), Some(source)))
        }
    }
}

fn format_diagnostic(
    message: &str,
    identifier: Option<&str>,
    span: Option<SourceSpan>,
    source_name: &str,
    source: &str,
) -> String {
    let mut builder = build_runtime_error(message);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    if let Some(span) = span {
        builder = builder.with_span(span);
    }
    builder
        .build()
        .format_diagnostic_with_source(Some(source_name), Some(source))
}
#[derive(Parser)]
#[command(
    name = "runmat",
    version = env!("CARGO_PKG_VERSION"),
    about = "High-performance MATLAB/Octave code runtime",
    long_about = r#"
RunMat is a modern, high-performance runtime for MATLAB/Octave code built 
by Dystr (https://dystr.com).

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
  runmat --version --detailed              # Show detailed version information
"#,
    after_help = r#"
Environment Variables:
  RUNMAT_DEBUG=1              Enable debug logging
  RUNMAT_LOG_LEVEL=debug      Set log level (error, warn, info, debug, trace)
  RUNMAT_KERNEL_IP=127.0.0.1  Kernel IP address  
  RUNMAT_KERNEL_KEY=<key>     Kernel authentication key
  RUNMAT_TIMEOUT=300          Execution timeout in seconds
  RUNMAT_CALLSTACK_LIMIT=200  Maximum call stack frames to record
  RUNMAT_ERROR_NAMESPACE=RunMat Error identifier namespace prefix
  RUNMAT_CONFIG=<path>        Path to configuration file
  RUNMAT_SNAPSHOT_PATH=<path> Snapshot file to preload standard library
  
  Garbage Collector:
  RUNMAT_GC_PRESET=<preset>   GC preset (low-latency, high-throughput, low-memory, debug)
  RUNMAT_GC_YOUNG_SIZE=<mb>   Young generation size in MB
  RUNMAT_GC_THREADS=<n>       Number of GC threads
  
  JIT Compiler:
  RUNMAT_JIT_ENABLE=1         Enable JIT compilation (default: true)
  RUNMAT_JIT_THRESHOLD=<n>    JIT compilation threshold (default: 10)
  RUNMAT_JIT_OPT_LEVEL=<0-3>  JIT optimization level (default: 2)

For more information, visit: https://github.com/runmat-org/runmat
"#
)]
#[command(propagate_version = true)]
struct Cli {
    /// Enable debug logging
    #[arg(short, long, env = "RUNMAT_DEBUG", value_parser = parse_bool_env)]
    debug: bool,

    /// Set log level
    #[arg(long, value_enum, env = "RUNMAT_LOG_LEVEL", default_value = "warn", value_parser = parse_log_level_env)]
    log_level: LogLevel,

    /// Execution timeout in seconds
    #[arg(long, env = "RUNMAT_TIMEOUT", default_value = "300")]
    timeout: u64,

    /// Maximum number of call stack frames to record
    #[arg(long, env = "RUNMAT_CALLSTACK_LIMIT", default_value = "200")]
    callstack_limit: usize,

    /// Emit bytecode disassembly for a script (stdout if omitted path)
    #[arg(long, value_name = "PATH", num_args = 0..=1, default_missing_value = "-")]
    emit_bytecode: Option<PathBuf>,

    /// Error identifier namespace prefix
    #[arg(long, env = "RUNMAT_ERROR_NAMESPACE", default_value = "RunMat")]
    error_namespace: String,

    /// Configuration file path
    #[arg(long, env = "RUNMAT_CONFIG")]
    config: Option<PathBuf>,

    // JIT Compiler Options
    /// Disable JIT compilation (use interpreter only)
    #[arg(long, env = "RUNMAT_JIT_DISABLE", value_parser = parse_bool_env)]
    no_jit: bool,

    /// JIT compilation threshold (number of executions before JIT)
    #[arg(long, env = "RUNMAT_JIT_THRESHOLD", default_value = "10")]
    jit_threshold: u32,

    /// JIT optimization level (0-3)
    #[arg(
        long,
        value_enum,
        env = "RUNMAT_JIT_OPT_LEVEL",
        default_value = "speed"
    )]
    jit_opt_level: OptLevel,

    // Garbage Collector Options
    /// GC configuration preset
    #[arg(long, value_enum, env = "RUNMAT_GC_PRESET")]
    gc_preset: Option<GcPreset>,

    /// Young generation size in MB
    #[arg(long, env = "RUNMAT_GC_YOUNG_SIZE")]
    gc_young_size: Option<usize>,

    /// Maximum number of GC threads
    #[arg(long, env = "RUNMAT_GC_THREADS")]
    gc_threads: Option<usize>,

    /// Enable GC statistics collection
    #[arg(long, env = "RUNMAT_GC_STATS", value_parser = parse_bool_env)]
    gc_stats: bool,

    /// Verbose output for REPL and execution
    #[arg(short, long)]
    verbose: bool,

    /// Snapshot file to preload standard library
    #[arg(long, env = "RUNMAT_SNAPSHOT_PATH")]
    snapshot: Option<PathBuf>,

    // Plotting Options
    /// Plotting mode
    #[arg(long, value_enum, env = "RUNMAT_PLOT_MODE")]
    plot_mode: Option<PlotMode>,

    /// Force headless plotting mode
    #[arg(long, env = "RUNMAT_PLOT_HEADLESS", value_parser = parse_bool_env)]
    plot_headless: bool,

    /// Plotting backend
    #[arg(long, value_enum, env = "RUNMAT_PLOT_BACKEND")]
    plot_backend: Option<PlotBackend>,

    /// Override scatter target points for GPU decimation
    #[arg(long)]
    plot_scatter_target: Option<u32>,

    /// Override surface vertex budget for GPU LOD
    #[arg(long)]
    plot_surface_vertex_budget: Option<u64>,

    // config_file is now handled by the config field above
    /// Generate sample configuration file
    #[arg(long)]
    generate_config: bool,

    /// Install RunMat as a Jupyter kernel
    #[arg(long)]
    install_kernel: bool,

    /// Command to execute
    #[command(subcommand)]
    command: Option<Commands>,

    /// MATLAB script file to execute (alternative to subcommands)
    script: Option<PathBuf>,
}

#[derive(Subcommand, Clone)]
enum Commands {
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

    /// Interactive plotting window (requires GUI features)
    Plot {
        /// Plot mode override
        #[arg(long, value_enum)]
        mode: Option<PlotMode>,

        /// Window width
        #[arg(long)]
        width: Option<u32>,

        /// Window height
        #[arg(long)]
        height: Option<u32>,
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
    /// Package manager (coming soon)
    Pkg {
        #[command(subcommand)]
        pkg_command: PkgCommand,
    },
}

#[derive(Subcommand, Clone)]
enum GcCommand {
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
enum SnapshotCommand {
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
enum ConfigCommand {
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

#[derive(Subcommand, Clone)]
enum OrgCommand {
    /// List organizations
    List {
        /// Page size
        #[arg(long)]
        limit: Option<u32>,
        /// Pagination cursor
        #[arg(long)]
        cursor: Option<String>,
    },
}

#[derive(Subcommand, Clone)]
enum ProjectCommand {
    /// List projects for an organization
    List {
        /// Organization id
        #[arg(long)]
        org: Option<Uuid>,
        /// Page size
        #[arg(long)]
        limit: Option<u32>,
        /// Pagination cursor
        #[arg(long)]
        cursor: Option<String>,
    },
    /// Create a project
    Create {
        /// Organization id
        #[arg(long)]
        org: Option<Uuid>,
        /// Project name
        name: String,
    },
    /// Project membership commands
    Members {
        #[command(subcommand)]
        members_command: ProjectMembersCommand,
    },
    /// Project retention policy
    Retention {
        #[command(subcommand)]
        retention_command: ProjectRetentionCommand,
    },
    /// Set default project
    Select {
        /// Project id
        project: Uuid,
    },
}

#[derive(Subcommand, Clone)]
enum ProjectMembersCommand {
    /// List project members
    List {
        /// Project id
        #[arg(long)]
        project: Option<Uuid>,
        /// Page size
        #[arg(long)]
        limit: Option<u32>,
        /// Pagination cursor
        #[arg(long)]
        cursor: Option<String>,
    },
}

#[derive(Subcommand, Clone)]
enum ProjectRetentionCommand {
    /// Show retention settings
    Get {
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
    },
    /// Set retention max versions
    Set {
        /// Max versions to keep (0 = unlimited)
        max_versions: usize,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
    },
}

#[derive(Subcommand, Clone)]
enum FsCommand {
    /// Read a remote file
    Read {
        /// Remote path
        path: String,
        /// Output file (stdout when omitted)
        #[arg(long)]
        output: Option<PathBuf>,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
    },
    /// Write a remote file from local input
    Write {
        /// Remote path
        path: String,
        /// Input file
        input: PathBuf,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
    },
    /// List directory contents
    Ls {
        /// Remote path
        #[arg(default_value = "/")]
        path: String,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
    },
    /// Create directory
    Mkdir {
        /// Remote path
        path: String,
        /// Create parent directories
        #[arg(short, long)]
        recursive: bool,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
    },
    /// Remove file or directory
    Rm {
        /// Remote path
        path: String,
        /// Remove directory
        #[arg(long)]
        dir: bool,
        /// Remove directory recursively
        #[arg(short, long)]
        recursive: bool,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
    },
    /// List file history
    History {
        /// Remote path
        path: String,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
    },
    /// Restore file version
    Restore {
        /// Version id to restore
        version: Uuid,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
    },
    /// Delete file version
    HistoryDelete {
        /// Version id to delete
        version: Uuid,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
    },
    /// List filesystem snapshots
    SnapshotList {
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
    },
    /// Create filesystem snapshot
    SnapshotCreate {
        /// Optional message
        #[arg(long)]
        message: Option<String>,
        /// Parent snapshot override
        #[arg(long)]
        parent: Option<Uuid>,
        /// Optional tag
        #[arg(long)]
        tag: Option<String>,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
    },
    /// Restore filesystem snapshot
    SnapshotRestore {
        /// Snapshot id to restore
        snapshot: Uuid,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
    },
    /// Delete filesystem snapshot
    SnapshotDelete {
        /// Snapshot id to delete
        snapshot: Uuid,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
    },
    /// List snapshot tags
    SnapshotTagList {
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
    },
    /// Set snapshot tag
    SnapshotTagSet {
        /// Snapshot id to tag
        snapshot: Uuid,
        /// Tag name
        tag: String,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
    },
    /// Delete snapshot tag
    SnapshotTagDelete {
        /// Tag name
        tag: String,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
    },
    /// Clone project snapshots into a git repo
    GitClone {
        /// Destination directory
        directory: PathBuf,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
        /// Server override
        #[arg(long)]
        server: Option<String>,
    },
    /// Pull latest snapshots into git repo
    GitPull {
        /// Repo directory
        #[arg(default_value = ".")]
        directory: PathBuf,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
        /// Server override
        #[arg(long)]
        server: Option<String>,
    },
    /// Push git repo history into project snapshots
    GitPush {
        /// Repo directory
        #[arg(default_value = ".")]
        directory: PathBuf,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
        /// Server override
        #[arg(long)]
        server: Option<String>,
    },
    /// List shard manifest history
    ManifestHistory {
        /// Remote path
        path: String,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
    },
    /// Restore shard manifest version
    ManifestRestore {
        /// Version id to restore
        version: Uuid,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
    },
    /// Update shard manifest with partial edits
    ManifestUpdate {
        /// Remote path
        path: String,
        /// Base manifest version id
        #[arg(long)]
        base_version: Uuid,
        /// Manifest JSON file
        #[arg(long)]
        manifest: PathBuf,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
    },
}

#[derive(Subcommand, Clone)]
enum RemoteCommand {
    /// Run a script with remote filesystem
    Run {
        /// Script path
        script: PathBuf,
        /// Project id override
        #[arg(long)]
        project: Option<Uuid>,
        /// Server URL override
        #[arg(long)]
        server: Option<String>,
    },
}

#[derive(Subcommand, Clone)]
enum PkgCommand {
    /// Add a dependency (coming soon)
    Add { name: String },
    /// Remove a dependency (coming soon)
    Remove { name: String },
    /// Install dependencies (coming soon)
    Install,
    /// Update dependencies (coming soon)
    Update,
    /// Publish current package (coming soon)
    Publish,
}

#[derive(Clone, Debug, ValueEnum)]
enum CompressionAlg {
    /// No compression
    None,
    /// LZ4 compression (fast)
    Lz4,
    /// Zstd compression (balanced)
    Zstd,
}

#[derive(Clone, ValueEnum)]
enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

#[derive(Clone, Debug, ValueEnum)]
enum OptLevel {
    /// No optimization
    None,
    /// Minimal optimization
    Size,
    /// Balanced optimization (default)
    Speed,
    /// Maximum optimization
    Aggressive,
}

#[derive(Clone, Debug, ValueEnum)]
enum GcPreset {
    /// Minimize pause times
    LowLatency,
    /// Maximize throughput
    HighThroughput,
    /// Minimize memory usage
    LowMemory,
    /// Debug and analysis mode
    Debug,
}

impl From<LogLevel> for log::LevelFilter {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Error => log::LevelFilter::Error,
            LogLevel::Warn => log::LevelFilter::Warn,
            LogLevel::Info => log::LevelFilter::Info,
            LogLevel::Debug => log::LevelFilter::Debug,
            LogLevel::Trace => log::LevelFilter::Trace,
        }
    }
}

impl From<GcPreset> for GcConfig {
    fn from(preset: GcPreset) -> Self {
        match preset {
            GcPreset::LowLatency => GcConfig::low_latency(),
            GcPreset::HighThroughput => GcConfig::high_throughput(),
            GcPreset::LowMemory => GcConfig::low_memory(),
            GcPreset::Debug => GcConfig::debug(),
        }
    }
}

/// Custom parser for boolean environment variables that accepts both "1"/"0" and "true"/"false"
fn parse_bool_env(s: &str) -> Result<bool, String> {
    match s.to_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        "" => Ok(false), // Empty string defaults to false
        _ => Err(format!(
            "Invalid boolean value '{s}'. Expected: 1/0, true/false, yes/no, on/off"
        )),
    }
}

/// Custom parser for log level environment variables that handles empty strings
fn parse_log_level_env(s: &str) -> Result<LogLevel, String> {
    if s.is_empty() {
        return Ok(LogLevel::Info); // Default to info for empty string
    }

    match s.to_lowercase().as_str() {
        "error" => Ok(LogLevel::Error),
        "warn" => Ok(LogLevel::Warn),
        "info" => Ok(LogLevel::Info),
        "debug" => Ok(LogLevel::Debug),
        "trace" => Ok(LogLevel::Trace),
        _ => Err(format!(
            "Invalid log level '{s}'. Expected: error, warn, info, debug, trace"
        )),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Handle config generation first
    if cli.generate_config {
        let sample_config = ConfigLoader::generate_sample_config();
        println!("{sample_config}");
        return Ok(());
    }

    // Handle kernel installation
    if cli.install_kernel {
        return install_jupyter_kernel().await;
    }

    // Load configuration with CLI overrides
    let mut config = match load_configuration(&cli) {
        Ok(c) => c,
        Err(e) => {
            // Be forgiving for pkg commands: if config path is a directory `.runmat`, continue with defaults
            if matches!(cli.command, Some(Commands::Pkg { .. })) {
                eprintln!("Warning: {e}. Using default configuration for pkg command.");
                RunMatConfig::default()
            } else {
                return Err(e);
            }
        }
    };
    apply_cli_overrides(&mut config, &cli);

    telemetry_sink::init(&config.telemetry);

    // Initialize logging based on final config
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

    // Initialize acceleration provider based on unified configuration
    let accel_options: AccelerateInitOptions = (&config.accelerate).into();
    runmat_accelerate::initialize_acceleration_provider_with(&accel_options);

    info!("RunMat v{} starting", env!("CARGO_PKG_VERSION"));
    debug!("Configuration loaded: {config:?}");

    // Configure Garbage Collector
    configure_gc_from_config(&config)?;
    configure_plotting_from_config(&config);
    report_plot_context_status(&config);

    // Initialize GUI system if needed
    let wants_gui = match config.plotting.mode {
        PlotMode::Gui => true,
        PlotMode::Auto => !config.plotting.force_headless && is_gui_available(),
        PlotMode::Headless | PlotMode::Jupyter => false,
    };

    let _gui_initialized = if wants_gui {
        info!("Initializing GUI plotting system");

        // Register this thread as the main thread for GUI operations
        runmat_plot::register_main_thread();

        // Initialize native window system (handles macOS main thread requirements)
        match runmat_plot::gui::initialize_native_window() {
            Ok(()) => {
                info!("Native window system initialized successfully");
            }
            Err(e) => {
                info!("Native window initialization failed: {e}, using thread manager");
            }
        }

        // Initialize GUI thread manager for cross-platform compatibility
        match runmat_plot::initialize_gui_manager() {
            Ok(()) => {
                info!("GUI thread manager initialized successfully");

                // Perform a health check to ensure the system is working
                match runmat_plot::health_check_global() {
                    Ok(result) => {
                        info!("GUI system health check: {result}");
                        true
                    }
                    Err(e) => {
                        error!("GUI system health check failed: {e}");
                        // Continue anyway, might work when actually needed
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

    // Handle command or script execution
    let command = cli.command.clone();
    let mut script = cli.script.clone();
    let mut emit_bytecode = cli.emit_bytecode.clone();
    if command.is_none() && script.is_none() {
        if let Some(path) = emit_bytecode.clone() {
            let is_matlab = path
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("m"))
                .unwrap_or(false);
            if is_matlab || path.exists() {
                script = Some(path);
                emit_bytecode = Some(PathBuf::from("-"));
            }
        }
    }
    match (command, script) {
        (Some(command), None) => execute_command(command, &cli, &config).await,
        (None, Some(script)) => execute_script(script, emit_bytecode, &cli, &config).await,
        (None, None) => {
            // Default to REPL
            execute_repl(&config).await
        }
        (Some(_), Some(_)) => {
            error!("Cannot specify both command and script file");
            std::process::exit(1);
        }
    }
}

/// Load configuration from files and environment
fn load_configuration(cli: &Cli) -> Result<RunMatConfig> {
    // Check if config file was explicitly provided via CLI (not just env var)
    // We can detect this by checking if RUNMAT_CONFIG env var matches the cli.config value
    let config_from_env = std::env::var("RUNMAT_CONFIG").ok().map(PathBuf::from);

    if let Some(config_file) = &cli.config {
        // If config matches env var, it came from environment - be graceful
        let is_from_env = config_from_env.as_ref() == Some(config_file);

        if config_file.exists() {
            if config_file.is_dir() {
                // Ignore directories per policy; fall back to standard loader
                info!(
                    "Config path is a directory, ignoring: {}",
                    config_file.display()
                );
            } else {
                info!("Loading configuration from: {}", config_file.display());
                return ConfigLoader::load_from_file(config_file);
            }
        } else if !is_from_env {
            // Only exit if explicitly specified via CLI, not env var
            error!(
                "Specified config file does not exist: {}",
                config_file.display()
            );
            std::process::exit(1);
        }
        // If from env var and doesn't exist, fall through to standard loader
    }

    // Use the standard loader (which will also check RUNMAT_CONFIG but gracefully)
    match ConfigLoader::load() {
        Ok(c) => Ok(c),
        Err(e) => {
            // Ignore directory config paths and fall back to defaults
            if let Ok(conf_env) = std::env::var("RUNMAT_CONFIG") {
                let p = PathBuf::from(conf_env);
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
    // JIT settings
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

    // Runtime settings
    config.runtime.timeout = cli.timeout;
    config.runtime.callstack_limit = cli.callstack_limit;
    config.runtime.error_namespace = cli.error_namespace.clone();
    config.runtime.verbose = cli.verbose;
    if let Some(snapshot) = &cli.snapshot {
        config.runtime.snapshot_path = Some(snapshot.clone());
    }

    // GC settings
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

    // Plotting settings
    if let Some(plot_mode) = &cli.plot_mode {
        config.plotting.mode = *plot_mode;
        // Also set environment variable so runtime can see it
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

    // Logging settings
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

    // Apply custom GC settings from config
    if let Some(young_size) = config.gc.young_size_mb {
        gc_config.young_generation_size = young_size * 1024 * 1024; // Convert MB to bytes
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

async fn execute_command(command: Commands, cli: &Cli, config: &RunMatConfig) -> Result<()> {
    match command {
        Commands::Repl { verbose } => {
            // Create a temporary config with the verbose override
            let mut repl_config = config.clone();
            repl_config.runtime.verbose = verbose || config.runtime.verbose;
            execute_repl(&repl_config).await
        }
        Commands::Kernel {
            ip,
            key,
            transport,
            signature_scheme,
            shell_port,
            iopub_port,
            stdin_port,
            control_port,
            hb_port,
            connection_file,
        } => {
            execute_kernel(
                ip,
                key,
                transport,
                signature_scheme,
                shell_port,
                iopub_port,
                stdin_port,
                control_port,
                hb_port,
                connection_file,
                cli.timeout,
            )
            .await
        }
        Commands::KernelConnection { connection_file } => {
            execute_kernel_with_connection(connection_file, cli.timeout).await
        }
        Commands::Run { file, args } => {
            execute_script_with_args(file, args, cli.emit_bytecode.clone(), cli, config).await
        }
        Commands::Version { detailed } => {
            show_version(detailed);
            Ok(())
        }
        Commands::Info => show_system_info(cli).await,
        Commands::AccelInfo { json, reset } => show_accel_info(json, reset).await,
        #[cfg(feature = "wgpu")]
        Commands::AccelCalibrate {
            input,
            dry_run,
            json,
        } => execute_accel_calibrate(input, dry_run, json).await,
        Commands::Gc { gc_command } => execute_gc_command(gc_command).await,
        Commands::Benchmark {
            file,
            iterations,
            jit,
        } => execute_benchmark(file, iterations, jit, cli, config).await,
        Commands::Snapshot { snapshot_command } => execute_snapshot_command(snapshot_command).await,
        Commands::Plot {
            mode,
            width,
            height,
        } => execute_plot_command(mode, width, height, config).await,
        Commands::Config { config_command } => execute_config_command(config_command, config).await,
        Commands::Login {
            server,
            api_key,
            email,
            org,
            project,
        } => remote::execute_login(server, api_key, email, org, project).await,
        Commands::Org { org_command } => remote::execute_org_command(org_command).await,
        Commands::Project { project_command } => {
            remote::execute_project_command(project_command).await
        }
        Commands::Fs { fs_command } => remote::execute_fs_command(fs_command).await,
        Commands::Remote { remote_command } => remote::execute_remote_command(remote_command).await,
        Commands::Pkg { pkg_command } => execute_pkg_command(pkg_command).await,
    }
}

async fn execute_repl(config: &RunMatConfig) -> Result<()> {
    info!("Starting RunMat REPL");
    if config.runtime.verbose {
        info!("Verbose mode enabled");
    }
    let session_start = Instant::now();

    let enable_jit = config.jit.enabled;
    info!(
        "JIT compiler: {}",
        if enable_jit { "enabled" } else { "disabled" }
    );

    // Create enhanced REPL engine with optional snapshot loading
    let mut engine = RunMatSession::with_snapshot(
        enable_jit,
        config.runtime.verbose,
        config.runtime.snapshot_path.as_ref(),
    )
    .context("Failed to create REPL engine")?;
    engine.set_telemetry_consent(config.telemetry.enabled);
    engine.set_telemetry_sink(telemetry_sink());
    engine.set_compat_mode(parser_compat(config.language.compat));
    engine.set_callstack_limit(config.runtime.callstack_limit);
    engine.set_error_namespace(config.runtime.error_namespace.clone());
    if let Some(cid) = telemetry_client_id() {
        engine.set_telemetry_client_id(Some(cid));
    }
    let repl_run = engine.telemetry_run(TelemetryRunConfig {
        kind: TelemetryRunKind::Repl,
        jit_enabled: config.jit.enabled,
        accelerate_enabled: config.accelerate.enabled,
    });

    info!("RunMat REPL ready");

    // Use rustyline for better REPL experience
    use rustyline::error::ReadlineError;
    use rustyline::DefaultEditor;

    let mut rl = DefaultEditor::new().context("Failed to initialize line editor")?;

    let stdin_is_tty = atty::is(atty::Stream::Stdin);
    if !stdin_is_tty {
        let mut buffer = String::new();
        io::stdin()
            .read_to_string(&mut buffer)
            .context("Failed to read piped input")?;
        for raw_line in buffer.lines() {
            if !process_repl_line(raw_line, &mut engine, config).await? {
                break;
            }
        }
        finalize_repl_session(&engine, config, session_start, repl_run);
        return Ok(());
    }

    println!(
        "RunMat v{} by Dystr (https://dystr.com)",
        env!("CARGO_PKG_VERSION")
    );
    println!("Fast, free, modern MATLAB runtime with JIT compilation and GC");
    println!();

    if enable_jit {
        println!(
            "JIT compiler: enabled (Cranelift optimization level: {:?})",
            config.jit.optimization_level
        );
    } else {
        println!("JIT compiler: disabled (interpreter mode)");
    }
    println!(
        "Garbage collector: {:?}",
        config
            .gc
            .preset
            .map(|p| format!("{p:?}"))
            .unwrap_or_else(|| "default".to_string())
    );
    if let Some(snapshot_info) = engine.snapshot_info() {
        println!("{snapshot_info}");
    } else {
        println!("No snapshot loaded - standard library will be compiled on demand");
    }
    println!("Type 'help' for help, 'exit' to quit, '.info' for system information");
    println!();

    loop {
        let readline = rl.readline("runmat> ");
        match readline {
            Ok(line) => {
                let line = line.trim();
                let _ = rl.add_history_entry(line);

                if !process_repl_line(line, &mut engine, config).await? {
                    break;
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("CTRL-C");
                break;
            }
            Err(ReadlineError::Eof) => {
                println!("CTRL-D");
                break;
            }
            Err(err) => {
                println!("Error: {:?}", err);
                break;
            }
        }
    }

    finalize_repl_session(&engine, config, session_start, repl_run);
    Ok(())
}

async fn process_repl_line(
    line: &str,
    engine: &mut RunMatSession,
    config: &RunMatConfig,
) -> Result<bool> {
    if line == "exit" || line == "quit" {
        return Ok(false);
    }
    if line == "help" {
        show_repl_help();
        return Ok(true);
    }
    if line == ".info" {
        engine.show_system_info();
        return Ok(true);
    }
    if line == ".stats" {
        let stats = engine.stats();
        println!("Execution Statistics:");
        println!(
            "  Total: {}, JIT: {}, Interpreter: {}",
            stats.total_executions, stats.jit_compiled, stats.interpreter_fallback
        );
        println!("  Average time: {:.2}ms", stats.average_execution_time_ms);
        return Ok(true);
    }
    if line == ".gc-info" {
        let gc_stats = engine.gc_stats();
        println!("Garbage Collector Statistics:");
        println!("{}", gc_stats.summary_report());
        return Ok(true);
    }
    if line == ".gc" {
        let gc_stats = engine.gc_stats();
        println!("{}", gc_stats.summary_report());
        return Ok(true);
    }
    if line == ".gc-collect" {
        match gc_collect_major() {
            Ok(collected) => println!("Collected {collected} objects"),
            Err(e) => println!("GC collection failed: {e}"),
        }
        return Ok(true);
    }
    if line == ".reset-stats" {
        engine.reset_stats();
        println!("Statistics reset");
        return Ok(true);
    }
    if line.is_empty() {
        return Ok(true);
    }

    match engine.execute(line).await {
        Ok(result) => {
            emit_execution_streams(&result.streams);
            if let Some(error) = result.error {
                eprintln!(
                    "{}",
                    error.format_diagnostic_with_source(Some("<repl>"), Some(line))
                );
            } else if result.value.is_some()
                && config.runtime.verbose
                && result.execution_time_ms > 10
            {
                println!(
                    "  ({}ms {})",
                    result.execution_time_ms,
                    if result.used_jit {
                        "JIT"
                    } else {
                        "interpreter"
                    }
                );
            }
        }
        Err(e) => {
            if let Some(diag) = format_frontend_error(&e, "<repl>", line) {
                eprintln!("{diag}");
            } else {
                eprintln!("Execution error: {e}");
            }
        }
    }

    Ok(true)
}

fn emit_execution_streams(streams: &[ExecutionStreamEntry]) {
    for entry in streams {
        match entry.stream {
            ExecutionStreamKind::Stdout => println!("{}", entry.text),
            ExecutionStreamKind::Stderr => eprintln!("{}", entry.text),
        }
    }
}

fn finalize_repl_session(
    engine: &RunMatSession,
    _config: &RunMatConfig,
    session_start: Instant,
    repl_run: Option<runmat_core::TelemetryRunGuard>,
) {
    let stats = engine.stats();
    let counters = RuntimeExecutionCounters {
        total_executions: stats.total_executions as u64,
        jit_compiled: stats.jit_compiled as u64,
        interpreter_fallback: stats.interpreter_fallback as u64,
    };
    if let Some(run) = repl_run {
        run.finish(TelemetryRunFinish {
            duration: Some(session_start.elapsed()),
            success: true,
            jit_used: stats.jit_compiled > 0,
            error: None,
            counters: Some(counters),
            provider: capture_provider_snapshot(),
        });
    }

    info!("RunMat REPL exiting");
}

#[allow(clippy::too_many_arguments)]
async fn execute_kernel(
    ip: String,
    key: Option<String>,
    transport: String,
    signature_scheme: String,
    shell_port: u16,
    iopub_port: u16,
    stdin_port: u16,
    control_port: u16,
    hb_port: u16,
    connection_file: Option<PathBuf>,
    timeout: u64,
) -> Result<()> {
    info!("Starting RunMat Jupyter kernel");

    let mut connection = ConnectionInfo {
        ip,
        transport,
        signature_scheme,
        key: key.unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
        shell_port,
        iopub_port,
        stdin_port,
        control_port,
        hb_port,
    };

    // Assign ports if they're 0 (auto-assign)
    if shell_port == 0 || iopub_port == 0 || stdin_port == 0 || control_port == 0 || hb_port == 0 {
        connection
            .assign_ports()
            .context("Failed to assign kernel ports")?;
    }

    // Write connection file if requested
    if let Some(path) = connection_file {
        connection
            .write_to_file(&path)
            .with_context(|| format!("Failed to write connection file to {path:?}"))?;
        info!("Connection file written to {path:?}");
    }

    let config = KernelConfig {
        connection,
        session_id: uuid::Uuid::new_v4().to_string(),
        debug: log::log_enabled!(log::Level::Debug),
        execution_timeout: Some(timeout),
    };

    let mut server = KernelServer::new(config);

    info!("Starting kernel server...");
    server
        .start()
        .await
        .context("Failed to start kernel server")?;

    // Keep running until interrupted
    info!("Kernel is ready. Press Ctrl+C to stop.");
    tokio::signal::ctrl_c()
        .await
        .context("Failed to listen for ctrl-c")?;

    info!("Shutting down kernel...");
    server
        .stop()
        .await
        .context("Failed to stop kernel server")?;

    Ok(())
}

async fn execute_kernel_with_connection(connection_file: PathBuf, timeout: u64) -> Result<()> {
    info!("Starting kernel with connection file: {connection_file:?}");

    let connection = ConnectionInfo::from_file(&connection_file)
        .with_context(|| format!("Failed to load connection file: {connection_file:?}"))?;

    let config = KernelConfig {
        connection,
        session_id: uuid::Uuid::new_v4().to_string(),
        debug: log::log_enabled!(log::Level::Debug),
        execution_timeout: Some(timeout),
    };

    let mut server = KernelServer::new(config);

    server
        .start()
        .await
        .context("Failed to start kernel server")?;

    // Keep running until interrupted
    tokio::signal::ctrl_c()
        .await
        .context("Failed to listen for ctrl-c")?;

    server
        .stop()
        .await
        .context("Failed to stop kernel server")?;

    Ok(())
}

async fn execute_script(
    script: PathBuf,
    emit_bytecode_path: Option<PathBuf>,
    cli: &Cli,
    config: &RunMatConfig,
) -> Result<()> {
    execute_script_with_args(script, vec![], emit_bytecode_path, cli, config).await
}

pub(crate) async fn execute_script_with_remote_provider(
    script: PathBuf,
    config: &RunMatConfig,
) -> Result<()> {
    let cli = Cli::parse();
    execute_script_with_args(script, vec![], None, &cli, config).await
}

async fn execute_script_with_args(
    script: PathBuf,
    _args: Vec<String>,
    emit_bytecode_path: Option<PathBuf>,
    _cli: &Cli,
    config: &RunMatConfig,
) -> Result<()> {
    info!("Executing script: {script:?}");

    let content = fs::read_to_string(&script)
        .with_context(|| format!("Failed to read script file: {script:?}"))?;

    if let Some(path) = &emit_bytecode_path {
        let output = emit_bytecode(&content, config)
            .with_context(|| format!("Failed to emit bytecode for {script:?}"))?;
        write_bytecode_output(path, &output)?;
        return Ok(());
    }

    let enable_jit = config.jit.enabled;
    let mut engine = RunMatSession::with_snapshot(
        enable_jit,
        config.runtime.verbose,
        config.runtime.snapshot_path.as_ref(),
    )
    .context("Failed to create execution engine")?;
    engine.set_telemetry_consent(config.telemetry.enabled);
    engine.set_telemetry_sink(telemetry_sink());
    engine.set_compat_mode(parser_compat(config.language.compat));
    engine.set_callstack_limit(config.runtime.callstack_limit);
    engine.set_error_namespace(config.runtime.error_namespace.clone());
    if let Some(cid) = telemetry_client_id() {
        engine.set_telemetry_client_id(Some(cid));
    }
    engine.set_source_name_override(Some(script.to_string_lossy().to_string()));
    let mut script_run = engine.telemetry_run(TelemetryRunConfig {
        kind: TelemetryRunKind::Script,
        jit_enabled: config.jit.enabled,
        accelerate_enabled: config.accelerate.enabled,
    });
    let start_time = Instant::now();
    let result = match engine.execute(&content).await {
        Ok(result) => result,
        Err(err) => {
            if let Some(run) = script_run.take() {
                run.finish(TelemetryRunFinish {
                    duration: Some(start_time.elapsed()),
                    success: false,
                    jit_used: false,
                    error: Some("runtime_error".to_string()),
                    counters: None,
                    provider: capture_provider_snapshot(),
                });
            }
            if let Some(diag) =
                format_frontend_error(&err, script.to_string_lossy().as_ref(), &content)
            {
                eprintln!("{diag}");
            } else {
                eprintln!("Execution error: {err}");
            }
            std::process::exit(1);
        }
    };

    let execution_time = start_time.elapsed();
    emit_execution_streams(&result.streams);

    let provider_snapshot = capture_provider_snapshot();
    let error_payload = result
        .error
        .as_ref()
        .and_then(|err| err.identifier().map(|value| value.to_string()))
        .or_else(|| result.error.as_ref().map(|_| "runtime_error".to_string()));
    let success = error_payload.is_none();
    if let Some(run) = script_run.take() {
        run.finish(TelemetryRunFinish {
            duration: Some(execution_time),
            success,
            jit_used: result.used_jit,
            error: error_payload.clone(),
            counters: None,
            provider: provider_snapshot,
        });
    }

    if let Some(error) = error_payload {
        eprintln!("{error}");
        std::process::exit(1);
    } else {
        if result.used_jit {
            info!("Script executed successfully in {:?} (JIT)", execution_time);
        } else {
            info!("Script executed successfully in {:?}", execution_time);
        }
        if let Some(value) = result.value {
            // For script execution, suppress implicit value printing unless verbose is enabled.
            // Scripts should print their own outputs (e.g., RESULT_ok ...).
            if config.runtime.verbose {
                println!("{value:?}");
            }
        }
    }

    engine.set_source_name_override(None);

    #[cfg(feature = "wgpu")]
    dump_provider_telemetry_if_requested();

    Ok(())
}

fn emit_bytecode(source: &str, config: &RunMatConfig) -> Result<String> {
    let options = ParserOptions::new(parser_compat(config.language.compat));
    let ast = runmat_parser::parse_with_options(source, options)
        .map_err(|err| anyhow::anyhow!(format!("Parse error: {err:?}")))?;
    let lowering = runmat_hir::lower(&ast, &LoweringContext::empty())
        .map_err(|err| anyhow::anyhow!(format!("Lowering error: {err:?}")))?;
    let mut bytecode = runmat_ignition::compile(&lowering.hir, &HashMap::new())
        .map_err(|err| anyhow::anyhow!(format!("Compile error: {err:?}")))?;
    bytecode.var_names = lowering
        .var_names
        .iter()
        .map(|(id, name)| (id.0, name.clone()))
        .collect();
    Ok(disassemble_bytecode(&bytecode))
}

fn write_bytecode_output(path: &PathBuf, output: &str) -> Result<()> {
    if path.as_os_str() == "-" {
        println!("{output}");
        return Ok(());
    }
    let mut file = fs::File::create(path)
        .with_context(|| format!("Failed to create bytecode output file {}", path.display()))?;
    file.write_all(output.as_bytes())
        .with_context(|| format!("Failed to write bytecode output file {}", path.display()))?;
    Ok(())
}

fn disassemble_bytecode(bytecode: &runmat_ignition::Bytecode) -> String {
    let mut out = String::new();
    if !bytecode.var_names.is_empty() {
        let mut entries: Vec<_> = bytecode.var_names.iter().collect();
        entries.sort_by_key(|(idx, _)| *idx);
        let _ = writeln!(&mut out, "# Variables");
        for (idx, name) in entries {
            let _ = writeln!(&mut out, "v{} = {}", idx, name);
        }
        let _ = writeln!(&mut out);
    }
    let _ = writeln!(&mut out, "# Bytecode");
    for (idx, instr) in bytecode.instructions.iter().enumerate() {
        let mut line = format!("{:04}: {}", idx, format_instr(instr, &bytecode.var_names));
        if let Some(span) = bytecode.instr_spans.get(idx) {
            if span.start != 0 || span.end != 0 {
                let _ = write!(line, "  ; span {}..{}", span.start, span.end);
            }
        }
        let _ = writeln!(&mut out, "{line}");
    }
    out
}

fn format_instr(instr: &Instr, var_names: &HashMap<usize, String>) -> String {
    let label = |idx: usize| var_names.get(&idx).map(|n| n.as_str()).unwrap_or("?");
    match instr {
        Instr::LoadVar(idx) => format!("LoadVar {} ({})", idx, label(*idx)),
        Instr::StoreVar(idx) => format!("StoreVar {} ({})", idx, label(*idx)),
        Instr::LoadLocal(idx) => format!("LoadLocal {}", idx),
        Instr::StoreLocal(idx) => format!("StoreLocal {}", idx),
        Instr::EmitVar {
            var_index,
            label: emit,
        } => {
            format!("EmitVar {} ({}) {:?}", var_index, label(*var_index), emit)
        }
        Instr::EmitStackTop { label: emit } => format!("EmitStackTop {:?}", emit),
        other => format!("{other:?}"),
    }
}

#[cfg(feature = "wgpu")]
fn dump_provider_telemetry_if_requested() {
    use std::fs;

    let path = match std::env::var("RUNMAT_TELEMETRY_OUT") {
        Ok(p) if !p.trim().is_empty() => p,
        _ => return,
    };

    let provider = match runmat_accelerate_api::provider() {
        Some(p) => p,
        None => return,
    };

    let info = provider.device_info_struct();
    let telemetry = provider.telemetry_snapshot();

    let mut payload = serde_json::Map::new();
    match serde_json::to_value(&info) {
        Ok(value) => {
            payload.insert("device".to_string(), value);
        }
        Err(err) => log::warn!("Failed to serialize device info for telemetry dump: {err}"),
    }
    match serde_json::to_value(&telemetry) {
        Ok(value) => {
            payload.insert("telemetry".to_string(), value);
        }
        Err(err) => log::warn!("Failed to serialize telemetry snapshot: {err}"),
    }
    if let Some(report) = runmat_accelerate::auto_offload_report() {
        match serde_json::to_value(&report) {
            Ok(value) => {
                payload.insert("auto_offload".to_string(), value);
            }
            Err(err) => log::warn!("Failed to serialize auto-offload report: {err}"),
        }
    }

    let json_payload = serde_json::Value::Object(payload);
    if let Err(err) = fs::write(
        &path,
        serde_json::to_string_pretty(&json_payload).unwrap_or_default(),
    ) {
        log::warn!("Failed to write telemetry snapshot to {path}: {err}");
    }

    let reset_flag = std::env::var("RUNMAT_TELEMETRY_RESET")
        .map(|v| {
            matches!(
                v.as_str(),
                "1" | "true" | "TRUE" | "True" | "yes" | "YES" | "Yes" | "on" | "ON"
            )
        })
        .unwrap_or(false);

    if reset_flag {
        provider.reset_telemetry();
        runmat_accelerate::reset_auto_offload_log();
    }
}

async fn execute_gc_command(gc_command: GcCommand) -> Result<()> {
    match gc_command {
        GcCommand::Stats => {
            let stats = gc_stats();
            println!("{}", stats.summary_report());
        }
        GcCommand::Minor => {
            let start = Instant::now();
            match gc_collect_minor() {
                Ok(collected) => {
                    let duration = start.elapsed();
                    println!("Minor GC collected {collected} objects in {duration:?}");
                }
                Err(e) => {
                    error!("Minor GC failed: {e}");
                    std::process::exit(1);
                }
            }
        }
        GcCommand::Major => {
            let start = Instant::now();
            match gc_collect_major() {
                Ok(collected) => {
                    let duration = start.elapsed();
                    println!("Major GC collected {collected} objects in {duration:?}");
                }
                Err(e) => {
                    error!("Major GC failed: {e}");
                    std::process::exit(1);
                }
            }
        }
        GcCommand::Config => {
            println!("Current GC Configuration:");
            let config = gc_get_config();
            println!(
                "  Young Generation Size: {} MB",
                config.young_generation_size / 1024 / 1024
            );
            println!(
                "  Minor GC Threshold: {} objects",
                config.minor_gc_threshold
            );
            println!(
                "  Major GC Threshold: {} objects",
                config.major_gc_threshold
            );
            println!("  Max GC Threads: {}", config.max_gc_threads);
            println!(
                "  Collection Statistics: {}",
                if config.collect_statistics {
                    "enabled"
                } else {
                    "disabled"
                }
            );
            println!(
                "  Verbose Logging: {}",
                if config.verbose_logging {
                    "enabled"
                } else {
                    "disabled"
                }
            );
        }
        GcCommand::Stress { allocations } => {
            info!("Starting GC stress test with {allocations} allocations");

            let start_time = Instant::now();
            let initial_stats = gc_stats();

            println!("Running GC stress test with {allocations} allocations...");

            // Perform stress test
            let mut _objects = Vec::new();
            for i in 0..allocations {
                let value = Value::Num(i as f64);
                match gc_allocate(value) {
                    Ok(ptr) => {
                        _objects.push(ptr);

                        // Trigger periodic collections to stress the GC
                        if i % 1000 == 0 && i > 0 {
                            let _ = gc_collect_minor();
                        }
                        if i % 5000 == 0 && i > 0 {
                            let _ = gc_collect_major();
                        }
                    }
                    Err(e) => {
                        error!("Allocation failed at iteration {i}: {e}");
                        break;
                    }
                }

                // Progress reporting
                if i % (allocations / 10).max(1) == 0 {
                    println!("  Progress: {i}/{allocations} allocations");
                }
            }

            let duration = start_time.elapsed();
            let final_stats = gc_stats();

            // Report results
            println!("GC Stress Test Results:");
            println!("  Duration: {duration:?}");
            println!("  Allocations completed: {}", _objects.len());
            println!(
                "  Allocation rate: {:.2} allocs/sec",
                _objects.len() as f64 / duration.as_secs_f64()
            );
            println!(
                "  Total collections: {}",
                final_stats
                    .minor_collections
                    .load(std::sync::atomic::Ordering::Relaxed)
                    - initial_stats
                        .minor_collections
                        .load(std::sync::atomic::Ordering::Relaxed)
                    + final_stats
                        .major_collections
                        .load(std::sync::atomic::Ordering::Relaxed)
                    - initial_stats
                        .major_collections
                        .load(std::sync::atomic::Ordering::Relaxed)
            );
            println!(
                "  Final memory: {} bytes",
                final_stats
                    .current_memory_usage
                    .load(std::sync::atomic::Ordering::Relaxed)
            );

            // Force a final cleanup
            match gc_collect_major() {
                Ok(collected) => println!("  Final collection freed {collected} objects"),
                Err(e) => error!("Final collection failed: {e}"),
            }
        }
    }
    Ok(())
}

async fn execute_benchmark(
    file: PathBuf,
    iterations: u32,
    jit: bool,
    _cli: &Cli,
    config: &RunMatConfig,
) -> Result<()> {
    info!("Benchmarking script: {file:?} ({iterations} iterations, JIT: {jit})");

    let content = fs::read_to_string(&file)
        .with_context(|| format!("Failed to read script file: {file:?}"))?;

    let mut engine = RunMatSession::with_snapshot(jit, false, _cli.snapshot.as_ref())
        .context("Failed to create execution engine")?;
    engine.set_telemetry_consent(config.telemetry.enabled);
    engine.set_telemetry_sink(telemetry_sink());
    engine.set_compat_mode(parser_compat(config.language.compat));
    engine.set_callstack_limit(config.runtime.callstack_limit);
    engine.set_error_namespace(config.runtime.error_namespace.clone());
    if let Some(cid) = telemetry_client_id() {
        engine.set_telemetry_client_id(Some(cid));
    }
    engine.set_source_name_override(Some(file.to_string_lossy().to_string()));
    let mut bench_run = engine.telemetry_run(TelemetryRunConfig {
        kind: TelemetryRunKind::Benchmark,
        jit_enabled: jit,
        accelerate_enabled: config.accelerate.enabled,
    });

    let mut total_time = Duration::ZERO;
    let mut jit_executions: u64 = 0;
    let mut interpreter_executions: u64 = 0;

    println!("Warming up...");
    // Warmup runs
    for _ in 0..3 {
        let _ = engine.execute(&content).await.map_err(anyhow::Error::new)?;
    }

    println!("Running benchmark...");
    for i in 1..=iterations {
        let result = engine.execute(&content).await.map_err(anyhow::Error::new)?;

        let iter_duration = Duration::from_millis(result.execution_time_ms);
        if let Some(error) = result
            .error
            .as_ref()
            .and_then(|err| err.identifier().map(|value| value.to_string()))
            .or_else(|| result.error.as_ref().map(|_| "runtime_error".to_string()))
        {
            total_time += iter_duration;
            let counters = RuntimeExecutionCounters {
                total_executions: i as u64,
                jit_compiled: jit_executions + if result.used_jit { 1 } else { 0 },
                interpreter_fallback: interpreter_executions + if result.used_jit { 0 } else { 1 },
            };
            if let Some(run) = bench_run.take() {
                run.finish(TelemetryRunFinish {
                    duration: Some(total_time),
                    success: false,
                    jit_used: result.used_jit,
                    error: Some(error.clone()),
                    counters: Some(counters),
                    provider: capture_provider_snapshot(),
                });
            }
            error!("Benchmark iteration {i} failed: {error}");
            std::process::exit(1);
        }

        total_time += iter_duration;
        if result.used_jit {
            jit_executions += 1;
        } else {
            interpreter_executions += 1;
        }

        if i % 10 == 0 {
            println!("  Completed {i} iterations");
        }
    }

    let avg_time = total_time / iterations;
    println!("\nBenchmark Results:");
    println!("  Total iterations: {iterations}");
    println!("  JIT executions: {jit_executions}");
    println!("  Interpreter executions: {interpreter_executions}");
    println!("  Total time: {total_time:?}");
    println!("  Average time: {avg_time:?}");
    println!(
        "  Throughput: {:.2} executions/second",
        iterations as f64 / total_time.as_secs_f64()
    );

    let counters = RuntimeExecutionCounters {
        total_executions: iterations as u64,
        jit_compiled: jit_executions,
        interpreter_fallback: interpreter_executions,
    };
    if let Some(run) = bench_run.take() {
        run.finish(TelemetryRunFinish {
            duration: Some(total_time),
            success: true,
            jit_used: jit_executions > 0,
            error: None,
            counters: Some(counters),
            provider: capture_provider_snapshot(),
        });
    }

    engine.set_source_name_override(None);

    Ok(())
}

/// Execute the plot command (interactive plotting window)
async fn execute_plot_command(
    mode: Option<PlotMode>,
    width: Option<u32>,
    height: Option<u32>,
    config: &RunMatConfig,
) -> Result<()> {
    info!("Starting interactive plotting window");

    // Determine the plotting mode
    let plot_mode = mode.unwrap_or(config.plotting.mode);

    match plot_mode {
        PlotMode::Auto => {
            // Auto-detect environment for plotting
            if config.plotting.force_headless || !is_gui_available() {
                info!("Auto-detected headless environment");
                execute_headless_plot().await
            } else {
                info!("Auto-detected GUI environment");
                execute_gui_plot(width, height, config).await
            }
        }
        PlotMode::Gui => execute_gui_plot(width, height, config).await,
        PlotMode::Headless => execute_headless_plot().await,
        PlotMode::Jupyter => {
            info!("Jupyter plotting mode not yet implemented");
            println!("Jupyter plotting mode will be available in future releases.");
            Ok(())
        }
    }
}

/// Execute GUI plotting
async fn execute_gui_plot(
    _width: Option<u32>,
    _height: Option<u32>,
    _config: &RunMatConfig,
) -> Result<()> {
    info!("Initializing GUI plotting window");

    // For now, return success since we have the unified plotting system in the runtime
    // This function may not be needed anymore with the new architecture
    match Ok::<(), anyhow::Error>(()) {
        Ok(()) => {
            info!("GUI plotting window closed successfully");
            Ok(())
        }
        Err(e) => {
            error!("GUI plotting failed: {e}");
            Err(anyhow::anyhow!("GUI plotting failed: {}", e))
        }
    }
}

/// Execute headless plotting (generates static images)
async fn execute_headless_plot() -> Result<()> {
    info!("Generating sample static plots");

    // Generate some sample plots to demonstrate headless functionality
    let sample_data_x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let sample_data_y = vec![1.0, 4.0, 2.0, 8.0, 3.0];

    let options = runmat_plot::PlotOptions::default();

    match runmat_plot::plot_line(&sample_data_x, &sample_data_y, "sample_plot.png", options) {
        Ok(()) => {
            println!("Sample plot generated: sample_plot.png");
            Ok(())
        }
        Err(e) => {
            error!("Failed to generate plot: {e}");
            Err(anyhow::anyhow!("Plot generation failed: {}", e))
        }
    }
}

/// Check if GUI is available
fn is_gui_available() -> bool {
    // Simple heuristic: check if we're in a TTY and not in a known headless environment
    use std::env;

    // Check for headless environment indicators
    if env::var("CI").is_ok()
        || env::var("GITHUB_ACTIONS").is_ok()
        || env::var("HEADLESS").is_ok()
        || env::var("NO_GUI").is_ok()
    {
        return false;
    }

    // Check if running in SSH without X11 forwarding
    if env::var("SSH_CLIENT").is_ok() && env::var("DISPLAY").is_err() {
        return false;
    }

    // Use atty to check if stdout is a TTY
    atty::is(atty::Stream::Stdout)
}

/// Execute config command
async fn execute_config_command(
    config_command: ConfigCommand,
    config: &RunMatConfig,
) -> Result<()> {
    match config_command {
        ConfigCommand::Show => {
            println!("Current RunMat Configuration:");
            println!("==============================");

            let yaml =
                serde_yaml::to_string(config).context("Failed to serialize configuration")?;
            println!("{yaml}");
        }
        ConfigCommand::Generate { output } => {
            let sample_config = RunMatConfig::default();
            ConfigLoader::save_to_file(&sample_config, &output)
                .with_context(|| format!("Failed to write config to {}", output.display()))?;

            println!("Sample configuration generated: {}", output.display());
            println!("Edit this file to customize your RunMat settings.");
        }
        ConfigCommand::Validate { config_file } => {
            match ConfigLoader::load_from_file(&config_file) {
                Ok(_) => {
                    println!("Configuration file is valid: {}", config_file.display());
                }
                Err(e) => {
                    error!("Configuration validation failed: {e}");
                    std::process::exit(1);
                }
            }
        }
        ConfigCommand::Paths => {
            println!("RunMat Configuration File Locations:");
            println!("====================================");
            println!();

            if let Ok(config_path) = std::env::var("RUNMAT_CONFIG") {
                println!("Environment override: {config_path}");
            }

            println!("Current directory:");
            if let Ok(current_dir) = std::env::current_dir() {
                for name in &[
                    ".runmat.yaml",
                    ".runmat.yml",
                    ".runmat.json",
                    ".runmat.toml",
                ] {
                    let path = current_dir.join(name);
                    let exists = if path.exists() { " (exists)" } else { "" };
                    println!("  {}{}", path.display(), exists);
                }
            }

            println!();
            println!("Home directory:");
            if let Some(home_dir) = dirs::home_dir() {
                for name in &[".runmat.yaml", ".runmat.yml", ".runmat.json"] {
                    let path = home_dir.join(name);
                    let exists = if path.exists() { " (exists)" } else { "" };
                    println!("  {}{}", path.display(), exists);
                }

                let config_dir = home_dir.join(".config/runmat");
                for name in &["config.yaml", "config.yml", "config.json"] {
                    let path = config_dir.join(name);
                    let exists = if path.exists() { " (exists)" } else { "" };
                    println!("  {}{}", path.display(), exists);
                }
            }

            #[cfg(unix)]
            {
                println!();
                println!("System-wide:");
                for name in &[
                    "/etc/runmat/config.yaml",
                    "/etc/runmat/config.yml",
                    "/etc/runmat/config.json",
                ] {
                    let path = std::path::Path::new(name);
                    let exists = if path.exists() { " (exists)" } else { "" };
                    println!("  {}{}", path.display(), exists);
                }
            }
        }
    }
    Ok(())
}

async fn execute_pkg_command(pkg_command: PkgCommand) -> Result<()> {
    let msg = "RunMat package manager is coming soon. Track progress in the repo.";
    match pkg_command {
        PkgCommand::Add { name } => println!("pkg add {name}: {msg}"),
        PkgCommand::Remove { name } => println!("pkg remove {name}: {msg}"),
        PkgCommand::Install => println!("pkg install: {msg}"),
        PkgCommand::Update => println!("pkg update: {msg}"),
        PkgCommand::Publish => println!("pkg publish: {msg}"),
    }
    Ok(())
}

impl From<CompressionAlg> for runmat_snapshot::CompressionAlgorithm {
    fn from(alg: CompressionAlg) -> Self {
        use runmat_snapshot::CompressionAlgorithm;
        match alg {
            CompressionAlg::None => CompressionAlgorithm::None,
            CompressionAlg::Lz4 => CompressionAlgorithm::Lz4,
            CompressionAlg::Zstd => CompressionAlgorithm::Zstd,
        }
    }
}

async fn execute_snapshot_command(snapshot_command: SnapshotCommand) -> Result<()> {
    match snapshot_command {
        SnapshotCommand::Create {
            output,
            optimization,
            compression,
        } => {
            info!("Creating snapshot: {output:?}");

            let mut config = SnapshotConfig::default();

            // Set compression if specified
            if let Some(comp) = compression {
                config.compression_enabled = !matches!(comp, CompressionAlg::None);
                config.compression_algorithm = comp.into();
            }

            // Note: optimization level affects JIT hints in the data, not the builder directly
            let _optimization_level = match optimization {
                OptLevel::None => runmat_snapshot::OptimizationLevel::None,
                OptLevel::Size => runmat_snapshot::OptimizationLevel::Basic,
                OptLevel::Speed => runmat_snapshot::OptimizationLevel::Aggressive,
                OptLevel::Aggressive => runmat_snapshot::OptimizationLevel::MaxPerformance,
            };

            let builder = SnapshotBuilder::new(config);

            builder
                .build_and_save(&output)
                .with_context(|| format!("Failed to build and save snapshot to {output:?}"))?;

            println!("Snapshot created successfully: {output:?}");
        }
        SnapshotCommand::Info { snapshot } => {
            info!("Loading snapshot info: {snapshot:?}");

            let mut loader = SnapshotLoader::new(SnapshotConfig::default());
            let (loaded, stats) = loader
                .load(&snapshot)
                .with_context(|| format!("Failed to load snapshot from {snapshot:?}"))?;

            println!("Snapshot Information:");
            println!("  File: {snapshot:?}");
            println!("  Version: {}", loaded.metadata.runmat_version);
            println!("  Created: {:?}", loaded.metadata.created_at);
            println!("  Tool Version: {}", loaded.metadata.tool_version);
            println!("  Build Config: {:?}", loaded.metadata.build_config);
            println!("  Builtin Functions: {}", loaded.builtins.functions.len());
            println!(
                "  HIR Cache Functions: {}",
                loaded.hir_cache.functions.len()
            );
            println!("  HIR Cache Patterns: {}", loaded.hir_cache.patterns.len());
            println!(
                "  Bytecode Cache (stdlib): {}",
                loaded.bytecode_cache.stdlib_bytecode.len()
            );
            println!(
                "  Bytecode Cache (sequences): {}",
                loaded.bytecode_cache.operation_sequences.len()
            );
            println!(
                "  Bytecode Cache (hotspots): {}",
                loaded.bytecode_cache.hotspots.len()
            );
            println!("  GC Presets: {}", loaded.gc_presets.presets.len());
            println!("  Load Time: {:?}", stats.load_time);
            println!("  Total Size: {} bytes", stats.total_size);
            println!("  Compressed Size: {} bytes", stats.compressed_size);
            println!("  Compression Ratio: {:.2}x", stats.compression_ratio);
        }
        SnapshotCommand::Presets => {
            println!("Available Snapshot Presets:");
            println!();

            let presets = vec![
                (
                    "development",
                    SnapshotPreset::Development,
                    "Fast development iteration",
                ),
                (
                    "production",
                    SnapshotPreset::Production,
                    "Production deployment",
                ),
                (
                    "high-performance",
                    SnapshotPreset::HighPerformance,
                    "High-performance computing",
                ),
                (
                    "low-memory",
                    SnapshotPreset::LowMemory,
                    "Memory-constrained environments",
                ),
                (
                    "network-optimized",
                    SnapshotPreset::NetworkOptimized,
                    "Network-optimized (minimal size)",
                ),
                (
                    "debug",
                    SnapshotPreset::Debug,
                    "Debug-friendly (maximum validation)",
                ),
            ];

            for (name, preset, description) in presets {
                let config = preset.config();
                println!("  {name}");
                println!("    Description: {description}");
                println!("    Compression: {:?}", config.compression_algorithm);
                println!(
                    "    Validation: {}",
                    if config.validation_enabled {
                        "enabled"
                    } else {
                        "disabled"
                    }
                );
                println!(
                    "    Memory Mapping: {}",
                    if config.memory_mapping_enabled {
                        "enabled"
                    } else {
                        "disabled"
                    }
                );
                println!(
                    "    Parallel Loading: {}",
                    if config.parallel_loading {
                        "enabled"
                    } else {
                        "disabled"
                    }
                );
                println!();
            }
        }
        SnapshotCommand::Validate { snapshot } => {
            info!("Validating snapshot: {snapshot:?}");

            let mut loader = SnapshotLoader::new(SnapshotConfig::default());
            match loader.load(&snapshot) {
                Ok((_, stats)) => {
                    println!("Snapshot validation passed: {snapshot:?}");
                    println!("  Load time: {:?}", stats.load_time);
                    println!("  File size: {} bytes", stats.total_size);
                    if stats.compressed_size > 0 {
                        println!("  Compressed size: {} bytes", stats.compressed_size);
                        println!("  Compression ratio: {:.2}x", stats.compression_ratio);
                    }
                }
                Err(e) => {
                    error!("Snapshot validation failed: {e}");
                    std::process::exit(1);
                }
            }
        }
    }
    Ok(())
}

fn show_version(detailed: bool) {
    println!("RunMat v{}", env!("CARGO_PKG_VERSION"));

    if detailed {
        println!(
            "Built with Rust {}",
            std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "unknown".to_string())
        );
        println!(
            "Target: {}",
            std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string())
        );
        println!(
            "Profile: {}",
            if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            }
        );
    }
}

async fn show_system_info(cli: &Cli) -> Result<()> {
    println!("RunMat System Information");
    println!("==========================");
    println!();

    println!("Version: {}", env!("CARGO_PKG_VERSION"));
    println!(
        "Rust Version: {}",
        std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "unknown".to_string())
    );
    println!(
        "Target: {}",
        std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string())
    );
    println!();

    println!("Runtime Configuration:");
    println!(
        "  JIT Compiler: {}",
        if !cli.no_jit { "enabled" } else { "disabled" }
    );
    println!("  JIT Threshold: {}", cli.jit_threshold);
    println!("  JIT Optimization: {:?}", cli.jit_opt_level);
    println!(
        "  GC Preset: {:?}",
        cli.gc_preset
            .as_ref()
            .map(|p| format!("{p:?}"))
            .unwrap_or_else(|| "default".to_string())
    );
    if let Some(young_size) = cli.gc_young_size {
        println!("  GC Young Generation: {young_size}MB");
    }
    if let Some(threads) = cli.gc_threads {
        println!("  GC Threads: {threads}");
    }
    println!("  GC Statistics: {}", cli.gc_stats);
    println!();

    println!("Environment:");
    println!("  RUNMAT_DEBUG: {:?}", std::env::var("RUNMAT_DEBUG").ok());
    println!(
        "  RUNMAT_LOG_LEVEL: {:?}",
        std::env::var("RUNMAT_LOG_LEVEL").ok()
    );
    println!(
        "  RUNMAT_TIMEOUT: {:?}",
        std::env::var("RUNMAT_TIMEOUT").ok()
    );
    println!(
        "  RUNMAT_JIT_ENABLE: {:?}",
        std::env::var("RUNMAT_JIT_ENABLE").ok()
    );
    println!(
        "  RUNMAT_GC_PRESET: {:?}",
        std::env::var("RUNMAT_GC_PRESET").ok()
    );
    println!();

    // Show GC stats
    let gc_stats = gc_stats();
    println!("Garbage Collector Status:");
    println!("{}", gc_stats.summary_report());
    println!();

    println!("Available Commands:");
    println!("  repl                 Start interactive REPL with JIT");
    println!("  --install-kernel     Install RunMat as Jupyter kernel");
    println!("  kernel               Start Jupyter kernel");
    println!("  kernel-connection    Start kernel with connection file");
    println!("  run <file>           Execute MATLAB script");
    println!("  gc stats             Show GC statistics");
    println!("  gc major             Force major GC collection");
    println!("  benchmark <file>     Benchmark script execution");
    println!("  snapshot create      Create standard library snapshot");
    println!("  snapshot info        Inspect snapshot file");
    println!("  snapshot presets     List available presets");
    println!("  snapshot validate    Validate snapshot file");
    println!("  version              Show version information");
    println!("  info                 Show this system information");

    Ok(())
}

#[cfg(feature = "wgpu")]
async fn show_accel_info(json: bool, reset: bool) -> Result<()> {
    if let Some(p) = runmat_accelerate_api::provider() {
        let info = p.device_info_struct();
        let telemetry = p.telemetry_snapshot();

        if json {
            let mut payload = serde_json::Map::new();
            payload.insert("device".to_string(), serde_json::to_value(&info)?);
            payload.insert("telemetry".to_string(), serde_json::to_value(&telemetry)?);
            if let Some(report) = runmat_accelerate::auto_offload_report() {
                payload.insert("auto_offload".to_string(), serde_json::to_value(&report)?);
            }
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::Value::Object(payload))?
            );
        } else {
            println!("Acceleration Provider Info");
            println!("==========================");
            println!(
                "Device: {} ({})",
                info.name,
                info.backend.clone().unwrap_or_default()
            );
            println!(
                "Fused pipeline cache: hits={}, misses={}",
                telemetry.fusion_cache_hits, telemetry.fusion_cache_misses
            );
            println!(
                "Bind group cache: hits={}, misses={}",
                telemetry.bind_group_cache_hits, telemetry.bind_group_cache_misses
            );
            println!(
                "Reduction defaults: two_pass_mode={}, two_pass_threshold={}, workgroup_size={} (env: RUNMAT_REDUCTION_TWO_PASS / RUNMAT_TWO_PASS_THRESHOLD / RUNMAT_REDUCTION_WG)",
                p.reduction_two_pass_mode().as_str(),
                p.two_pass_threshold(),
                p.default_reduction_workgroup_size()
            );
            if let Some(ms) = p.last_warmup_millis() {
                println!("Warmup: last duration ~{} ms", ms);
            }
            let to_ms = |ns: u64| ns as f64 / 1_000_000.0;
            println!("Telemetry:");
            println!(
                "  uploads: {} bytes, downloads: {} bytes",
                telemetry.upload_bytes, telemetry.download_bytes
            );
            println!(
                "  fused_elementwise: count={} wall_ms={:.3}",
                telemetry.fused_elementwise.count,
                to_ms(telemetry.fused_elementwise.total_wall_time_ns)
            );
            println!(
                "  fused_reduction: count={} wall_ms={:.3}",
                telemetry.fused_reduction.count,
                to_ms(telemetry.fused_reduction.total_wall_time_ns)
            );
            println!(
                "  matmul: count={} wall_ms={:.3}",
                telemetry.matmul.count,
                to_ms(telemetry.matmul.total_wall_time_ns)
            );

            if let Some(report) = runmat_accelerate::auto_offload_report() {
                println!("Auto-offload:");
                println!("  source: {}", report.base_source.as_str());
                println!("  env_overrides_applied: {}", report.env_overrides_applied);
                if let Some(path) = report.cache_path.as_deref() {
                    println!("  cache: {}", path);
                }
                if let Some(ms) = report.calibrate_duration_ms {
                    println!("  last_calibration_ms: {}", ms);
                }
                let thresholds = &report.thresholds;
                println!(
                    "  thresholds: unary={} binary={} reduction={} matmul_flops={} small_batch_dim={} small_batch_min_elems={}",
                    thresholds.unary_min_elems,
                    thresholds.binary_min_elems,
                    thresholds.reduction_min_elems,
                    thresholds.matmul_min_flops,
                    thresholds.small_batch_max_dim,
                    thresholds.small_batch_min_elems
                );
                if !report.decisions.is_empty() {
                    println!("  recent decisions:");
                    for entry in report.decisions.iter().rev().take(5) {
                        println!(
                            "    ts={} op={} decision={:?} reason={:?} elems={:?} flops={:?} batch={:?}",
                            entry.timestamp_ms,
                            entry.operation,
                            entry.decision,
                            entry.reason,
                            entry.elements,
                            entry.flops,
                            entry.batch
                        );
                    }
                }
            }
        }

        if reset {
            p.reset_telemetry();
            runmat_accelerate::reset_auto_offload_log();
        }
    } else if json {
        let payload = serde_json::json!({
            "device": serde_json::Value::Null,
            "telemetry": serde_json::Value::Null,
            "error": "no acceleration provider registered",
        });
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        println!("Acceleration Provider Info");
        println!("==========================");
        println!("No acceleration provider registered");
    }

    Ok(())
}

#[cfg(not(feature = "wgpu"))]
async fn show_accel_info(json: bool, _reset: bool) -> Result<()> {
    if json {
        let payload = serde_json::json!({
            "device": serde_json::Value::Null,
            "telemetry": serde_json::Value::Null,
            "error": "wgpu feature not enabled",
        });
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        println!("Acceleration Provider Info");
        println!("==========================");
        println!("This build was compiled without the 'wgpu' feature. No GPU provider available.");
    }
    Ok(())
}

#[cfg(feature = "wgpu")]
fn print_threshold_delta(label: &str, entry: &runmat_accelerate::ThresholdDeltaEntry) {
    let percent = entry.ratio.map(|r| (r - 1.0) * 100.0);
    match percent {
        Some(p) => println!(
            "  {:<28} {:>12.6e} -> {:>12.6e} (Δ {:>+12.6e}, {:+6.2}%)",
            label, entry.before, entry.after, entry.absolute, p
        ),
        None => println!(
            "  {:<28} {:>12.6e} -> {:>12.6e} (Δ {:>+12.6e})",
            label, entry.before, entry.after, entry.absolute
        ),
    }
}

#[cfg(feature = "wgpu")]
async fn execute_accel_calibrate(input: PathBuf, dry_run: bool, json: bool) -> Result<()> {
    let commit = !dry_run;
    let outcome = runmat_accelerate::apply_auto_offload_calibration_from_file(&input, commit)
        .with_context(|| format!("failed to apply calibration from {}", input.display()))?;

    if json {
        println!("{}", serde_json::to_string_pretty(&outcome)?);
        return Ok(());
    }

    println!("Auto-offload calibration");
    println!("========================");
    println!("Input: {}", input.display());
    if let Some(provider) = &outcome.provider {
        println!(
            "Provider: {} ({}) device_id={}",
            provider.name,
            provider.backend.clone().unwrap_or_default(),
            provider.device_id
        );
    }
    println!("Runs considered: {}", outcome.runs);
    println!("Mode: {}", if commit { "commit" } else { "dry-run" });

    if let Some(delta) = &outcome.delta {
        println!("\nUpdated coefficients (seconds per unit):");
        let mut printed = false;
        if let Some(entry) = &delta.cpu_elem_per_elem {
            print_threshold_delta("cpu_elem_per_elem", entry);
            printed = true;
        }
        if let Some(entry) = &delta.cpu_reduction_per_elem {
            print_threshold_delta("cpu_reduction_per_elem", entry);
            printed = true;
        }
        if let Some(entry) = &delta.cpu_matmul_per_flop {
            print_threshold_delta("cpu_matmul_per_flop", entry);
            printed = true;
        }
        if !printed {
            println!("  (no coefficient changes)");
        }
    } else {
        println!("\nCalibration sample did not yield coefficient adjustments.");
    }

    println!("\nThreshold snapshots:");
    println!(
        "  unary={} -> {}",
        outcome.before.unary_min_elems, outcome.after.unary_min_elems
    );
    println!(
        "  binary={} -> {}",
        outcome.before.binary_min_elems, outcome.after.binary_min_elems
    );
    println!(
        "  reduction={} -> {}",
        outcome.before.reduction_min_elems, outcome.after.reduction_min_elems
    );
    println!(
        "  matmul_flops={} -> {}",
        outcome.before.matmul_min_flops, outcome.after.matmul_min_flops
    );

    if commit {
        if let Some(path) = &outcome.persisted_to {
            println!("\nPersisted calibration cache: {path}");
        }
        println!("Restart RunMat sessions to load the updated thresholds.");
    } else {
        println!(
            "\nDry-run: thresholds were not persisted. Re-run without --dry-run to commit changes."
        );
    }

    Ok(())
}

fn show_repl_help() {
    println!("RunMat REPL Help");
    println!("=================");
    println!();
    println!("Commands:");
    println!("  exit, quit        Exit the REPL");
    println!("  help              Show this help message");
    println!("  .info             Show detailed system information");
    println!("  .stats            Show execution statistics");
    println!("  .gc               Show garbage collector statistics");
    println!("  .gc-info          Show garbage collector statistics with header");
    println!("  .gc-collect       Force garbage collection");
    println!("  .reset-stats      Reset execution statistics");
    println!();
    println!("MATLAB/Octave syntax is supported:");
    println!("  x = 1 + 2                         # Assignment");
    println!("  y = [1, 2, 3]                    # Vectors");
    println!("  z = [1, 2; 3, 4]                 # Matrices");
    println!("  if x > 0; disp('positive'); end  # Control flow");
    println!("  for i = 1:5; disp(i); end        # Loops");
    println!();
    println!("Features:");
    println!("  • JIT compilation with Cranelift for optimal performance");
    println!("  • Generational garbage collection with configurable policies");
    println!("  • High-performance BLAS/LAPACK operations on matrices");
    println!("  • Interpreter fallback for unsupported JIT patterns");
    println!("  • Real-time performance monitoring and statistics");
    println!();
    println!("Performance Tips:");
    println!("  • Repeated code automatically gets JIT compiled");
    println!("  • Matrix operations use optimized BLAS routines");
    println!("  • Use '.stats' to monitor JIT compilation effectiveness");
    println!("  • Use '.gc' to monitor memory usage and collection");
    println!();
    println!("Press Enter after each statement to execute.");
}

/// Install RunMat as a Jupyter kernel
async fn install_jupyter_kernel() -> Result<()> {
    use std::fs;

    info!("Installing RunMat as a Jupyter kernel");

    // Get the path to the current executable
    let current_exe = std::env::current_exe().context("Failed to get current executable path")?;

    // Find Jupyter kernel directory
    let kernel_dir =
        find_jupyter_kernel_dir().context("Failed to find Jupyter kernel directory")?;

    let runmat_kernel_dir = kernel_dir.join("runmat");

    // Create kernel directory
    fs::create_dir_all(&runmat_kernel_dir).with_context(|| {
        format!(
            "Failed to create kernel directory: {}",
            runmat_kernel_dir.display()
        )
    })?;

    // Create kernel.json
    let kernel_json = format!(
        r#"{{
  "argv": [
    "{}",
    "kernel-connection",
    "{{connection_file}}"
  ],
  "display_name": "RunMat",
  "language": "matlab",
  "metadata": {{
    "debugger": false
  }}
}}"#,
        current_exe.display()
    );

    let kernel_json_path = runmat_kernel_dir.join("kernel.json");
    fs::write(&kernel_json_path, kernel_json).with_context(|| {
        format!(
            "Failed to write kernel.json to {}",
            kernel_json_path.display()
        )
    })?;

    // Create logo files (optional - we'll create simple text-based ones for now)
    create_kernel_logos(&runmat_kernel_dir)?;

    println!("RunMat Jupyter kernel installed successfully!");
    println!("Kernel directory: {}", runmat_kernel_dir.display());
    println!();
    println!("You can now start Jupyter and select 'RunMat' as a kernel:");
    println!("  jupyter notebook");
    println!("  # or");
    println!("  jupyter lab");
    println!();
    println!("To verify the installation:");
    println!("  jupyter kernelspec list");

    Ok(())
}

/// Find the Jupyter kernel directory
fn find_jupyter_kernel_dir() -> Result<PathBuf> {
    // Try to get Jupyter data directory using standard methods
    if let Ok(output) = std::process::Command::new("jupyter")
        .args(["--data-dir"])
        .output()
    {
        if output.status.success() {
            let data_dir_str = String::from_utf8_lossy(&output.stdout);
            let data_dir = data_dir_str.trim();
            let kernels_dir = PathBuf::from(data_dir).join("kernels");
            if kernels_dir.exists() || kernels_dir.parent().is_some_and(|p| p.exists()) {
                return Ok(kernels_dir);
            }
        }
    }

    // Fallback to standard locations
    if let Some(home_dir) = dirs::home_dir() {
        // Try user-level installation first
        let user_kernels = home_dir.join(".local/share/jupyter/kernels");
        if user_kernels.exists() || user_kernels.parent().is_some_and(|p| p.exists()) {
            return Ok(user_kernels);
        }

        // macOS specific location
        #[cfg(target_os = "macos")]
        {
            let macos_kernels = home_dir.join("Library/Jupyter/kernels");
            if macos_kernels.exists() || macos_kernels.parent().is_some_and(|p| p.exists()) {
                return Ok(macos_kernels);
            }
        }

        // Windows specific location
        #[cfg(target_os = "windows")]
        {
            if let Ok(appdata) = std::env::var("APPDATA") {
                let windows_kernels = PathBuf::from(appdata).join("jupyter/kernels");
                if windows_kernels.exists() || windows_kernels.parent().is_some_and(|p| p.exists())
                {
                    return Ok(windows_kernels);
                }
            }
        }

        // Default fallback
        let default_kernels = home_dir.join(".local/share/jupyter/kernels");
        return Ok(default_kernels);
    }

    Err(anyhow::anyhow!(
        "Could not determine Jupyter kernel directory. Please install Jupyter first."
    ))
}

/// Create simple kernel logos
fn create_kernel_logos(kernel_dir: &std::path::Path) -> Result<()> {
    // For now, we'll skip logo creation since it requires image processing
    // In a full implementation, you'd want to include actual PNG logos

    // Create a simple text file that indicates logos could be added
    let logo_info = kernel_dir.join("logo-readme.txt");
    fs::write(
        logo_info,
        "RunMat kernel logos can be added here:\n- logo-32x32.png\n- logo-64x64.png",
    )
    .context("Failed to create logo info file")?;

    Ok(())
}

fn format_log_record(
    buf: &mut env_logger::fmt::Formatter,
    record: &log::Record,
) -> std::io::Result<()> {
    let timestamp = buf.timestamp_nanos();
    writeln!(
        buf,
        "[{} {:>5} {}] {}",
        timestamp,
        record.level(),
        record.target(),
        record.args()
    )
}
