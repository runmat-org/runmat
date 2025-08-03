//! RustMat - High-performance MATLAB/Octave runtime
//! 
//! A modern, V8-inspired MATLAB runtime with Jupyter kernel support,
//! JIT compilation, generational garbage collection, and excellent developer ergonomics.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use env_logger::Env;
use log::{info, error, debug};
use rustmat_kernel::{KernelConfig, KernelServer, ConnectionInfo};
use rustmat_gc::{GcConfig, gc_configure, gc_stats, gc_collect_major, gc_collect_minor};
use rustmat_repl::ReplEngine;
use std::path::PathBuf;
use std::fs;
use std::time::Duration;


/// RustMat - High-performance MATLAB/Octave runtime
#[derive(Parser)]
#[command(
    name = "rustmat",
    version = "0.0.1",
    about = "High-performance MATLAB/Octave runtime with JIT compilation and GC",
    long_about = r#"
RustMat is a modern, high-performance runtime for MATLAB/Octave code built in Rust.
It features a V8-inspired tiered execution model with a baseline interpreter feeding 
an optimizing JIT compiler built on Cranelift.

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
  rustmat                                    # Start interactive REPL with JIT
  rustmat --no-jit                          # Start REPL with interpreter only
  rustmat --gc-preset low-latency           # Optimize GC for low latency
  rustmat script.m                          # Execute MATLAB script
  rustmat --kernel                          # Start Jupyter kernel
  rustmat --kernel-connection connection.json # Start with connection file
  rustmat --version --detailed              # Show detailed version information
"#,
    after_help = r#"
Environment Variables:
  RUSTMAT_DEBUG=1              Enable debug logging
  RUSTMAT_LOG_LEVEL=debug      Set log level (error, warn, info, debug, trace)
  RUSTMAT_KERNEL_IP=127.0.0.1  Kernel IP address  
  RUSTMAT_KERNEL_KEY=<key>     Kernel authentication key
  RUSTMAT_TIMEOUT=300          Execution timeout in seconds
  RUSTMAT_CONFIG=<path>        Path to configuration file
  
  Garbage Collector:
  RUSTMAT_GC_PRESET=<preset>   GC preset (low-latency, high-throughput, low-memory, debug)
  RUSTMAT_GC_YOUNG_SIZE=<mb>   Young generation size in MB
  RUSTMAT_GC_THREADS=<n>       Number of GC threads
  
  JIT Compiler:
  RUSTMAT_JIT_ENABLE=1         Enable JIT compilation (default: true)
  RUSTMAT_JIT_THRESHOLD=<n>    JIT compilation threshold (default: 10)
  RUSTMAT_JIT_OPT_LEVEL=<0-3>  JIT optimization level (default: 2)

For more information, visit: https://github.com/rustmat/rustmat
"#
)]
#[command(propagate_version = true)]
struct Cli {
    /// Enable debug logging
    #[arg(short, long, env = "RUSTMAT_DEBUG")]
    debug: bool,

    /// Set log level
    #[arg(long, value_enum, env = "RUSTMAT_LOG_LEVEL", default_value = "info")]
    log_level: LogLevel,

    /// Execution timeout in seconds
    #[arg(long, env = "RUSTMAT_TIMEOUT", default_value = "300")]
    timeout: u64,

    /// Configuration file path
    #[arg(long, env = "RUSTMAT_CONFIG")]
    config: Option<PathBuf>,

    // JIT Compiler Options
    /// Disable JIT compilation (use interpreter only)
    #[arg(long, env = "RUSTMAT_JIT_DISABLE")]
    no_jit: bool,

    /// JIT compilation threshold (number of executions before JIT)
    #[arg(long, env = "RUSTMAT_JIT_THRESHOLD", default_value = "10")]
    jit_threshold: u32,

    /// JIT optimization level (0-3)
    #[arg(long, value_enum, env = "RUSTMAT_JIT_OPT_LEVEL", default_value = "speed")]
    jit_opt_level: OptLevel,

    // Garbage Collector Options
    /// GC configuration preset
    #[arg(long, value_enum, env = "RUSTMAT_GC_PRESET")]
    gc_preset: Option<GcPreset>,

    /// Young generation size in MB
    #[arg(long, env = "RUSTMAT_GC_YOUNG_SIZE")]
    gc_young_size: Option<usize>,

    /// Maximum number of GC threads
    #[arg(long, env = "RUSTMAT_GC_THREADS")]
    gc_threads: Option<usize>,

    /// Enable GC statistics collection
    #[arg(long, env = "RUSTMAT_GC_STATS")]
    gc_stats: bool,

    /// Verbose output for REPL and execution
    #[arg(short, long)]
    verbose: bool,

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
        #[arg(long, env = "RUSTMAT_KERNEL_IP", default_value = "127.0.0.1")]
        ip: String,

        /// Kernel authentication key
        #[arg(long, env = "RUSTMAT_KERNEL_KEY")]
        key: Option<String>,

        /// Transport protocol
        #[arg(long, default_value = "tcp")]
        transport: String,

        /// Signature scheme
        #[arg(long, default_value = "hmac-sha256")]
        signature_scheme: String,

        /// Shell socket port (0 for auto-assign)
        #[arg(long, env = "RUSTMAT_SHELL_PORT", default_value = "0")]
        shell_port: u16,

        /// IOPub socket port (0 for auto-assign)
        #[arg(long, env = "RUSTMAT_IOPUB_PORT", default_value = "0")]
        iopub_port: u16,

        /// Stdin socket port (0 for auto-assign)
        #[arg(long, env = "RUSTMAT_STDIN_PORT", default_value = "0")]
        stdin_port: u16,

        /// Control socket port (0 for auto-assign)
        #[arg(long, env = "RUSTMAT_CONTROL_PORT", default_value = "0")]
        control_port: u16,

        /// Heartbeat socket port (0 for auto-assign)
        #[arg(long, env = "RUSTMAT_HB_PORT", default_value = "0")]
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

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let log_level = if cli.debug {
        log::LevelFilter::Debug
    } else {
        cli.log_level.clone().into()
    };

    env_logger::Builder::from_env(Env::default().default_filter_or("info"))
        .filter_level(log_level)
        .init();

    info!("RustMat v{} starting", env!("CARGO_PKG_VERSION"));

    // Configure Garbage Collector
    configure_gc(&cli)?;

    // Handle command or script execution
    let command = cli.command.clone();
    let script = cli.script.clone();
    match (command, script) {
        (Some(command), None) => {
            execute_command(command, &cli).await
        }
        (None, Some(script)) => {
            execute_script(script, &cli).await
        }
        (None, None) => {
            // Default to REPL
            execute_repl_with_config(&cli).await
        }
        (Some(_), Some(_)) => {
            error!("Cannot specify both command and script file");
            std::process::exit(1);
        }
    }
}

fn configure_gc(cli: &Cli) -> Result<()> {
    let mut config = if let Some(preset) = &cli.gc_preset {
        preset.clone().into()
    } else {
        GcConfig::default()
    };

    // Apply custom GC settings
    if let Some(young_size) = cli.gc_young_size {
        config.young_generation_size = young_size * 1024 * 1024; // Convert MB to bytes
    }

    if let Some(threads) = cli.gc_threads {
        config.max_gc_threads = threads;
    }

    config.collect_statistics = cli.gc_stats;
    config.verbose_logging = cli.debug || cli.verbose;

    info!("Configuring GC with preset: {:?}", cli.gc_preset.as_ref().map(|p| format!("{p:?}")).unwrap_or_else(|| "default".to_string()));
    debug!("GC Configuration: young_gen={}MB, threads={}, stats={}", 
        config.young_generation_size / 1024 / 1024,
        config.max_gc_threads,
        config.collect_statistics);

    gc_configure(config)
        .context("Failed to configure garbage collector")?;

    Ok(())
}

async fn execute_command(command: Commands, cli: &Cli) -> Result<()> {
    match command {
        Commands::Repl { verbose } => {
            execute_repl_with_config_and_verbose(cli, verbose).await
        }
        Commands::Kernel { 
            ip, key, transport, signature_scheme,
            shell_port, iopub_port, stdin_port, control_port, hb_port,
            connection_file
        } => {
            execute_kernel(
                ip, key, transport, signature_scheme,
                shell_port, iopub_port, stdin_port, control_port, hb_port,
                connection_file, cli.timeout
            ).await
        }
        Commands::KernelConnection { connection_file } => {
            execute_kernel_with_connection(connection_file, cli.timeout).await
        }
        Commands::Run { file, args } => {
            execute_script_with_args(file, args, cli).await
        }
        Commands::Version { detailed } => {
            show_version(detailed);
            Ok(())
        }
        Commands::Info => {
            show_system_info(cli).await
        }
        Commands::Gc { gc_command } => {
            execute_gc_command(gc_command).await
        }
        Commands::Benchmark { file, iterations, jit } => {
            execute_benchmark(file, iterations, jit, cli).await
        }
    }
}

async fn execute_repl_with_config(cli: &Cli) -> Result<()> {
    execute_repl_with_config_and_verbose(cli, cli.verbose).await
}

async fn execute_repl_with_config_and_verbose(cli: &Cli, verbose: bool) -> Result<()> {
    info!("Starting RustMat REPL");
    if verbose || cli.verbose {
        info!("Verbose mode enabled");
    }

    let enable_jit = !cli.no_jit;
    info!("JIT compiler: {}", if enable_jit { "enabled" } else { "disabled" });

    // Create enhanced REPL engine
    let mut engine = ReplEngine::with_options(enable_jit, verbose || cli.verbose)
        .context("Failed to create REPL engine")?;

    info!("RustMat REPL ready");

    // Use rustyline for better REPL experience
    use std::io::{self, Write};

    println!("RustMat Interactive Console v{}", env!("CARGO_PKG_VERSION"));
    println!("High-performance MATLAB/Octave runtime with JIT compilation");
    if enable_jit {
        println!("JIT compiler: enabled (Cranelift optimization level: {:?})", cli.jit_opt_level);
    } else {
        println!("JIT compiler: disabled (interpreter mode)");
    }
    println!("Garbage collector: {:?}", cli.gc_preset.as_ref().map(|p| format!("{p:?}")).unwrap_or_else(|| "default".to_string()));
    println!("Type 'help' for help, 'exit' to quit, '.info' for system information");
    println!();

    let mut input = String::new();
    loop {
        print!("rustmat> ");
        io::stdout().flush().unwrap();
        
        input.clear();
        match io::stdin().read_line(&mut input) {
            Ok(_) => {
                let line = input.trim();
                if line == "exit" || line == "quit" {
                    break;
                }
                if line == "help" {
                    show_repl_help();
                    continue;
                }
                if line == ".info" {
                    engine.show_system_info();
                    continue;
                }
                if line == ".stats" {
                    let stats = engine.stats();
                    println!("Execution Statistics:");
                    println!("  Total: {}, JIT: {}, Interpreter: {}", 
                        stats.total_executions, stats.jit_compiled, stats.interpreter_fallback);
                    println!("  Average time: {:.2}ms", stats.average_execution_time_ms);
                    continue;
                }
                if line == ".gc" {
                    let gc_stats = engine.gc_stats();
                    println!("{}", gc_stats.summary_report());
                    continue;
                }
                if line.is_empty() {
                    continue;
                }

                // Execute the input using the enhanced engine
                match engine.execute(line) {
                    Ok(result) => {
                        if let Some(error) = result.error {
                            eprintln!("Error: {error}");
                        } else if let Some(value) = result.value {
                            println!("ans = {value:?}");
                            if verbose && result.execution_time_ms > 10 {
                                println!("  ({}ms {})", 
                                    result.execution_time_ms,
                                    if result.used_jit { "JIT" } else { "interpreter" });
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Execution error: {e}");
                    }
                }
            }
            Err(e) => {
                eprintln!("Error reading input: {e}");
                break;
            }
        }
    }

    info!("RustMat REPL exiting");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn execute_kernel(
    ip: String, key: Option<String>, transport: String, signature_scheme: String,
    shell_port: u16, iopub_port: u16, stdin_port: u16, control_port: u16, hb_port: u16,
    connection_file: Option<PathBuf>, timeout: u64
) -> Result<()> {
    info!("Starting RustMat Jupyter kernel");

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
        connection.assign_ports()
            .context("Failed to assign kernel ports")?;
    }

    // Write connection file if requested
    if let Some(path) = connection_file {
        connection.write_to_file(&path)
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
    server.start().await
        .context("Failed to start kernel server")?;

    // Keep running until interrupted
    info!("Kernel is ready. Press Ctrl+C to stop.");
    tokio::signal::ctrl_c().await
        .context("Failed to listen for ctrl-c")?;

    info!("Shutting down kernel...");
    server.stop().await
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
    
    server.start().await
        .context("Failed to start kernel server")?;

    // Keep running until interrupted  
    tokio::signal::ctrl_c().await
        .context("Failed to listen for ctrl-c")?;

    server.stop().await
        .context("Failed to stop kernel server")?;

    Ok(())
}

async fn execute_script(script: PathBuf, cli: &Cli) -> Result<()> {
    execute_script_with_args(script, vec![], cli).await
}

async fn execute_script_with_args(script: PathBuf, _args: Vec<String>, cli: &Cli) -> Result<()> {
    info!("Executing script: {script:?}");

    let content = fs::read_to_string(&script)
        .with_context(|| format!("Failed to read script file: {script:?}"))?;

    let enable_jit = !cli.no_jit;
    let mut engine = ReplEngine::with_options(enable_jit, cli.verbose)
        .context("Failed to create execution engine")?;

    let start_time = std::time::Instant::now();
    let result = engine.execute(&content)
        .context("Failed to execute script")?;

    let execution_time = start_time.elapsed();

    if let Some(error) = result.error {
        error!("Script execution failed: {error}");
        std::process::exit(1);
    } else {
        info!("Script executed successfully in {:?} ({})", 
            execution_time,
            if result.used_jit { "JIT" } else { "interpreter" });
        if let Some(value) = result.value {
            println!("{value:?}");
        }
    }

    Ok(())
}

async fn execute_gc_command(gc_command: GcCommand) -> Result<()> {
    match gc_command {
        GcCommand::Stats => {
            let stats = gc_stats();
            println!("{}", stats.summary_report());
        }
        GcCommand::Minor => {
            let start = std::time::Instant::now();
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
            let start = std::time::Instant::now();
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
            // TODO: Add method to get current config from GC
            println!("  (Configuration introspection not yet implemented)");
        }
        GcCommand::Stress { allocations } => {
            info!("Starting GC stress test with {allocations} allocations");
            // TODO: Implement GC stress test
            println!("GC stress test not yet implemented");
        }
    }
    Ok(())
}

async fn execute_benchmark(file: PathBuf, iterations: u32, jit: bool, _cli: &Cli) -> Result<()> {
    info!("Benchmarking script: {file:?} ({iterations} iterations, JIT: {jit})");

    let content = fs::read_to_string(&file)
        .with_context(|| format!("Failed to read script file: {file:?}"))?;

    let mut engine = ReplEngine::with_options(jit, false)
        .context("Failed to create execution engine")?;

    let mut total_time = Duration::ZERO;
    let mut jit_executions = 0;
    let mut interpreter_executions = 0;

    println!("Warming up...");
    // Warmup runs
    for _ in 0..3 {
        let _ = engine.execute(&content)?;
    }

    println!("Running benchmark...");
    for i in 1..=iterations {
        let result = engine.execute(&content)?;
        
        if let Some(error) = result.error {
            error!("Benchmark iteration {i} failed: {error}");
            std::process::exit(1);
        }

        total_time += Duration::from_millis(result.execution_time_ms);
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
    println!("  Throughput: {:.2} executions/second", 
        iterations as f64 / total_time.as_secs_f64());

    Ok(())
}

fn show_version(detailed: bool) {
    println!("RustMat v{}", env!("CARGO_PKG_VERSION"));
    
    if detailed {
        println!("Built with Rust {}", std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "unknown".to_string()));
        println!("Target: {}", std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string()));
        println!("Profile: {}", if cfg!(debug_assertions) { "debug" } else { "release" });
        println!("Features: jupyter-kernel, plotting, repl, jit, gc");
        println!();
        println!("Components:");
        println!("  • rustmat-lexer: MATLAB/Octave tokenizer");
        println!("  • rustmat-parser: Syntax parser with error recovery");
        println!("  • rustmat-hir: High-level intermediate representation");
        println!("  • rustmat-ignition: Baseline interpreter");
        println!("  • rustmat-turbine: JIT compiler with Cranelift");
        println!("  • rustmat-gc: Generational garbage collector");
        println!("  • rustmat-runtime: BLAS/LAPACK runtime with builtins");
        println!("  • rustmat-kernel: Jupyter kernel protocol");
        println!("  • rustmat-plot: Headless plotting backend");
    }
}

async fn show_system_info(cli: &Cli) -> Result<()> {
    println!("RustMat System Information");
    println!("==========================");
    println!();
    
    println!("Version: {}", env!("CARGO_PKG_VERSION"));
    println!("Rust Version: {}", std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "unknown".to_string()));
    println!("Target: {}", std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string()));
    println!();
    
    println!("Runtime Configuration:");
    println!("  JIT Compiler: {}", if !cli.no_jit { "enabled" } else { "disabled" });
    println!("  JIT Threshold: {}", cli.jit_threshold);
    println!("  JIT Optimization: {:?}", cli.jit_opt_level);
    println!("  GC Preset: {:?}", cli.gc_preset.as_ref().map(|p| format!("{p:?}")).unwrap_or_else(|| "default".to_string()));
    if let Some(young_size) = cli.gc_young_size {
        println!("  GC Young Generation: {young_size}MB");
    }
    if let Some(threads) = cli.gc_threads {
        println!("  GC Threads: {threads}");
    }
    println!("  GC Statistics: {}", cli.gc_stats);
    println!();
    
    println!("Environment:");
    println!("  RUSTMAT_DEBUG: {:?}", std::env::var("RUSTMAT_DEBUG").ok());
    println!("  RUSTMAT_LOG_LEVEL: {:?}", std::env::var("RUSTMAT_LOG_LEVEL").ok());
    println!("  RUSTMAT_TIMEOUT: {:?}", std::env::var("RUSTMAT_TIMEOUT").ok());
    println!("  RUSTMAT_JIT_ENABLE: {:?}", std::env::var("RUSTMAT_JIT_ENABLE").ok());
    println!("  RUSTMAT_GC_PRESET: {:?}", std::env::var("RUSTMAT_GC_PRESET").ok());
    println!();

    // Show GC stats
    let gc_stats = gc_stats();
    println!("Garbage Collector Status:");
    println!("{}", gc_stats.summary_report());
    println!();

    println!("Available Commands:");
    println!("  repl                 Start interactive REPL with JIT");
    println!("  kernel               Start Jupyter kernel");
    println!("  kernel-connection    Start kernel with connection file");
    println!("  run <file>           Execute MATLAB script");
    println!("  gc stats             Show GC statistics");
    println!("  gc major             Force major GC collection");
    println!("  benchmark <file>     Benchmark script execution");
    println!("  version              Show version information");
    println!("  info                 Show this system information");

    Ok(())
}

fn show_repl_help() {
    println!("RustMat REPL Help");
    println!("=================");
    println!();
    println!("Commands:");
    println!("  exit, quit        Exit the REPL");
    println!("  help              Show this help message");
    println!("  .info             Show detailed system information");
    println!("  .stats            Show execution statistics");
    println!("  .gc               Show garbage collector statistics");
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