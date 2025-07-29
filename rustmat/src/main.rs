//! RustMat - High-performance MATLAB/Octave runtime
//! 
//! A modern, V8-inspired MATLAB runtime with Jupyter kernel support,
//! fast startup, and excellent developer ergonomics.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use env_logger::Env;
use log::{info, error};
use rustmat_kernel::{KernelConfig, KernelServer, ConnectionInfo};
use std::path::PathBuf;
use std::fs;
use uuid;

/// RustMat - High-performance MATLAB/Octave runtime
#[derive(Parser)]
#[command(
    name = "rustmat",
    version = "0.0.1",
    about = "High-performance MATLAB/Octave runtime with Jupyter kernel support",
    long_about = r#"
RustMat is a modern, high-performance runtime for MATLAB/Octave code built in Rust.
It features a tiered execution model inspired by V8, with a baseline interpreter
feeding an optimizing JIT compiler.

Key features:
• Fast startup with snapshotting
• Jupyter kernel protocol support
• Compatible with MATLAB/Octave syntax
• World-class error messages and debugging
• Modern tooling and extensibility

Examples:
  rustmat                                    # Start interactive REPL
  rustmat script.m                          # Execute MATLAB script
  rustmat --kernel                          # Start Jupyter kernel
  rustmat --kernel-connection connection.json # Start with connection file
  rustmat --version                         # Show version information
"#,
    after_help = r#"
Environment Variables:
  RUSTMAT_DEBUG=1              Enable debug logging
  RUSTMAT_LOG_LEVEL=debug      Set log level (error, warn, info, debug, trace)
  RUSTMAT_KERNEL_IP=127.0.0.1  Kernel IP address  
  RUSTMAT_KERNEL_KEY=<key>     Kernel authentication key
  RUSTMAT_TIMEOUT=300          Execution timeout in seconds
  RUSTMAT_CONFIG=<path>        Path to configuration file

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

    /// Command to execute
    #[command(subcommand)]
    command: Option<Commands>,

    /// MATLAB script file to execute (alternative to subcommands)
    script: Option<PathBuf>,
}

#[derive(Subcommand)]
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
}

#[derive(Clone, ValueEnum)]
enum LogLevel {
    Error,
    Warn,
    Info, 
    Debug,
    Trace,
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

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let log_level = if cli.debug {
        log::LevelFilter::Debug
    } else {
        cli.log_level.into()
    };

    env_logger::Builder::from_env(Env::default().default_filter_or("info"))
        .filter_level(log_level)
        .init();

    info!("RustMat v{} starting", env!("CARGO_PKG_VERSION"));

    // Handle command or script execution
    match (cli.command, cli.script) {
        (Some(command), None) => {
            execute_command(command, cli.timeout).await
        }
        (None, Some(script)) => {
            execute_script(script, cli.timeout).await
        }
        (None, None) => {
            // Default to REPL
            execute_repl(false).await
        }
        (Some(_), Some(_)) => {
            error!("Cannot specify both command and script file");
            std::process::exit(1);
        }
    }
}

async fn execute_command(command: Commands, timeout: u64) -> Result<()> {
    match command {
        Commands::Repl { verbose } => {
            execute_repl(verbose).await
        }
        Commands::Kernel { 
            ip, key, transport, signature_scheme,
            shell_port, iopub_port, stdin_port, control_port, hb_port,
            connection_file
        } => {
            execute_kernel(
                ip, key, transport, signature_scheme,
                shell_port, iopub_port, stdin_port, control_port, hb_port,
                connection_file, timeout
            ).await
        }
        Commands::KernelConnection { connection_file } => {
            execute_kernel_with_connection(connection_file, timeout).await
        }
        Commands::Run { file, args } => {
            execute_script_with_args(file, args, timeout).await
        }
        Commands::Version { detailed } => {
            show_version(detailed);
            Ok(())
        }
        Commands::Info => {
            show_system_info().await
        }
    }
}

async fn execute_repl(verbose: bool) -> Result<()> {
    info!("Starting RustMat REPL");
    if verbose {
        info!("Verbose mode enabled");
    }

    // For now, use the existing REPL from rustmat-repl
    // TODO: Integrate with new execution engine
    println!("RustMat Interactive Console v{}", env!("CARGO_PKG_VERSION"));
    println!("Type 'exit' or 'quit' to exit, 'help' for help.");
    println!();

    let mut input = String::new();
    loop {
        print!(">> ");
        use std::io::{self, Write};
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
                if line.is_empty() {
                    continue;
                }

                // Execute the input using the execution engine
                let mut engine = rustmat_kernel::ExecutionEngine::new();
                match engine.execute(line) {
                    Ok(result) => {
                        match result.status {
                            rustmat_kernel::execution::ExecutionStatus::Success => {
                                if let Some(value) = result.result {
                                    println!("ans = {:?}", value);
                                }
                            }
                            rustmat_kernel::execution::ExecutionStatus::Error => {
                                if let Some(error) = result.error {
                                    eprintln!("Error: {}", error.message);
                                    if verbose {
                                        for trace in error.traceback {
                                            eprintln!("  {}", trace);
                                        }
                                    }
                                }
                            }
                            _ => {
                                eprintln!("Execution interrupted or timed out");
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Internal error: {}", e);
                    }
                }
            }
            Err(e) => {
                eprintln!("Error reading input: {}", e);
                break;
            }
        }
    }

    info!("RustMat REPL exiting");
    Ok(())
}

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
            .with_context(|| format!("Failed to write connection file to {:?}", path))?;
        info!("Connection file written to {:?}", path);
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
    info!("Starting kernel with connection file: {:?}", connection_file);

    let connection = ConnectionInfo::from_file(&connection_file)
        .with_context(|| format!("Failed to load connection file: {:?}", connection_file))?;

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

async fn execute_script(script: PathBuf, timeout: u64) -> Result<()> {
    execute_script_with_args(script, vec![], timeout).await
}

async fn execute_script_with_args(script: PathBuf, _args: Vec<String>, _timeout: u64) -> Result<()> {
    info!("Executing script: {:?}", script);

    let content = fs::read_to_string(&script)
        .with_context(|| format!("Failed to read script file: {:?}", script))?;

    let mut engine = rustmat_kernel::ExecutionEngine::new();
    let result = engine.execute(&content)
        .context("Failed to execute script")?;

    match result.status {
        rustmat_kernel::execution::ExecutionStatus::Success => {
            info!("Script executed successfully in {}ms", result.execution_time_ms);
            if let Some(value) = result.result {
                println!("{:?}", value);
            }
        }
        rustmat_kernel::execution::ExecutionStatus::Error => {
            if let Some(error) = result.error {
                error!("Script execution failed: {}", error.message);
                for trace in error.traceback {
                    eprintln!("  {}", trace);
                }
                std::process::exit(1);
            }
        }
        _ => {
            error!("Script execution was interrupted or timed out");
            std::process::exit(1);
        }
    }

    Ok(())
}

fn show_version(detailed: bool) {
    println!("RustMat v{}", env!("CARGO_PKG_VERSION"));
    
    if detailed {
        println!("Built with Rust {}", std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "unknown".to_string()));
        println!("Target: {}", std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string()));
        println!("Profile: {}", if cfg!(debug_assertions) { "debug" } else { "release" });
        println!("Features: jupyter-kernel, plotting, repl");
        println!();
        println!("Components:");
        println!("  • rustmat-lexer: MATLAB/Octave tokenizer");
        println!("  • rustmat-parser: Syntax parser with error recovery");
        println!("  • rustmat-hir: High-level intermediate representation");
        println!("  • rustmat-ignition: Baseline interpreter");
        println!("  • rustmat-kernel: Jupyter kernel protocol");
        println!("  • rustmat-plot: Headless plotting backend");
        println!("  • rustmat-runtime: Built-in functions and runtime");
    }
}

async fn show_system_info() -> Result<()> {
    println!("RustMat System Information");
    println!("==========================");
    println!();
    
    println!("Version: {}", env!("CARGO_PKG_VERSION"));
    println!("Rust Version: {}", std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "unknown".to_string()));
    println!("Target: {}", std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string()));
    println!();
    
    println!("Environment:");
    println!("  RUSTMAT_DEBUG: {:?}", std::env::var("RUSTMAT_DEBUG").ok());
    println!("  RUSTMAT_LOG_LEVEL: {:?}", std::env::var("RUSTMAT_LOG_LEVEL").ok());
    println!("  RUSTMAT_TIMEOUT: {:?}", std::env::var("RUSTMAT_TIMEOUT").ok());
    println!();

    // Show execution engine stats
    let engine = rustmat_kernel::ExecutionEngine::new();
    let stats = engine.stats();
    println!("Execution Engine:");
    println!("  Timeout: {:?} seconds", stats.timeout_seconds);
    println!("  Debug: {}", stats.debug_enabled);
    println!();

    println!("Available Commands:");
    println!("  repl                 Start interactive REPL");
    println!("  kernel               Start Jupyter kernel");
    println!("  kernel-connection    Start kernel with connection file");
    println!("  run <file>           Execute MATLAB script");
    println!("  version              Show version information");
    println!("  info                 Show this system information");

    Ok(())
}

fn show_repl_help() {
    println!("RustMat REPL Help");
    println!("=================");
    println!();
    println!("Commands:");
    println!("  exit, quit    Exit the REPL");
    println!("  help          Show this help message");
    println!();
    println!("MATLAB/Octave syntax is supported:");
    println!("  x = 1 + 2                    # Assignment");
    println!("  y = [1, 2, 3]               # Vectors");
    println!("  z = [1, 2; 3, 4]            # Matrices");
    println!("  if x > 0; disp('positive'); end  # Control flow");
    println!("  for i = 1:5; disp(i); end   # Loops");
    println!();
    println!("Press Enter after each statement to execute.");
} 