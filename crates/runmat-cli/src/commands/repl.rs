use anyhow::{Context, Result};
use log::info;
use runmat_config::RunMatConfig;
use runmat_core::{RunMatSession, TelemetryHost, TelemetryRunConfig, TelemetryRunFinish};
use runmat_gc::gc_collect_major;
use runmat_time::Instant;
use std::io::{self, Read};

use crate::commands::session::create_session;
use crate::commands::streams::emit_execution_streams;
use crate::diagnostics::format_frontend_error;
use crate::telemetry::{capture_provider_snapshot, RuntimeExecutionCounters, TelemetryRunKind};

pub async fn execute_repl(config: &RunMatConfig) -> Result<()> {
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

    let mut engine = create_session(
        enable_jit,
        config.runtime.verbose,
        config.runtime.snapshot_path.as_ref(),
        config,
        "Failed to create REPL engine",
    )?;
    let repl_run = engine.telemetry_run(TelemetryRunConfig {
        kind: TelemetryRunKind::Repl,
        jit_enabled: config.jit.enabled,
        accelerate_enabled: config.accelerate.enabled,
    });

    info!("RunMat REPL ready");

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
        finalize_repl_session(&engine, session_start, repl_run);
        return Ok(());
    }

    println!("RunMat v{}", env!("CARGO_PKG_VERSION"));
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

    finalize_repl_session(&engine, session_start, repl_run);
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

fn finalize_repl_session(
    engine: &RunMatSession,
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
            failure: None,
            host: Some(TelemetryHost::Cli),
            counters: Some(counters),
            provider: capture_provider_snapshot(),
        });
    }

    info!("RunMat REPL exiting");
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
