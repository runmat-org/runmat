use anyhow::Result;
use log::info;
use rustmat_gc::gc_test_context;
use rustmat_repl::ReplEngine;
use std::io::{self, Write};

fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();

    // Initialize the REPL engine with GC test context for safety
    let mut engine = gc_test_context(ReplEngine::new)?;

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut line = String::new();

    // Welcome message
    println!("RustMat REPL v{}", env!("CARGO_PKG_VERSION"));
    println!("High-performance MATLAB/Octave runtime with JIT compilation");
    println!("Type 'help' for help, 'exit' to quit, '.info' for system information");
    println!();

    loop {
        line.clear();
        print!("rustmat> ");
        if stdout.flush().is_err() {
            break;
        }

        if stdin.read_line(&mut line).unwrap_or(0) == 0 {
            println!(); // EOF - add newline before exit
            break;
        }

        let input = line.trim();

        // Handle special commands
        match input {
            "exit" | "quit" => {
                println!("Goodbye!");
                break;
            }
            "help" => {
                show_help();
                continue;
            }
            ".info" => {
                engine.show_system_info();
                continue;
            }
            ".stats" => {
                let stats = engine.stats();
                println!("Execution Statistics:");
                println!(
                    "  Total: {}, JIT: {}, Interpreter: {}",
                    stats.total_executions, stats.jit_compiled, stats.interpreter_fallback
                );
                println!("  Average time: {:.2}ms", stats.average_execution_time_ms);
                continue;
            }
            ".gc-info" => {
                let gc_stats = engine.gc_stats();
                println!("Garbage Collector Statistics:");
                println!("{}", gc_stats.summary_report());
                continue;
            }
            ".gc-collect" => {
                match rustmat_gc::gc_collect_major() {
                    Ok(collected) => println!("Collected {collected} objects"),
                    Err(e) => println!("GC collection failed: {e}"),
                }
                continue;
            }
            ".reset-stats" => {
                engine.reset_stats();
                println!("Statistics reset");
                continue;
            }
            "" => continue, // Empty line
            _ => {}
        }

        // Execute the input
        match engine.execute(input) {
            Ok(result) => {
                if let Some(error) = result.error {
                    eprintln!("Error: {error}");
                } else if let Some(value) = result.value {
                    println!("ans = {}", value);
                    if result.execution_time_ms > 100 {
                        println!(
                            "  (executed in {}ms{})",
                            result.execution_time_ms,
                            if result.used_jit {
                                " via JIT"
                            } else {
                                " via interpreter"
                            }
                        );
                    }
                } else {
                    // No output (e.g., assignment statements)
                }
            }
            Err(e) => {
                eprintln!("Execution error: {e}");
            }
        }
    }

    info!("RustMat REPL exiting");
    Ok(())
}

fn show_help() {
    println!("RustMat REPL Help");
    println!("=================");
    println!();
    println!("Commands:");
    println!("  exit, quit        Exit the REPL");
    println!("  help              Show this help message");
    println!("  .info             Show detailed system information");
    println!("  .stats            Show execution statistics");
    println!("  .gc-info          Show garbage collector statistics");
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
    println!("  • JIT compilation with Cranelift");
    println!("  • Generational garbage collection");
    println!("  • High-performance BLAS/LAPACK operations");
    println!("  • Interpreter fallback for unsupported patterns");
    println!();
    println!("Press Enter after each statement to execute.");
}
