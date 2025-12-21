use anyhow::Result;
use log::info;
use runmat_gc::gc_test_context;
use runmat_repl::{commands::CommandResult, ReplEngine};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;

fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();

    // Ensure all runtime builtins are linked (prevents dead-code elimination of inventory registrations)
    runmat_runtime::ensure_builtins_linked();

    // Initialize acceleration provider (prefer WGPU when available)
    runmat_accelerate::initialize_acceleration_provider();

    // Initialize the REPL engine with GC test context for safety
    let mut engine = gc_test_context(ReplEngine::new)?;

    // Check if running in test mode (for stable output in CI/tests)
    let test_mode = std::env::var("RUNMAT_REPL_TEST").is_ok();

    // Initialize rustyline for interactive line editing with history
    let mut rl = DefaultEditor::new()?;

    // Welcome message
    println!("RunMat REPL v{}", env!("CARGO_PKG_VERSION"));
    println!("High-performance MATLAB/Octave language runtime");
    println!("Type 'help' for help, 'exit' to quit, '.info' for system information");
    println!();

    loop {
        // In test mode, print prompt explicitly for reproducibility
        if test_mode {
            print!("runmat> ");
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }

        let readline = rl.readline(if test_mode { "" } else { "runmat> " });
        match readline {
            Ok(line) => {
                let input = line.trim();

                // Add non-empty lines to history
                if !input.is_empty() {
                    let _ = rl.add_history_entry(input);
                }

                // Handle special commands first
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
                        match runmat_gc::gc_collect_major() {
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
                    _ => {
                        // Try to parse as a shell command
                        match runmat_repl::commands::parse_and_execute(input, &mut engine) {
                            CommandResult::Handled(output) => {
                                println!("{output}");
                                continue;
                            }
                            CommandResult::Clear => {
                                engine.clear_variables();
                                println!("Variables cleared");
                                continue;
                            }
                            CommandResult::Exit => {
                                println!("Goodbye!");
                                break;
                            }
                            CommandResult::NotCommand => {
                                // Fall through to expression evaluation
                            }
                        }
                    }
                }

                // Execute the input
                match engine.execute(input) {
                    Ok(result) => {
                        if let Some(error) = result.error {
                            eprintln!("Error: {error}");
                        } else if let Some(value) = result.value {
                            // Display result (format style can be changed with 'format' command)
                            println!("ans = {value}");
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
                        } else if let Some(type_info) = result.type_info {
                            println!("({})", type_info);
                        }
                    }
                    Err(e) => {
                        eprintln!("Execution error: {e}");
                    }
                }
            }
            Err(ReadlineError::Interrupted) => {
                // Ctrl+C: return to prompt, don't exit (per REPL spec)
                // This differs from some other REPL implementations that exit on Ctrl+C.
                // See docs/repl-spec.md section 3.4 for design rationale.
                continue;
            }
            Err(ReadlineError::Eof) => {
                // Ctrl+D: graceful exit
                println!();
                break;
            }
            Err(err) => {
                eprintln!("Error: {:?}", err);
                break;
            }
        }
    }

    info!("RunMat REPL exiting");
    Ok(())
}

fn show_help() {
    println!("RunMat REPL Help");
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
    println!("  format [compact|loose]  Set or show output format");
    println!();
    println!("MATLAB/Octave compatible syntax is supported:");
    println!("  x = 1 + 2                         # Assignment");
    println!("  y = [1, 2, 3]                    # Vectors");
    println!("  z = [1, 2; 3, 4]                 # Matrices");
    println!("  if x > 0; disp('positive'); end  # Control flow");
    println!("  for i = 1:5; disp(i); end        # Loops");
    println!();
    println!("Features:");
    println!("  • First-class symbolic mathematics (sym, diff, int, solve)");
    println!("  • JIT compilation with Cranelift");
    println!("  • Generational garbage collection");
    println!("  • High-performance BLAS/LAPACK operations");
    println!("  • Interpreter fallback for unsupported patterns");
    println!();
    println!("Press Enter after each statement to execute.");
}
