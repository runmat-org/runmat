use anyhow::{Context, Result};
use log::info;
use runmat_config::runtime::{GcPreset, JitOptLevel, RunMatRuntimeConfig};
use runmat_core::{
    abi::{ExecutionRequest, HostExecutionPolicy, RuntimeFlow, SourceInput},
    RunMatSession, TelemetryHost, TelemetryRunConfig, TelemetryRunFinish,
};
use runmat_gc::gc_collect_major;
use runmat_time::Instant;
use std::io::{self, Read, Write};
use std::process::{Command, Stdio};
use std::sync::mpsc;

use crate::cli::Cli;
use crate::commands::package::install_project_for_source;
use crate::commands::session::create_session;
use crate::commands::streams::emit_execution_streams;
use crate::diagnostics::{format_compact_runtime_diagnostic, format_frontend_error};
use crate::presentation::{self, StreamStyles};
use crate::telemetry::{capture_provider_snapshot, RuntimeExecutionCounters, TelemetryRunKind};

pub async fn execute_repl(config: &RunMatRuntimeConfig, cli: &Cli) -> Result<()> {
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
        config,
        "Failed to create REPL engine",
    )?;
    let cwd = std::env::current_dir().context("Failed to resolve current directory")?;
    let _project_lease = install_project_for_source(&mut engine, &cwd, cli).await?;
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
        if !process_repl_input(&buffer, &mut engine, config).await? {
            finalize_repl_session(&engine, session_start, repl_run);
            return Ok(());
        }
        finalize_repl_session(&engine, session_start, repl_run);
        return Ok(());
    }

    print_repl_banner(config);

    loop {
        let prompt = presentation::stdout().brand("runmat> ");
        let readline = rl.readline(&prompt);
        match readline {
            Ok(line) => {
                let _ = rl.add_history_entry(line.as_str());

                if !process_repl_input(&line, &mut engine, config).await? {
                    break;
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("{}", presentation::stdout().muted("CTRL-C"));
                break;
            }
            Err(ReadlineError::Eof) => {
                println!("{}", presentation::stdout().muted("CTRL-D"));
                break;
            }
            Err(err) => {
                println!("{}: {:?}", presentation::stdout().error("Error"), err);
                break;
            }
        }
    }

    finalize_repl_session(&engine, session_start, repl_run);
    Ok(())
}

fn print_repl_banner(config: &RunMatRuntimeConfig) {
    let styles = presentation::stdout();

    println!(
        "{}",
        styles.brand(format!("RunMat {}", env!("CARGO_PKG_VERSION")))
    );
    println!(
        "{}",
        styles.heading("MATLAB-compatible runtime for CPU + GPU")
    );
    println!("{}", styles.muted("https://runmat.com"));
    println!();
    println!("{}", format_gpu_line(config, &styles));
    println!("{}", format_runtime_line(config, &styles));
    println!();
    println!("{}", format_help_line(&styles));
    println!();
}

fn format_gpu_line(config: &RunMatRuntimeConfig, styles: &StreamStyles) -> String {
    let label = styles.label("GPU:");

    if !config.accelerate.enabled {
        return format!("{label} {}", styles.value("disabled by config"));
    }

    if let Some(provider) = runmat_accelerate_api::provider() {
        let info = provider.device_info_struct();
        let auto_offload = if config.accelerate.auto_offload.enabled {
            styles.muted("(auto-offload enabled)")
        } else {
            styles.muted("(auto-offload disabled)")
        };

        if matches!(
            info.backend.as_deref(),
            Some(backend) if backend.eq_ignore_ascii_case("inprocess")
        ) || info.name.eq_ignore_ascii_case("InProcess")
        {
            return format!(
                "{} {} {}",
                label,
                styles.value("CPU fallback"),
                auto_offload
            );
        }

        let backend = info
            .backend
            .as_deref()
            .map(titlecase_backend)
            .unwrap_or("GPU");

        return format!(
            "{} {} {}",
            label,
            styles.value(format!("{} ({backend})", info.name)),
            auto_offload
        );
    }

    let unavailable = if cfg!(feature = "wgpu") {
        "unavailable"
    } else {
        "unavailable in this build"
    };

    format!("{label} {}", styles.value(unavailable))
}

fn format_runtime_line(config: &RunMatRuntimeConfig, styles: &StreamStyles) -> String {
    let jit_value = if config.jit.enabled {
        styles.value(jit_opt_level_label(config.jit.optimization_level))
    } else {
        styles.value("off")
    };
    let gc_value = styles.value(gc_preset_label(config.gc.preset));
    format!(
        "{} {}\n{} {}",
        styles.label("JIT:"),
        jit_value,
        styles.label("GC:"),
        gc_value
    )
}

fn format_help_line(styles: &StreamStyles) -> String {
    let help = styles.help("help");
    let info = styles.help(".info");
    let shell = styles.help("!cmd");
    let exit = styles.help("exit");
    [
        styles.muted("Enter code to execute, or"),
        format!(" `{help}`"),
        styles.muted(","),
        format!(" `{exit}`"),
        styles.muted(" or"),
        format!(" `{info}`"),
        styles.muted("; use"),
        format!(" `{shell}`"),
        styles.muted(" for shell"),
        styles.muted("."),
    ]
    .concat()
}

fn titlecase_backend(value: &str) -> &str {
    match value.to_ascii_lowercase().as_str() {
        "metal" => "Metal",
        "vulkan" => "Vulkan",
        "dx12" => "DX12",
        "dx11" => "DX11",
        "opengl" => "OpenGL",
        "webgpu" => "WebGPU",
        other => {
            if other == "cuda" {
                "CUDA"
            } else {
                value
            }
        }
    }
}

fn jit_opt_level_label(level: JitOptLevel) -> &'static str {
    match level {
        JitOptLevel::None => "none",
        JitOptLevel::Size => "size",
        JitOptLevel::Speed => "speed",
        JitOptLevel::Aggressive => "aggressive",
    }
}

fn gc_preset_label(preset: Option<GcPreset>) -> &'static str {
    match preset {
        Some(GcPreset::LowLatency) => "low-latency",
        Some(GcPreset::HighThroughput) => "high-throughput",
        Some(GcPreset::LowMemory) => "low-memory",
        Some(GcPreset::Debug) => "debug",
        None => "default",
    }
}

async fn process_repl_input(
    input: &str,
    engine: &mut RunMatSession,
    config: &RunMatRuntimeConfig,
) -> Result<bool> {
    if input.is_empty() {
        return process_repl_line("", engine, config).await;
    }

    for raw_line in input.lines() {
        if !process_repl_line(raw_line.trim(), engine, config).await? {
            return Ok(false);
        }
    }

    Ok(true)
}

async fn process_repl_line(
    line: &str,
    engine: &mut RunMatSession,
    config: &RunMatRuntimeConfig,
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
        println!(
            "{}",
            presentation::stdout().heading("Execution Statistics:")
        );
        println!(
            "  Total: {}, JIT: {}, Interpreter: {}",
            stats.total_executions, stats.jit_compiled, stats.interpreter_fallback
        );
        println!("  Average time: {:.2}ms", stats.average_execution_time_ms);
        return Ok(true);
    }
    if line == ".gc-info" {
        let gc_stats = engine.gc_stats();
        println!(
            "{}",
            presentation::stdout().heading("Garbage Collector Statistics:")
        );
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
            Ok(collected) => println!(
                "{}",
                presentation::stdout().success(format!("Collected {collected} objects"))
            ),
            Err(e) => println!(
                "{}: {e}",
                presentation::stdout().error("GC collection failed")
            ),
        }
        return Ok(true);
    }
    if line == ".reset-stats" {
        engine.reset_stats();
        println!("{}", presentation::stdout().success("Statistics reset"));
        return Ok(true);
    }
    if line.trim().is_empty() {
        return Ok(true);
    }
    if let Some(command) = line.trim_start().strip_prefix('!') {
        run_shell_escape(command);
        return Ok(true);
    }

    let request = ExecutionRequest::for_source(
        SourceInput::Text {
            name: "<repl>".to_string(),
            text: line.to_string(),
        },
        crate::diagnostics::parser_compat(config.language.compat),
        HostExecutionPolicy::default(),
        engine.workspace_handle(),
    );
    let response = engine.execute_request(request).await;
    match response.result {
        Ok(outcome) => {
            emit_execution_streams(&outcome.streams);
            for diagnostic in &outcome.diagnostics {
                eprintln!("{}", format_compact_runtime_diagnostic(diagnostic));
            }
            if !matches!(outcome.flow, RuntimeFlow::NoValue)
                && config.runtime.verbose
                && outcome
                    .profiling
                    .as_ref()
                    .is_some_and(|profiling| profiling.total_ms > 10)
            {
                if let Some(profiling) = outcome.profiling {
                    println!(
                        "  {}",
                        presentation::stdout().muted(format!("({}ms)", profiling.total_ms))
                    );
                }
            }
        }
        Err(e) => {
            if let Some(diag) = response.source_context.source_text().and_then(|source| {
                format_frontend_error(&e, response.source_context.source_name(), source)
            }) {
                eprintln!("{diag}");
            } else {
                eprintln!("{}: {e}", presentation::stderr().error("Execution error"));
            }
        }
    }

    Ok(true)
}

fn run_shell_escape(command: &str) {
    let command = command.trim_start();
    if command.is_empty() {
        eprintln!(
            "{}",
            presentation::stderr().warning("Shell command is empty")
        );
        return;
    }

    let mut shell = if cfg!(windows) {
        let mut command_process = Command::new("cmd");
        command_process.args(["/C", command]);
        command_process
    } else {
        let mut command_process = Command::new("sh");
        command_process.args(["-c", command]);
        command_process
    };

    match shell.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn() {
        Ok(mut child) => {
            let stdout = child.stdout.take();
            let stderr = child.stderr.take();
            let (sender, receiver) = mpsc::channel();

            let stdout_thread = stdout.map(|mut stdout| {
                let sender = sender.clone();
                std::thread::spawn(move || -> io::Result<()> {
                    let mut buffer = [0; 8192];
                    loop {
                        let bytes_read = stdout.read(&mut buffer)?;
                        if bytes_read == 0 {
                            break;
                        }
                        if sender.send((true, buffer[..bytes_read].to_vec())).is_err() {
                            break;
                        }
                    }
                    Ok(())
                })
            });
            let stderr_thread = stderr.map(|mut stderr| {
                let sender = sender.clone();
                std::thread::spawn(move || -> io::Result<()> {
                    let mut buffer = [0; 8192];
                    loop {
                        let bytes_read = stderr.read(&mut buffer)?;
                        if bytes_read == 0 {
                            break;
                        }
                        if sender.send((false, buffer[..bytes_read].to_vec())).is_err() {
                            break;
                        }
                    }
                    Ok(())
                })
            });
            drop(sender);

            {
                let mut stdout = io::stdout().lock();
                let mut stderr = io::stderr().lock();
                for (is_stdout, chunk) in receiver {
                    if is_stdout {
                        let _ = stdout.write_all(&chunk);
                        let _ = stdout.flush();
                    } else {
                        let _ = stderr.write_all(&chunk);
                        let _ = stderr.flush();
                    }
                }
            }

            if let Some(stdout_thread) = stdout_thread {
                match stdout_thread.join() {
                    Ok(Ok(())) => {}
                    Ok(Err(err)) => eprintln!(
                        "{}: {err}",
                        presentation::stderr().error("Failed to read shell stdout")
                    ),
                    Err(_) => eprintln!(
                        "{}",
                        presentation::stderr().error("Failed to read shell stdout")
                    ),
                }
            }
            if let Some(stderr_thread) = stderr_thread {
                match stderr_thread.join() {
                    Ok(Ok(())) => {}
                    Ok(Err(err)) => eprintln!(
                        "{}: {err}",
                        presentation::stderr().error("Failed to read shell stderr")
                    ),
                    Err(_) => eprintln!(
                        "{}",
                        presentation::stderr().error("Failed to read shell stderr")
                    ),
                }
            }

            match child.wait() {
                Ok(status) => {
                    if !status.success() {
                        if let Some(code) = status.code() {
                            eprintln!(
                                "{}",
                                presentation::stderr()
                                    .warning(format!("Shell command exited with status {code}"))
                            );
                        } else {
                            eprintln!(
                                "{}",
                                presentation::stderr()
                                    .warning("Shell command terminated without exit status")
                            );
                        }
                    }
                }
                Err(err) => {
                    eprintln!(
                        "{}: {err}",
                        presentation::stderr().error("Failed to wait for shell command")
                    );
                }
            }
        }
        Err(err) => {
            eprintln!(
                "{}: {err}",
                presentation::stderr().error("Failed to execute shell command")
            );
        }
    }
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
    let styles = presentation::stdout();
    println!("{}", styles.brand("RunMat REPL"));
    println!();
    println!("{}", styles.heading("Commands"));
    println!("  {}Show this help", styles.help(format!("{:<16}", "help")));
    println!(
        "  {}Exit the REPL",
        styles.help(format!("{:<16}", "exit, quit"))
    );
    println!(
        "  {}Show runtime information",
        styles.help(format!("{:<16}", ".info"))
    );
    println!(
        "  {}Show execution statistics",
        styles.help(format!("{:<16}", ".stats"))
    );
    println!(
        "  {}Show garbage collector summary",
        styles.help(format!("{:<16}", ".gc"))
    );
    println!(
        "  {}Show garbage collector summary with header",
        styles.help(format!("{:<16}", ".gc-info"))
    );
    println!(
        "  {}Force garbage collection",
        styles.help(format!("{:<16}", ".gc-collect"))
    );
    println!(
        "  {}Reset execution statistics",
        styles.help(format!("{:<16}", ".reset-stats"))
    );
    println!(
        "  {}Run a shell command and print its output",
        styles.help(format!("{:<16}", "!cmd"))
    );
    println!();
    println!("{}", styles.heading("Examples"));
    println!("  {}", styles.muted("x = 1 + 2"));
    println!("  {}", styles.muted("y = [1, 2; 3, 4]"));
    println!("  {}", styles.muted("for i = 1:5; disp(i); end"));
    println!("  {}", styles.muted("!pwd"));
    println!();
    println!("Use `{}` for runtime details.", styles.help(".info"));
}
