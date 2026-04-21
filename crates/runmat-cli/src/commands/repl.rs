use anyhow::{Context, Result};
use log::info;
use owo_colors::OwoColorize;
use runmat_config::{GcPreset, JitOptLevel, RunMatConfig};
use runmat_core::{RunMatSession, TelemetryHost, TelemetryRunConfig, TelemetryRunFinish};
use runmat_gc::gc_collect_major;
use runmat_time::Instant;
use std::io::{self, Read};
use supports_color::Stream;

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

    print_repl_banner(config, &engine);

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

#[derive(Clone, Copy)]
struct BannerCapabilities {
    color: bool,
    truecolor: bool,
}

#[derive(Clone, Copy)]
enum BannerTone {
    Label,
    Brand,
    Bright,
    Muted,
}

fn print_repl_banner(config: &RunMatConfig, engine: &RunMatSession) {
    let caps = detect_banner_capabilities();

    println!(
        "{}",
        style_text(
            &format!("RunMat {}", env!("CARGO_PKG_VERSION")),
            &caps,
            BannerTone::Brand
        )
    );
    println!(
        "{}",
        style_text(
            "MATLAB-compatible runtime for CPU + GPU",
            &caps,
            BannerTone::Bright
        )
    );
    println!(
        "{}",
        style_text("https://runmat.com", &caps, BannerTone::Muted)
    );
    println!();
    println!("{}", format_gpu_line(config, &caps));
    println!("{}", format_runtime_line(config, engine, &caps));
    println!();
    println!("{}", format_help_line(&caps));
    println!();
}

fn detect_banner_capabilities() -> BannerCapabilities {
    let decorated = atty::is(atty::Stream::Stdout)
        && std::env::var("TERM")
            .map(|term| term != "dumb")
            .unwrap_or(true);
    let color_level = if decorated {
        supports_color::on(Stream::Stdout)
    } else {
        None
    };
    BannerCapabilities {
        color: color_level.is_some(),
        truecolor: color_level.map(|level| level.has_16m).unwrap_or(false),
    }
}

fn format_gpu_line(config: &RunMatConfig, caps: &BannerCapabilities) -> String {
    let label = style_text("GPU:", caps, BannerTone::Label);

    if !config.accelerate.enabled {
        return format!(
            "{label} {}",
            style_text("disabled by config", caps, BannerTone::Bright)
        );
    }

    if let Some(provider) = runmat_accelerate_api::provider() {
        let info = provider.device_info_struct();
        let auto_offload = if config.accelerate.auto_offload.enabled {
            style_text("(auto-offload enabled)", caps, BannerTone::Muted)
        } else {
            style_text("(auto-offload disabled)", caps, BannerTone::Muted)
        };

        if matches!(
            info.backend.as_deref(),
            Some(backend) if backend.eq_ignore_ascii_case("inprocess")
        ) || info.name.eq_ignore_ascii_case("InProcess")
        {
            return format!(
                "{} {} {}",
                label,
                style_text("CPU fallback", caps, BannerTone::Bright),
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
            style_text(
                &format!("{} ({backend})", info.name),
                caps,
                BannerTone::Bright
            ),
            auto_offload
        );
    }

    let unavailable = if cfg!(feature = "wgpu") {
        "unavailable"
    } else {
        "unavailable in this build"
    };

    format!(
        "{label} {}",
        style_text(unavailable, caps, BannerTone::Bright)
    )
}

fn format_runtime_line(
    config: &RunMatConfig,
    engine: &RunMatSession,
    caps: &BannerCapabilities,
) -> String {
    let jit_value = if config.jit.enabled {
        style_text(
            jit_opt_level_label(config.jit.optimization_level),
            caps,
            BannerTone::Bright,
        )
    } else {
        style_text("off", caps, BannerTone::Bright)
    };
    let gc_value = style_text(gc_preset_label(config.gc.preset), caps, BannerTone::Bright);
    let snapshot_value = if engine.snapshot_info().is_some() {
        style_text("loaded", caps, BannerTone::Bright)
    } else {
        style_text("none", caps, BannerTone::Bright)
    };

    format!(
        "{} {}\n{} {}\n{} {}",
        style_text("JIT:", caps, BannerTone::Label),
        jit_value,
        style_text("GC:", caps, BannerTone::Label),
        gc_value,
        style_text("Snapshot:", caps, BannerTone::Label),
        snapshot_value
    )
}

fn format_help_line(caps: &BannerCapabilities) -> String {
    let help = style_text("help", caps, BannerTone::Bright);
    let info = style_text(".info", caps, BannerTone::Bright);
    let exit = style_text("exit", caps, BannerTone::Bright);
    [
        style_text("Enter code to execute, or", caps, BannerTone::Muted),
        format!(" `{help}`"),
        style_text(",", caps, BannerTone::Muted),
        format!(" `{exit}`"),
        style_text(" or", caps, BannerTone::Muted),
        format!(" `{info}`"),
        style_text(".", caps, BannerTone::Muted),
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

fn style_text(text: &str, caps: &BannerCapabilities, tone: BannerTone) -> String {
    if !caps.color {
        return text.to_string();
    }

    match tone {
        BannerTone::Label => {
            if caps.truecolor {
                format!("{}", text.truecolor(79, 140, 255).bold())
            } else {
                format!("{}", text.bright_blue().bold())
            }
        }
        BannerTone::Brand => {
            if caps.truecolor {
                format!("{}", text.truecolor(194, 108, 255).bold())
            } else {
                format!("{}", text.bright_magenta().bold())
            }
        }
        BannerTone::Bright => format!("{}", text.bold().bright_white()),
        BannerTone::Muted => format!("{}", text.dimmed()),
    }
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
    println!("RunMat REPL");
    println!();
    println!("Commands");
    println!("  help            Show this help");
    println!("  exit, quit      Exit the REPL");
    println!("  .info           Show runtime information");
    println!("  .stats          Show execution statistics");
    println!("  .gc             Show garbage collector summary");
    println!("  .gc-info        Show garbage collector summary with header");
    println!("  .gc-collect     Force garbage collection");
    println!("  .reset-stats    Reset execution statistics");
    println!();
    println!("Examples");
    println!("  x = 1 + 2");
    println!("  y = [1, 2; 3, 4]");
    println!("  for i = 1:5; disp(i); end");
    println!();
    println!("Use `.info` for runtime details.");
}
