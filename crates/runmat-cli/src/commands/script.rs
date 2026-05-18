use anyhow::{Context, Result};
use log::{info, warn};
use runmat_config::{discover_project_manifest_from, load_project_manifest, RunMatConfig};
use runmat_core::{
    abi::{DiagnosticSeverity, ExecutionOutcome, RuntimeFlow},
    TelemetryHost, TelemetryRunConfig, TelemetryRunFinish,
};
use runmat_time::Instant;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::cli::{CaptureFiguresMode, Cli, FigureSize};
use crate::commands::accel::dump_provider_telemetry_if_requested;
use crate::commands::bytecode::{emit_bytecode, write_bytecode_output};
use crate::commands::session::create_session;
use crate::commands::streams::emit_execution_streams;
use crate::diagnostics::format_frontend_error;
use crate::telemetry::{capture_provider_snapshot, TelemetryRunKind};
use crate::AlreadyReportedCliError;

pub async fn execute_script(
    script: PathBuf,
    emit_bytecode_path: Option<PathBuf>,
    cli: &Cli,
    config: &RunMatConfig,
) -> Result<()> {
    execute_script_with_args(script, vec![], emit_bytecode_path, cli, config).await
}

pub async fn execute_script_with_args(
    script: PathBuf,
    _args: Vec<String>,
    emit_bytecode_path: Option<PathBuf>,
    cli: &Cli,
    config: &RunMatConfig,
) -> Result<()> {
    let script = resolve_script_input(script)?;
    info!("Executing script: {script:?}");

    let content = fs::read_to_string(&script)
        .with_context(|| format!("Failed to read script file: {script:?}"))?;

    execute_script_contents(script, content, emit_bytecode_path, cli, config).await
}

fn resolve_script_input(script: PathBuf) -> Result<PathBuf> {
    if script.exists() {
        return Ok(script);
    }
    let cwd = std::env::current_dir().context("failed to resolve current working directory")?;
    let Some(name) = entrypoint_name_candidate(&script) else {
        return Ok(script);
    };
    let Some(manifest_path) = discover_project_manifest_from(&cwd) else {
        return Ok(script);
    };
    let manifest = load_project_manifest(&manifest_path).with_context(|| {
        format!(
            "failed to load discovered project manifest {}",
            manifest_path.display()
        )
    })?;
    let Some(entrypoint) = manifest.entrypoints.iter().find(|entry| entry.name == name) else {
        return Ok(script);
    };
    if let Some(path) = &entrypoint.path {
        let project_root = manifest_path.parent().unwrap_or_else(|| Path::new("."));
        if let Some(resolved) = resolve_entrypoint_file(project_root, path) {
            info!(
                "Resolved project entrypoint '{}' via {} -> {}",
                name,
                manifest_path.display(),
                resolved.display()
            );
            return Ok(resolved);
        }
        return Err(anyhow::anyhow!(
            "entrypoint '{}' resolved from {} but target path '{}' does not exist",
            name,
            manifest_path.display(),
            path.display()
        ));
    }
    if entrypoint.module.is_some() && entrypoint.function.is_some() {
        return Err(anyhow::anyhow!(
            "entrypoint '{}' in {} targets module/function and is not yet executable from CLI `run`",
            name,
            manifest_path.display()
        ));
    }
    Ok(script)
}

fn entrypoint_name_candidate(script: &Path) -> Option<String> {
    if script.extension().is_some() {
        return None;
    }
    if script.components().count() != 1 {
        return None;
    }
    script
        .file_name()
        .and_then(|name| name.to_str())
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(ToOwned::to_owned)
}

fn resolve_entrypoint_file(project_root: &Path, path: &Path) -> Option<PathBuf> {
    let direct = project_root.join(path);
    if direct.is_file() {
        return Some(direct);
    }
    if direct.extension().is_none() {
        let with_ext = direct.with_extension("m");
        if with_ext.is_file() {
            return Some(with_ext);
        }
    }
    None
}

pub(crate) async fn execute_script_contents(
    script: PathBuf,
    content: String,
    emit_bytecode_path: Option<PathBuf>,
    cli: &Cli,
    config: &RunMatConfig,
) -> Result<()> {
    info!("Executing script source: {script:?}");

    if let Some(path) = &emit_bytecode_path {
        let output = emit_bytecode(&content, config)
            .with_context(|| format!("Failed to emit bytecode for {script:?}"))?;
        write_bytecode_output(path, &output)?;
        return Ok(());
    }

    let enable_jit = config.jit.enabled;
    let mut engine = create_session(
        enable_jit,
        config.runtime.verbose,
        config.runtime.snapshot_path.as_ref(),
        config,
        "Failed to create execution engine",
    )?;
    engine.set_source_name_override(Some(script.to_string_lossy().to_string()));
    let mut script_run = engine.telemetry_run(TelemetryRunConfig {
        kind: TelemetryRunKind::Script,
        jit_enabled: config.jit.enabled,
        accelerate_enabled: config.accelerate.enabled,
    });
    let start_time = Instant::now();
    let outcome = match engine.execute_outcome(&content).await {
        Ok(outcome) => outcome,
        Err(err) => {
            let failure = err.telemetry_failure_info();
            if let Some(run) = script_run.take() {
                run.finish(TelemetryRunFinish {
                    duration: Some(start_time.elapsed()),
                    success: false,
                    jit_used: false,
                    error: Some(failure.code.clone()),
                    failure: Some(failure),
                    host: Some(TelemetryHost::Cli),
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
            return Err(AlreadyReportedCliError.into());
        }
    };

    let execution_time = start_time.elapsed();
    emit_execution_streams(&outcome.streams);

    let provider_snapshot = capture_provider_snapshot();
    let error_payload = outcome_error_code(&outcome);
    let success = error_payload.is_none();

    if let Some(artifacts_plan) = ScriptArtifactsPlan::from_cli(cli)? {
        if let Err(err) = write_script_artifacts(
            &artifacts_plan,
            &script,
            &outcome,
            execution_time,
            success,
            error_payload.as_deref(),
        )
        .await
        {
            warn!("Failed to write run artifacts: {err}");
            eprintln!("Warning: failed to write run artifacts: {err}");
        }
    }

    if let Some(run) = script_run.take() {
        run.finish(TelemetryRunFinish {
            duration: Some(execution_time),
            success,
            jit_used: outcome.used_jit,
            error: error_payload.clone(),
            failure: None,
            host: Some(TelemetryHost::Cli),
            counters: None,
            provider: provider_snapshot,
        });
    }

    if let Some(error) = error_payload {
        eprintln!("{error}");
        return Err(AlreadyReportedCliError.into());
    } else {
        if outcome.used_jit {
            info!("Script executed successfully in {:?} (JIT)", execution_time);
        } else {
            info!("Script executed successfully in {:?}", execution_time);
        }
        if let RuntimeFlow::Single(value) = outcome.flow {
            if config.runtime.verbose {
                println!("{value:?}");
            }
        }
    }

    engine.set_source_name_override(None);
    dump_provider_telemetry_if_requested();

    Ok(())
}

#[derive(Clone, Debug)]
struct ScriptArtifactsPlan {
    artifacts_dir: PathBuf,
    manifest_path: PathBuf,
    capture_figures: CaptureFiguresMode,
    figure_size: FigureSize,
    max_figures: usize,
}

impl ScriptArtifactsPlan {
    fn from_cli(cli: &Cli) -> Result<Option<Self>> {
        let artifacts_dir = match (&cli.artifacts_dir, &cli.artifacts_manifest) {
            (Some(dir), _) => Some(dir.clone()),
            (None, Some(manifest)) => manifest.parent().map(normalize_manifest_parent),
            (None, None) => None,
        };

        let Some(artifacts_dir) = artifacts_dir else {
            return Ok(None);
        };

        let manifest_path = cli
            .artifacts_manifest
            .clone()
            .unwrap_or_else(|| artifacts_dir.join("run_manifest.json"));

        Ok(Some(Self {
            artifacts_dir,
            manifest_path,
            capture_figures: cli.capture_figures,
            figure_size: cli.figure_size.clone(),
            max_figures: cli.max_figures,
        }))
    }
}

fn normalize_manifest_parent(parent: &Path) -> PathBuf {
    if parent.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        parent.to_path_buf()
    }
}

async fn write_script_artifacts(
    plan: &ScriptArtifactsPlan,
    script: &Path,
    outcome: &ExecutionOutcome,
    execution_time: Duration,
    success: bool,
    error_identifier: Option<&str>,
) -> Result<()> {
    fs::create_dir_all(&plan.artifacts_dir).with_context(|| {
        format!(
            "Failed to create artifacts directory {}",
            plan.artifacts_dir.display()
        )
    })?;

    let figure_exports = export_touched_figures(plan, &outcome.figures_touched).await;

    let mut stdout_bytes: usize = 0;
    let mut stderr_bytes: usize = 0;
    for stream in &outcome.streams {
        match stream.stream {
            runmat_core::ExecutionStreamKind::Stdout => stdout_bytes += stream.text.len(),
            runmat_core::ExecutionStreamKind::Stderr => stderr_bytes += stream.text.len(),
            runmat_core::ExecutionStreamKind::ClearScreen => {}
        }
    }

    let manifest = serde_json::json!({
        "schema_version": "runmat.artifacts.v1",
        "script": script.to_string_lossy(),
        "success": success,
        "execution_time_ms": execution_time.as_millis() as u64,
        "used_jit": outcome.used_jit,
        "error_identifier": error_identifier,
        "figures_touched": outcome.figures_touched,
        "figure_exports": figure_exports,
        "stream_summary": {
            "entry_count": outcome.streams.len(),
            "stdout_bytes": stdout_bytes,
            "stderr_bytes": stderr_bytes,
        },
        "capture": {
            "capture_figures": format!("{:?}", plan.capture_figures).to_lowercase(),
            "figure_width": plan.figure_size.width,
            "figure_height": plan.figure_size.height,
            "max_figures": plan.max_figures,
        }
    });

    if let Some(parent) = plan.manifest_path.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "Failed to create manifest parent directory {}",
                parent.display()
            )
        })?;
    }
    fs::write(
        &plan.manifest_path,
        serde_json::to_string_pretty(&manifest)
            .context("Failed to serialize run artifacts manifest")?,
    )
    .with_context(|| {
        format!(
            "Failed to write artifacts manifest {}",
            plan.manifest_path.display()
        )
    })?;

    info!(
        "Wrote run artifacts manifest: {}",
        plan.manifest_path.display()
    );
    Ok(())
}

async fn export_touched_figures(
    plan: &ScriptArtifactsPlan,
    figures_touched: &[u32],
) -> Vec<serde_json::Value> {
    use runmat_runtime::builtins::plotting::{render_figure_snapshot, FigureHandle};

    let capture_enabled = match plan.capture_figures {
        CaptureFiguresMode::Off => false,
        CaptureFiguresMode::Auto => !figures_touched.is_empty(),
        CaptureFiguresMode::On => true,
    };
    if !capture_enabled {
        return Vec::new();
    }

    let figures_dir = plan.artifacts_dir.join("figures");
    if let Err(err) = fs::create_dir_all(&figures_dir) {
        warn!(
            "Failed to create figures artifact directory {}: {}",
            figures_dir.display(),
            err
        );
        return Vec::new();
    }

    let mut exports = Vec::new();

    for (index, handle_raw) in figures_touched.iter().enumerate().take(plan.max_figures) {
        let handle = FigureHandle::from(*handle_raw);
        match render_figure_snapshot(
            handle,
            plan.figure_size.width,
            plan.figure_size.height,
            None,
        )
        .await
        {
            Ok(bytes) => {
                let file_name = format!("figure_{:03}_h{}.png", index + 1, handle_raw);
                let file_path = figures_dir.join(file_name);
                if let Err(err) = fs::write(&file_path, bytes.as_slice()) {
                    exports.push(serde_json::json!({
                        "handle": handle_raw,
                        "ok": false,
                        "error": format!("failed_to_write: {}", err),
                    }));
                    continue;
                }
                exports.push(serde_json::json!({
                    "handle": handle_raw,
                    "ok": true,
                    "path": file_path.to_string_lossy(),
                    "format": "png",
                    "width": plan.figure_size.width,
                    "height": plan.figure_size.height,
                }));
            }
            Err(err) => {
                exports.push(serde_json::json!({
                    "handle": handle_raw,
                    "ok": false,
                    "error": err.to_string(),
                }));
            }
        }
    }

    exports
}

#[cfg(test)]
mod tests {
    use super::resolve_script_input;
    use once_cell::sync::Lazy;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::Mutex;

    static CWD_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

    #[test]
    fn resolves_named_entrypoint_to_manifest_path_target() {
        let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
        let tmp = tempfile::TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(tmp.path().join("src/main.m"), "x = 1;").unwrap();
        fs::write(
            tmp.path().join("runmat.toml"),
            r#"
[package]
name = "demo"

[sources]
roots = ["src"]

[[entrypoints]]
name = "main"
path = "src/main"
"#,
        )
        .unwrap();

        let original = std::env::current_dir().unwrap();
        std::env::set_current_dir(tmp.path()).unwrap();
        let resolved = resolve_script_input(PathBuf::from("main")).expect("resolve entrypoint");
        std::env::set_current_dir(original).unwrap();

        assert_eq!(
            resolved.canonicalize().unwrap(),
            tmp.path().join("src/main.m").canonicalize().unwrap()
        );
    }

    #[test]
    fn rejects_module_function_entrypoint_for_cli_run_path() {
        let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
        let tmp = tempfile::TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(
            tmp.path().join("runmat.toml"),
            r#"
[package]
name = "demo"

[sources]
roots = ["src"]

[[entrypoints]]
name = "server"
module = "app.server"
function = "main"
"#,
        )
        .unwrap();

        let original = std::env::current_dir().unwrap();
        std::env::set_current_dir(tmp.path()).unwrap();
        let err = resolve_script_input(PathBuf::from("server"))
            .expect_err("module/function entrypoint is not a script path");
        std::env::set_current_dir(original).unwrap();

        assert!(err
            .to_string()
            .contains("not yet executable from CLI `run`"));
    }
}

fn outcome_error_code(outcome: &ExecutionOutcome) -> Option<String> {
    outcome
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.severity == DiagnosticSeverity::Error)
        .map(|diagnostic| diagnostic.code.clone())
}
