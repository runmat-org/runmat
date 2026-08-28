use anyhow::{Context, Result};
use log::info;
use std::fs;
use std::path::Path;

use crate::document::{migrate_legacy_desktop_config, RunmatConfigDocument, RunmatConfigFormat};
use crate::runtime::RunMatRuntimeConfig;

/// Load runtime configuration through the canonical RunMat config document.
pub(crate) fn load_from_file(path: &Path) -> Result<RunMatRuntimeConfig> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read config file: {}", path.display()))?;
    let format = RunmatConfigFormat::from_path(path)?;
    let migrated = migrate_legacy_desktop_config(&content, format)
        .with_context(|| format!("Failed to migrate config: {}", path.display()))?;
    let document = RunmatConfigDocument::parse(migrated.source, format)
        .with_context(|| format!("Failed to parse config: {}", path.display()))?;
    Ok(document.runtime().clone())
}

/// Save runtime configuration without replacing package, Desktop, test, or
/// future top-level sections.
pub(crate) fn save_to_file(config: &RunMatRuntimeConfig, path: &Path) -> Result<()> {
    let format = RunmatConfigFormat::from_path(path)?;
    let source = if path.exists() {
        fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?
    } else {
        empty_document(format)
    };
    let migrated = migrate_legacy_desktop_config(&source, format)
        .with_context(|| format!("Failed to migrate config: {}", path.display()))?;
    let document = RunmatConfigDocument::parse(migrated.source, format)
        .with_context(|| format!("Failed to parse config: {}", path.display()))?;
    let updated = document.with_runtime(config)?;
    fs::write(path, updated.source())
        .with_context(|| format!("Failed to write config file: {}", path.display()))?;
    info!("Configuration saved to: {}", path.display());
    Ok(())
}

/// Generate a sample runmat.toml file containing package + runtime sections.
pub(crate) fn generate_sample_config() -> String {
    let sample = r#"[package]
name = "example"
version = "0.1.0"
runmat-version = ">=0.4.0"

[sources]
roots = ["src"]

[dependencies]
utils = { path = "../utils", version = "0.1.0" }

[entrypoints.main]
module = "app"
function = "main"

[runtime]
callstack_limit = 200
error_namespace = "RunMat"
verbose = false

language = { compat = "runmat" }
logging = { level = "warn", debug = false, file = "" }
telemetry = { enabled = true, show_payloads = false, http_endpoint = "", udp_endpoint = "udp.telemetry.runmat.com:7846", queue_size = 256, sync_mode = false, drain_mode = "all", drain_timeout_ms = 50, require_ingestion_key = true }
jit = { enabled = true, threshold = 10, optimization_level = "speed" }
gc = { preset = "low-latency", young_size_mb = 128, threads = 8, collect_stats = false }
accelerate = { enabled = true, provider = "wgpu", allow_inprocess_fallback = true, wgpu_power_preference = "auto", wgpu_force_fallback_adapter = false, auto_offload = { enabled = true, calibrate = true, profile_path = ".runmat/auto_offload.json", log_level = "trace" } }
plotting = { mode = "auto", force_headless = false, backend = "auto", scatter_target_points = 250000, surface_vertex_budget = 400000 }

[runtime.fea]
# artifact_store = "filesystem" # default; use "in_memory" for ephemeral/test runs
# artifact_root = "artifacts"
# study_artifact_root = "artifacts/studies"
# geometry_prep_artifact_root = "artifacts/geometry-prep"
# thermo_field_artifact_root = "artifacts/thermo-fields"
# artifact_max_runs = 0
# artifact_max_runs_per_kind = 0
# geometry_prep_max_artifacts = 0
# geometry_prep_max_artifacts_per_geometry = 0
# geometry_prep_max_age_seconds = 0
geometry_prep_require_latest_revision = true
"#;
    sample.to_string()
}

pub(crate) fn render_runtime_config(config: &RunMatRuntimeConfig, path: &Path) -> Result<String> {
    let format = RunmatConfigFormat::from_path(path)?;
    let document = RunmatConfigDocument::parse(empty_document(format), format)?;
    Ok(document.with_runtime(config)?.into_source())
}

fn empty_document(format: RunmatConfigFormat) -> String {
    match format {
        RunmatConfigFormat::Toml => String::new(),
        RunmatConfigFormat::Json => "{}\n".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::generate_sample_config;

    #[test]
    fn generated_config_has_no_startup_snapshot_setting() {
        assert!(!generate_sample_config().contains("snapshot_path"));
    }
}
