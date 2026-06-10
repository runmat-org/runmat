use anyhow::{anyhow, Context, Result};
use log::info;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

use crate::runtime::{
    AccelerateConfig, FeaConfig, GcConfig, JitConfig, LanguageConfig, LoggingConfig,
    PlottingConfig, RunMatRuntimeConfig, TelemetryConfig,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConfigFormat {
    Toml,
    Json,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct RuntimeFileDocument {
    #[serde(default)]
    runtime: RuntimeFileSection,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct RuntimeFileSection {
    callstack_limit: Option<usize>,
    error_namespace: Option<String>,
    verbose: Option<bool>,
    snapshot_path: Option<PathBuf>,
    language: Option<LanguageConfig>,
    logging: Option<LoggingConfig>,
    telemetry: Option<TelemetryConfig>,
    jit: Option<JitConfig>,
    gc: Option<GcConfig>,
    accelerate: Option<AccelerateConfig>,
    plotting: Option<PlottingConfig>,
    fea: Option<FeaConfig>,
}

impl RuntimeFileSection {
    fn apply_to(self, config: &mut RunMatRuntimeConfig) {
        if let Some(callstack_limit) = self.callstack_limit {
            config.runtime.callstack_limit = callstack_limit;
        }
        if let Some(error_namespace) = self.error_namespace {
            config.runtime.error_namespace = error_namespace;
        }
        if let Some(verbose) = self.verbose {
            config.runtime.verbose = verbose;
        }
        if let Some(snapshot_path) = self.snapshot_path {
            config.runtime.snapshot_path = Some(snapshot_path);
        }
        if let Some(language) = self.language {
            config.language = language;
        }
        if let Some(logging) = self.logging {
            config.logging = logging;
        }
        if let Some(telemetry) = self.telemetry {
            config.telemetry = telemetry;
        }
        if let Some(jit) = self.jit {
            config.jit = jit;
        }
        if let Some(gc) = self.gc {
            config.gc = gc;
        }
        if let Some(accelerate) = self.accelerate {
            config.accelerate = accelerate;
        }
        if let Some(plotting) = self.plotting {
            config.plotting = plotting;
        }
        if let Some(fea) = self.fea {
            config.fea = fea;
        }
    }
}

impl From<&RunMatRuntimeConfig> for RuntimeFileDocument {
    fn from(value: &RunMatRuntimeConfig) -> Self {
        Self {
            runtime: RuntimeFileSection {
                callstack_limit: Some(value.runtime.callstack_limit),
                error_namespace: Some(value.runtime.error_namespace.clone()),
                verbose: Some(value.runtime.verbose),
                snapshot_path: value.runtime.snapshot_path.clone(),
                language: Some(value.language.clone()),
                logging: Some(value.logging.clone()),
                telemetry: Some(value.telemetry.clone()),
                jit: Some(value.jit.clone()),
                gc: Some(value.gc.clone()),
                accelerate: Some(value.accelerate.clone()),
                plotting: Some(value.plotting.clone()),
                fea: Some(value.fea.clone()),
            },
        }
    }
}

/// Load runtime configuration from a canonical runmat.toml/runmat.json file.
pub(crate) fn load_from_file(path: &Path) -> Result<RunMatRuntimeConfig> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read config file: {}", path.display()))?;

    let format = format_from_path(path)?;
    let parsed: RuntimeFileDocument = parse_document(&content, format, path)?;

    let mut config = RunMatRuntimeConfig::default();
    parsed.runtime.apply_to(&mut config);
    Ok(config)
}

/// Save runtime configuration to a canonical runmat.toml/runmat.json file.
pub(crate) fn save_to_file(config: &RunMatRuntimeConfig, path: &Path) -> Result<()> {
    let format = format_from_path(path)?;
    let content = render_runtime_config(config, format)?;

    fs::write(path, content)
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

pub(crate) fn render_runtime_config(
    config: &RunMatRuntimeConfig,
    format: ConfigFormat,
) -> Result<String> {
    let doc = RuntimeFileDocument::from(config);
    match format {
        ConfigFormat::Toml => {
            toml::to_string_pretty(&doc).context("Failed to serialize config to TOML")
        }
        ConfigFormat::Json => {
            serde_json::to_string_pretty(&doc).context("Failed to serialize config to JSON")
        }
    }
}

pub(crate) fn format_from_path(path: &Path) -> Result<ConfigFormat> {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("toml") => Ok(ConfigFormat::Toml),
        Some("json") => Ok(ConfigFormat::Json),
        Some(other) => Err(anyhow!(
            "Unsupported config extension .{other}; expected .toml or .json"
        )),
        None => Err(anyhow!("Config file must have .toml or .json extension")),
    }
}

fn parse_document(content: &str, format: ConfigFormat, path: &Path) -> Result<RuntimeFileDocument> {
    match format {
        ConfigFormat::Toml => toml::from_str(content)
            .with_context(|| format!("Failed to parse TOML config: {}", path.display())),
        ConfigFormat::Json => serde_json::from_str(content)
            .with_context(|| format!("Failed to parse JSON config: {}", path.display())),
    }
}
