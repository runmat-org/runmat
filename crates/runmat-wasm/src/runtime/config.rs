use runmat_accelerate::{
    AccelPowerPreference, AccelerateInitOptions, AccelerateProviderPreference, AutoOffloadOptions,
};
use runmat_core::{CompatMode, WorkspaceExportMode};
use runmat_telemetry::TelemetryRunKind;
use serde::Deserialize;
use wasm_bindgen::prelude::JsValue;

use runmat_runtime::builtins::plotting::{set_scatter_target_points, set_surface_vertex_budget};

#[derive(Debug, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub(crate) struct InitOptions {
    #[serde(default)]
    pub(crate) enable_jit: Option<bool>,
    #[serde(default)]
    pub(crate) verbose: Option<bool>,
    #[serde(default)]
    pub(crate) enable_gpu: Option<bool>,
    #[serde(default)]
    pub(crate) snapshot_url: Option<String>,
    #[serde(default)]
    pub(crate) snapshot_bytes: Option<Vec<u8>>,
    #[serde(default)]
    pub(crate) telemetry_consent: Option<bool>,
    #[serde(default)]
    pub(crate) wgpu_power_preference: Option<String>,
    #[serde(default)]
    pub(crate) wgpu_force_fallback_adapter: Option<bool>,
    #[cfg(target_arch = "wasm32")]
    #[serde(skip)]
    pub(crate) snapshot_stream: Option<JsValue>,
    #[serde(default)]
    pub(crate) scatter_target_points: Option<u32>,
    #[serde(default)]
    pub(crate) surface_vertex_budget: Option<u64>,
    #[serde(default)]
    pub(crate) telemetry_id: Option<String>,
    #[serde(default, rename = "telemetryRunKind")]
    pub(crate) telemetry_run_kind: Option<String>,
    #[serde(default)]
    pub(crate) emit_fusion_plan: Option<bool>,
    #[serde(default)]
    pub(crate) language_compat: Option<String>,
    #[serde(default)]
    pub(crate) log_level: Option<String>,
    #[serde(default)]
    pub(crate) gpu_buffer_pool_max_per_key: Option<u32>,
    #[serde(default)]
    pub(crate) callstack_limit: Option<usize>,
    #[serde(default)]
    pub(crate) error_namespace: Option<String>,
    #[cfg(target_arch = "wasm32")]
    #[serde(skip)]
    pub(crate) telemetry_emitter: Option<JsValue>,
}

#[derive(Clone)]
pub(crate) struct SessionConfig {
    pub(crate) enable_jit: bool,
    pub(crate) verbose: bool,
    pub(crate) telemetry_consent: bool,
    pub(crate) telemetry_client_id: Option<String>,
    pub(crate) telemetry_run_kind: TelemetryRunKind,
    pub(crate) enable_gpu: bool,
    pub(crate) wgpu_power_preference: AccelPowerPreference,
    pub(crate) wgpu_force_fallback_adapter: bool,
    pub(crate) auto_offload: AutoOffloadOptions,
    pub(crate) emit_fusion_plan: bool,
    pub(crate) language_compat: CompatMode,
    pub(crate) gpu_buffer_pool_max_per_key: Option<u32>,
    pub(crate) callstack_limit: usize,
    pub(crate) error_namespace: String,
}

impl SessionConfig {
    pub(crate) fn from_options(opts: &InitOptions) -> Self {
        Self {
            enable_jit: opts.enable_jit.unwrap_or(false) && cfg!(feature = "jit"),
            verbose: opts.verbose.unwrap_or(false),
            telemetry_consent: opts.telemetry_consent.unwrap_or(true),
            telemetry_client_id: opts.telemetry_id.clone(),
            telemetry_run_kind: parse_telemetry_run_kind(opts.telemetry_run_kind.as_deref()),
            enable_gpu: opts.enable_gpu.unwrap_or(true),
            wgpu_power_preference: parse_power_preference(opts.wgpu_power_preference.as_deref()),
            wgpu_force_fallback_adapter: opts.wgpu_force_fallback_adapter.unwrap_or(false),
            auto_offload: AutoOffloadOptions::default(),
            emit_fusion_plan: opts.emit_fusion_plan.unwrap_or(false),
            language_compat: parse_language_compat(opts.language_compat.as_deref()),
            gpu_buffer_pool_max_per_key: opts.gpu_buffer_pool_max_per_key,
            callstack_limit: opts
                .callstack_limit
                .unwrap_or(runmat_vm::DEFAULT_CALLSTACK_LIMIT),
            error_namespace: opts
                .error_namespace
                .clone()
                .unwrap_or_else(|| runmat_vm::DEFAULT_ERROR_NAMESPACE.to_string()),
        }
    }

    pub(crate) fn to_accel_options(&self) -> AccelerateInitOptions {
        AccelerateInitOptions {
            enabled: self.enable_gpu,
            provider: if self.enable_gpu {
                AccelerateProviderPreference::Wgpu
            } else {
                AccelerateProviderPreference::InProcess
            },
            allow_inprocess_fallback: true,
            wgpu_power_preference: self.wgpu_power_preference,
            wgpu_force_fallback_adapter: self.wgpu_force_fallback_adapter,
            auto_offload: self.auto_offload.clone(),
        }
    }

    pub(crate) fn apply_env_overrides(&self) {
        if let Some(max) = self.gpu_buffer_pool_max_per_key {
            log::info!("RunMat wasm: setting RUNMAT_WGPU_POOL_MAX_PER_KEY={}", max);
            std::env::set_var("RUNMAT_WGPU_POOL_MAX_PER_KEY", max.to_string());
        }
    }
}

pub(crate) fn apply_plotting_overrides(opts: &InitOptions) {
    if let Some(points) = opts.scatter_target_points {
        set_scatter_target_points(points);
    }
    if let Some(budget) = opts.surface_vertex_budget {
        set_surface_vertex_budget(budget);
    }
}

pub(crate) fn parse_language_compat(input: Option<&str>) -> CompatMode {
    input
        .and_then(parse_language_compat_from_str)
        .unwrap_or(CompatMode::Matlab)
}

pub(crate) fn parse_telemetry_run_kind(value: Option<&str>) -> TelemetryRunKind {
    match value.unwrap_or("repl").trim().to_ascii_lowercase().as_str() {
        "script" => TelemetryRunKind::Script,
        "benchmark" => TelemetryRunKind::Benchmark,
        "install" => TelemetryRunKind::Install,
        _ => TelemetryRunKind::Repl,
    }
}

pub(crate) fn parse_language_compat_from_str(value: &str) -> Option<CompatMode> {
    if value.eq_ignore_ascii_case("strict") {
        Some(CompatMode::Strict)
    } else if value.eq_ignore_ascii_case("matlab") || value.eq_ignore_ascii_case("runmat") {
        Some(CompatMode::Matlab)
    } else {
        None
    }
}

pub(crate) fn parse_workspace_export_mode(value: Option<&str>) -> WorkspaceExportMode {
    match value.unwrap_or("auto").trim().to_ascii_lowercase().as_str() {
        "off" => WorkspaceExportMode::Off,
        "force" => WorkspaceExportMode::Force,
        _ => WorkspaceExportMode::Auto,
    }
}

pub(crate) fn parse_power_preference(input: Option<&str>) -> AccelPowerPreference {
    match input.map(|s| s.to_ascii_lowercase()) {
        Some(ref value) if value.contains("low") => AccelPowerPreference::LowPower,
        Some(ref value) if value.contains("high") => AccelPowerPreference::HighPerformance,
        _ => AccelPowerPreference::Auto,
    }
}
