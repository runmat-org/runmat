use runmat_accelerate::{
    AccelPowerPreference, AccelerateInitOptions, AccelerateProviderPreference, AutoOffloadOptions,
};
use runmat_core::{CompatMode, WorkspaceExportMode};
use runmat_telemetry::TelemetryRunKind;
use serde::Deserialize;
use wasm_bindgen::prelude::JsValue;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub(crate) struct InitOptions {
    #[serde(default)]
    pub(crate) runtime_settings: Option<runmat_config::runtime::HostRuntimeSettings>,
    /// Explicit embedding authority; omitted means the normal browser host.
    #[serde(default)]
    pub(crate) application_settings: Option<ApplicationSettingsMode>,
    #[serde(default)]
    pub(crate) enable_jit: Option<bool>,
    #[serde(default)]
    pub(crate) verbose: Option<bool>,
    #[serde(default)]
    pub(crate) enable_gpu: Option<bool>,
    #[serde(default)]
    pub(crate) telemetry_consent: Option<bool>,
    #[serde(default)]
    pub(crate) wgpu_power_preference: Option<String>,
    #[serde(default)]
    pub(crate) wgpu_force_fallback_adapter: Option<bool>,
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

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub(crate) enum ApplicationSettingsMode {
    IsolatedMemory,
}

#[derive(Clone)]
pub(crate) struct SessionConfig {
    /// Original host layers/overrides, never an already merged project's policy.
    pub(crate) original_options: std::rc::Rc<InitOptions>,
    pub(crate) runtime: runmat_config::runtime::RunMatRuntimeConfig,
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
    pub(crate) fn from_options(opts: &InitOptions) -> Result<Self, String> {
        let mut runtime = opts
            .runtime_settings
            .as_ref()
            .map(|settings| settings.resolve())
            .transpose()
            .map_err(|error| error.to_string())?
            .map(|resolved| resolved.config)
            .unwrap_or_default();
        if let Some(enabled) = opts.enable_jit {
            runtime.jit.enabled = enabled;
        }
        if let Some(enabled) = opts.enable_gpu {
            runtime.accelerate.enabled = enabled;
        }
        if let Some(verbose) = opts.verbose {
            runtime.runtime.verbose = verbose;
        }
        if let Some(consent) = opts.telemetry_consent {
            runtime.telemetry.enabled = consent;
        }
        if let Some(limit) = opts.callstack_limit {
            runtime.runtime.callstack_limit = limit;
        }
        if let Some(namespace) = &opts.error_namespace {
            runtime.runtime.error_namespace = namespace.clone();
        }
        if let Some(compat) = &opts.language_compat {
            runtime.language.compat =
                serde_json::from_value(serde_json::Value::String(compat.clone()))
                    .map_err(|error| error.to_string())?;
        }
        if let Some(level) = &opts.log_level {
            runtime.logging.level =
                serde_json::from_value(serde_json::Value::String(level.clone()))
                    .map_err(|error| error.to_string())?;
        }
        if let Some(preference) = &opts.wgpu_power_preference {
            runtime.accelerate.wgpu_power_preference =
                serde_json::from_value(serde_json::Value::String(preference.clone()))
                    .map_err(|error| error.to_string())?;
        }
        if let Some(fallback) = opts.wgpu_force_fallback_adapter {
            runtime.accelerate.wgpu_force_fallback_adapter = fallback;
        }
        if let Some(points) = opts.scatter_target_points {
            runtime.plotting.scatter_target_points = Some(points);
        }
        if let Some(vertices) = opts.surface_vertex_budget {
            runtime.plotting.surface_vertex_budget = Some(vertices);
        }
        Self::from_runtime_and_options(runtime, opts)
    }

    /// Reuse the already resolved original worker policy without consulting
    /// another user/project layer or the worker's ambient environment.
    pub(crate) fn from_closed_runtime(
        runtime: runmat_config::runtime::RunMatRuntimeConfig,
    ) -> Result<Self, String> {
        Self::from_runtime_and_options(runtime, &InitOptions::default())
    }

    fn from_runtime_and_options(
        runtime: runmat_config::runtime::RunMatRuntimeConfig,
        opts: &InitOptions,
    ) -> Result<Self, String> {
        use runmat_config::runtime::{AccelPowerPreference as Power, LanguageCompatMode};
        #[cfg(target_arch = "wasm32")]
        crate::runtime::settings::validate_browser_capabilities(&runtime)?;
        let language_compat = match runtime.language.compat {
            LanguageCompatMode::RunMat => CompatMode::RunMat,
            LanguageCompatMode::Matlab => CompatMode::Matlab,
            LanguageCompatMode::Strict => CompatMode::Strict,
        };
        let power = match runtime.accelerate.wgpu_power_preference {
            Power::Auto => AccelPowerPreference::Auto,
            Power::LowPower => AccelPowerPreference::LowPower,
            Power::HighPerformance => AccelPowerPreference::HighPerformance,
        };
        let auto = &runtime.accelerate.auto_offload;
        let auto_offload = AutoOffloadOptions {
            enabled: auto.enabled,
            calibrate: auto.calibrate,
            log_level: match auto.log_level {
                runmat_config::runtime::AutoOffloadLogLevel::Off => {
                    runmat_accelerate::AutoOffloadLogLevel::Off
                }
                runmat_config::runtime::AutoOffloadLogLevel::Info => {
                    runmat_accelerate::AutoOffloadLogLevel::Info
                }
                runmat_config::runtime::AutoOffloadLogLevel::Trace => {
                    runmat_accelerate::AutoOffloadLogLevel::Trace
                }
            },
        };
        let error_namespace = if runtime.runtime.error_namespace.trim().is_empty() {
            error_namespace_for_compat(language_compat).to_string()
        } else {
            runtime.runtime.error_namespace.clone()
        };
        Ok(Self {
            original_options: std::rc::Rc::new(opts.clone()),
            enable_jit: runtime.jit.enabled
                && cfg!(feature = "jit")
                && !cfg!(target_arch = "wasm32"),
            verbose: runtime.runtime.verbose,
            telemetry_consent: runtime.telemetry.enabled,
            telemetry_client_id: opts.telemetry_id.clone(),
            telemetry_run_kind: parse_telemetry_run_kind(opts.telemetry_run_kind.as_deref()),
            enable_gpu: runtime.accelerate.enabled
                && runtime.accelerate.provider
                    != runmat_config::runtime::AccelerateProviderPreference::InProcess,
            wgpu_power_preference: power,
            wgpu_force_fallback_adapter: runtime.accelerate.wgpu_force_fallback_adapter,
            auto_offload,
            emit_fusion_plan: opts.emit_fusion_plan.unwrap_or(false),
            language_compat,
            gpu_buffer_pool_max_per_key: opts.gpu_buffer_pool_max_per_key,
            callstack_limit: runtime.runtime.callstack_limit,
            error_namespace,
            runtime,
        })
    }

    pub(crate) fn for_project(
        &self,
        project: runmat_config::runtime::HostConfigDocument,
        user_settings_json: Option<String>,
    ) -> Result<Self, String> {
        let mut options = self.original_options.as_ref().clone();
        let inherited_user = options
            .runtime_settings
            .as_ref()
            .filter(|layers| layers.policy == runmat_config::runtime::ConfigurationPolicy::Local)
            .and_then(|layers| layers.user_settings_json.clone());
        options.runtime_settings = Some(runmat_config::runtime::HostRuntimeSettings {
            user_settings_json: user_settings_json.or(inherited_user),
            project: Some(project),
            policy: runmat_config::runtime::ConfigurationPolicy::Local,
        });
        let mut selected = Self::from_options(&options)?;
        // Preserve the initial user/explicit host layer for the next project;
        // the just-selected manifest must not become inherited input.
        selected.original_options = self.original_options.clone();
        Ok(selected)
    }

    pub(crate) fn to_accel_options(&self) -> AccelerateInitOptions {
        AccelerateInitOptions {
            enabled: self.enable_gpu,
            provider: match self.runtime.accelerate.provider {
                runmat_config::runtime::AccelerateProviderPreference::Auto => {
                    AccelerateProviderPreference::Auto
                }
                runmat_config::runtime::AccelerateProviderPreference::Wgpu => {
                    AccelerateProviderPreference::Wgpu
                }
                runmat_config::runtime::AccelerateProviderPreference::InProcess => {
                    AccelerateProviderPreference::InProcess
                }
            },
            allow_inprocess_fallback: self.runtime.accelerate.allow_inprocess_fallback,
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

pub(crate) fn error_namespace_for_compat(mode: CompatMode) -> &'static str {
    match mode {
        CompatMode::Matlab => "MATLAB",
        CompatMode::RunMat | CompatMode::Strict => "RunMat",
    }
}

pub(crate) fn request_error_namespace(
    compatibility: Option<CompatMode>,
    explicit_namespace: Option<&str>,
) -> Option<String> {
    explicit_namespace
        .map(str::trim)
        .filter(|namespace| !namespace.is_empty())
        .map(str::to_string)
        .or_else(|| {
            compatibility
                .map(error_namespace_for_compat)
                .map(str::to_string)
        })
}

#[cfg(test)]
pub(crate) fn parse_language_compat(input: Option<&str>) -> CompatMode {
    input
        .and_then(parse_language_compat_from_str)
        .unwrap_or(CompatMode::RunMat)
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
    } else if value.eq_ignore_ascii_case("matlab") {
        Some(CompatMode::Matlab)
    } else if value.eq_ignore_ascii_case("runmat") {
        Some(CompatMode::RunMat)
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

#[cfg(test)]
mod compatibility_tests {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn language_compat_defaults_to_runmat() {
        assert_eq!(parse_language_compat(None), CompatMode::RunMat);
    }

    #[wasm_bindgen_test]
    fn application_settings_requires_an_explicit_supported_mode() {
        let normal: InitOptions = serde_json::from_str("{}").unwrap();
        assert_eq!(normal.application_settings, None);
        let isolated: InitOptions =
            serde_json::from_str(r#"{"applicationSettings":"isolatedMemory"}"#).unwrap();
        assert_eq!(
            isolated.application_settings,
            Some(ApplicationSettingsMode::IsolatedMemory)
        );
        assert!(
            serde_json::from_str::<InitOptions>(r#"{"applicationSettings":"unknown"}"#).is_err()
        );
    }

    #[wasm_bindgen_test]
    fn language_compat_modes_remain_distinct() {
        assert_eq!(parse_language_compat(Some("runmat")), CompatMode::RunMat);
        assert_eq!(parse_language_compat(Some("matlab")), CompatMode::Matlab);
        assert_eq!(parse_language_compat(Some("strict")), CompatMode::Strict);
        assert_eq!(parse_language_compat_from_str("unknown"), None);
    }

    #[wasm_bindgen_test]
    fn language_mode_derives_the_default_error_namespace() {
        let matlab = InitOptions {
            language_compat: Some("matlab".to_string()),
            ..Default::default()
        };
        let runmat = InitOptions::default();
        assert_eq!(
            SessionConfig::from_options(&matlab)
                .unwrap()
                .error_namespace,
            "MATLAB"
        );
        assert_eq!(
            SessionConfig::from_options(&runmat)
                .unwrap()
                .error_namespace,
            "RunMat"
        );
    }

    #[wasm_bindgen_test]
    fn named_project_policy_uses_captured_user_not_previous_project_and_keeps_host_override() {
        use runmat_config::document::RunmatConfigFormat;
        use runmat_config::runtime::{HostConfigDocument, HostRuntimeSettings};
        let document = |compat: &str| HostConfigDocument {
            source: format!("[runtime.language]\ncompat = '{compat}'\n"),
            format: RunmatConfigFormat::Toml,
        };
        let initial = SessionConfig::from_options(&InitOptions {
            runtime_settings: Some(HostRuntimeSettings {
                user_settings_json: Some("{\"runtime\":{\"runtime\":{\"callstack_limit\":91},\"language\":{\"compat\":\"strict\"}}}".into()),
                project: Some(document("matlab")), ..Default::default()
            }),
            enable_gpu: Some(false), ..Default::default()
        }).unwrap();
        let next = initial.for_project(document("runmat"), None).unwrap();
        assert_eq!(next.language_compat, CompatMode::RunMat);
        assert_eq!(next.callstack_limit, 91);
        assert!(!next.enable_gpu);
        let next = next
            .for_project(
                document("matlab"),
                Some("{\"runtime\":{\"runtime\":{\"callstack_limit\":37}}}".into()),
            )
            .unwrap();
        assert_eq!(next.language_compat, CompatMode::Matlab);
        assert_eq!(next.callstack_limit, 37);
        let next = next.for_project(document("runmat"), None).unwrap();
        assert_eq!(next.callstack_limit, 91);
        let explicit = SessionConfig::from_options(&InitOptions {
            language_compat: Some("strict".into()),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(
            explicit
                .for_project(document("matlab"), None)
                .unwrap()
                .language_compat,
            CompatMode::Strict
        );
    }

    #[wasm_bindgen_test]
    fn request_namespace_tracks_a_compatibility_override_unless_explicit() {
        assert_eq!(
            request_error_namespace(Some(CompatMode::Matlab), None).as_deref(),
            Some("MATLAB")
        );
        assert_eq!(
            request_error_namespace(Some(CompatMode::Matlab), Some("Acme")).as_deref(),
            Some("Acme")
        );
        assert_eq!(request_error_namespace(None, None), None);
    }
}
