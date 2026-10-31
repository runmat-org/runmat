use std::sync::Arc;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use runmat_core::{TelemetryPlatformInfo, TelemetrySink};
use runmat_runtime::builtins::wasm_registry;

use crate::api::plot::ensure_figure_event_bridge;
use crate::api::session::RunMatWasm;
use crate::runtime::config::{ApplicationSettingsMode, InitOptions, SessionConfig};
use crate::runtime::execution::{execution_host_from_options, BrowserExecutionService};
use crate::runtime::filesystem::install_js_fs_provider;
use crate::runtime::gpu::{
    capture_gpu_adapter_info, initialize_gpu_provider, install_cpu_provider,
    validate_webgpu_runtime, GpuStatus,
};
use crate::runtime::logging::{init_logging_once, set_log_filter_override};
use crate::runtime::package_cache::JsPackageCacheBackend;
use crate::wire::errors::{init_error, init_error_with_details, js_value_to_string, InitErrorCode};

/// Exact SDK admission marker for string/null typed string projections.
#[wasm_bindgen(js_name = stringElementWireSchemaVersion)]
pub fn string_element_wire_schema_version() -> u16 {
    runmat_execution::schema::STRING_ELEMENT_WIRE_SCHEMA_VERSION
}

#[wasm_bindgen(js_name = initRunMat)]
pub async fn init_runmat(options: JsValue) -> Result<RunMatWasm, JsValue> {
    let mut parsed_opts: InitOptions = if options.is_null() || options.is_undefined() {
        InitOptions::default()
    } else {
        serde_wasm_bindgen::from_value(options.clone()).map_err(|err| {
            init_error(
                InitErrorCode::InvalidOptions,
                format!("Invalid init options: {err}"),
            )
        })?
    };
    let config = SessionConfig::from_options(&parsed_opts)
        .map_err(|error| init_error(InitErrorCode::InvalidOptions, error))?;
    #[cfg(target_arch = "wasm32")]
    crate::runtime::gpu::admit_browser_provider(&config)
        .map_err(|error| init_error(InitErrorCode::InvalidOptions, error))?;
    #[cfg(target_arch = "wasm32")]
    ensure_getrandom_js();
    crate::api::observability::configure_for_session(&config.runtime.logging.otel, &options)?;
    let effective_log_level = if config.runtime.logging.debug {
        "debug".to_owned()
    } else {
        format!("{:?}", config.runtime.logging.level).to_ascii_lowercase()
    };
    set_log_filter_override(&effective_log_level);
    init_logging_once();
    crate::runtime::settings::apply_resources(&config.runtime)
        .map_err(|error| init_error(InitErrorCode::InvalidOptions, error))?;
    #[cfg(target_arch = "wasm32")]
    ensure_figure_event_bridge();
    let dialog_host = install_fs_provider_from_options(&options).map_err(|err| {
        init_error_with_details(
            InitErrorCode::FilesystemProvider,
            "Failed to install filesystem provider",
            Some(err),
        )
    })?;
    let package_cache = package_cache_from_options(&options).await.map_err(|err| {
        init_error_with_details(
            InitErrorCode::PackageCacheProvider,
            "Failed to initialize package cache provider",
            Some(err),
        )
    })?;
    #[cfg(target_arch = "wasm32")]
    {
        if !options.is_null() && !options.is_undefined() {
            if let Ok(emitter_value) =
                js_sys::Reflect::get(&options, &JsValue::from_str("telemetryEmitter"))
            {
                if !emitter_value.is_null() && !emitter_value.is_undefined() {
                    parsed_opts.telemetry_emitter = Some(emitter_value);
                }
            }
        }
    }

    wasm_registry::register_all();
    log::debug!("RunMat wasm: creating execution session");

    config.apply_env_overrides();
    let mut session = runmat_core::RunMatSession::with_options(config.enable_jit, config.verbose)
        .map_err(|err| {
        init_error(
            InitErrorCode::SessionCreation,
            format!("Failed to initialize RunMat session: {err}"),
        )
    })?;
    session
        .apply_runtime_config(&config.runtime)
        .map_err(|error| init_error(InitErrorCode::InvalidOptions, error.to_string()))?;
    if let Some(host) = dialog_host.clone() {
        session.install_host_service(host);
    }
    let application_settings = matches!(
        parsed_opts.application_settings,
        Some(ApplicationSettingsMode::IsolatedMemory)
    )
    .then(|| std::rc::Rc::new(runmat_runtime::context::IsolatedMemoryPreferenceService::new()));
    session.set_telemetry_consent(config.telemetry_consent);
    if let Some(cid) = config.telemetry_client_id.clone() {
        session.set_telemetry_client_id(Some(cid));
    }
    session.set_emit_fusion_plan(config.emit_fusion_plan);
    session.set_compat_mode(config.language_compat);
    session.set_callstack_limit(config.callstack_limit);
    session.set_error_namespace(config.error_namespace.clone());
    let mut gpu_status = GpuStatus {
        requested: config.enable_gpu,
        active: false,
        error: None,
        adapter: None,
    };

    if config.enable_gpu {
        match validate_webgpu_runtime() {
            Ok(()) => match initialize_gpu_provider(&config).await {
                Ok(_) => {
                    gpu_status.active = true;
                    gpu_status.adapter = capture_gpu_adapter_info();
                    #[cfg(target_arch = "wasm32")]
                    {
                        if let Err(err) =
                            runmat_runtime::builtins::plotting::context::ensure_context_from_provider(
                            )
                        {
                            let message = err.message().to_string();
                            log::warn!(
                                "RunMat wasm: unable to install shared plotting context: {message}"
                            );
                            gpu_status.error = Some(message);
                        }
                    }
                }
                Err(err) => {
                    if !config.runtime.accelerate.allow_inprocess_fallback {
                        return Err(init_error_with_details(
                            InitErrorCode::SessionCreation,
                            "GPU initialization failed and CPU fallback is disabled",
                            Some(err),
                        ));
                    }
                    let message = js_value_to_string(err.clone());
                    log::warn!(
                        "RunMat wasm: GPU initialization failed (falling back to CPU): {message}"
                    );
                    gpu_status.error = Some(message);
                    install_cpu_provider(&config)?;
                }
            },
            Err(reason) => {
                if !config.runtime.accelerate.allow_inprocess_fallback {
                    return Err(init_error(
                        InitErrorCode::SessionCreation,
                        format!("WebGPU unavailable and CPU fallback is disabled: {reason}"),
                    ));
                }
                log::warn!(
                    "RunMat wasm: WebGPU capability preflight failed (falling back to CPU): {reason}"
                );
                gpu_status.error = Some(reason);
                install_cpu_provider(&config)?;
            }
        }
    } else {
        install_cpu_provider(&config)?;
    }

    // Build the browser scheduler after provider initialization so its typed
    // worker inventory reflects the accelerator that can actually execute
    // this session's programs.
    let execution_service =
        BrowserExecutionService::new(execution_host_from_options(&options).map_err(|error| {
            init_error_with_details(
                InitErrorCode::InvalidOptions,
                "Failed to initialize browser execution host",
                Some(error),
            )
        })?)
        .map_err(|error| {
            init_error(
                InitErrorCode::SessionCreation,
                format!("Failed to initialize browser execution: {error}"),
            )
        })?;
    session.install_execution_services(execution_service.clone());
    if execution_service.has_owned_cpu_worker() {
        session.install_owned_builtin_work_service(execution_service.clone());
    }

    let telemetry_sink: Option<Arc<dyn TelemetrySink>> = {
        #[cfg(target_arch = "wasm32")]
        {
            if config.telemetry_consent {
                if let Some(callback) = parsed_opts
                    .telemetry_emitter
                    .as_ref()
                    .and_then(|value| value.clone().dyn_into::<js_sys::Function>().ok())
                {
                    Some(Arc::new(WasmTelemetrySink { callback }) as Arc<dyn TelemetrySink>)
                } else {
                    None
                }
            } else {
                None
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            None
        }
    };

    if telemetry_sink.is_some() {
        session.set_telemetry_platform_info(TelemetryPlatformInfo {
            os: Some("web".to_string()),
            arch: Some("wasm32".to_string()),
        });
        session.set_telemetry_sink(telemetry_sink.clone());
    }

    let handle = RunMatWasm::new(
        session,
        config,
        gpu_status,
        telemetry_sink,
        package_cache,
        execution_service,
        dialog_host,
    );
    match application_settings {
        Some(settings) => handle
            .with_application_settings(settings)
            .map_err(|err| init_error(InitErrorCode::SessionCreation, err)),
        None => Ok(handle),
    }
}

#[cfg(target_arch = "wasm32")]
async fn package_cache_from_options(
    options: &JsValue,
) -> Result<Option<Arc<dyn runmat_package_cache::CacheBackend>>, JsValue> {
    if options.is_null() || options.is_undefined() || !options.is_object() {
        return Ok(None);
    }
    let value = js_sys::Reflect::get(options, &JsValue::from_str("packageCacheProvider"))?;
    if value.is_null() || value.is_undefined() {
        return Ok(None);
    }
    let backend = JsPackageCacheBackend::new(&value)?;
    runmat_package_cache::CacheBackend::snapshot(&backend)
        .await
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    Ok(Some(Arc::new(backend)))
}

#[cfg(not(target_arch = "wasm32"))]
async fn package_cache_from_options(
    _options: &JsValue,
) -> Result<Option<Arc<dyn runmat_package_cache::CacheBackend>>, JsValue> {
    Ok(None)
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn install_fs_provider_from_options(
    options: &JsValue,
) -> Result<Option<std::rc::Rc<dyn runmat_runtime::context::RuntimeHostService>>, JsValue> {
    if options.is_null() || options.is_undefined() || !options.is_object() {
        return Ok(None);
    }
    let value = js_sys::Reflect::get(options, &JsValue::from_str("fsProvider"))?;
    if value.is_null() || value.is_undefined() {
        return Ok(None);
    }
    install_js_fs_provider(&value).map(Some)
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn install_fs_provider_from_options(
    _options: &JsValue,
) -> Result<Option<std::rc::Rc<dyn runmat_runtime::context::RuntimeHostService>>, JsValue> {
    Ok(None)
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn install_fs_provider_value(
    bindings: JsValue,
) -> Result<std::rc::Rc<dyn runmat_runtime::context::RuntimeHostService>, JsValue> {
    install_js_fs_provider(&bindings)
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn install_fs_provider_value(
    _bindings: JsValue,
) -> Result<std::rc::Rc<dyn runmat_runtime::context::RuntimeHostService>, JsValue> {
    Err(crate::wire::errors::js_error(
        "filesystem provider installation is only available when targeting wasm32",
    ))
}

#[cfg(target_arch = "wasm32")]
fn ensure_getrandom_js() {
    let mut buf = [0u8; 1];
    if let Err(err) = getrandom::getrandom(&mut buf) {
        log::warn!("RunMat wasm: failed to initialize JS randomness source: {err}");
    }
}

#[cfg(target_arch = "wasm32")]
struct WasmTelemetrySink {
    callback: js_sys::Function,
}

#[cfg(target_arch = "wasm32")]
unsafe impl Send for WasmTelemetrySink {}

#[cfg(target_arch = "wasm32")]
unsafe impl Sync for WasmTelemetrySink {}

#[cfg(target_arch = "wasm32")]
impl TelemetrySink for WasmTelemetrySink {
    fn emit(&self, payload_json: String) {
        let value = js_sys::JSON::parse(&payload_json).unwrap_or_else(|_| payload_json.into());
        let _ = self.callback.call1(&JsValue::NULL, &value);
    }
}
