use std::sync::Arc;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use runmat_builtins::Value;
use runmat_core::{TelemetryPlatformInfo, TelemetrySink};
use runmat_runtime::build_runtime_error;
use runmat_runtime::builtins::wasm_registry;

use crate::api::plot::ensure_figure_event_bridge;
use crate::api::session::RunMatWasm;
use crate::runtime::config::{apply_plotting_overrides, InitOptions, SessionConfig};
use crate::runtime::filesystem::install_js_fs_provider;
use crate::runtime::gpu::{
    capture_gpu_adapter_info, initialize_gpu_provider, install_cpu_provider, GpuStatus,
};
use crate::runtime::logging::{init_logging_once, set_log_filter_override};
use crate::runtime::snapshot::resolve_snapshot_bytes;
use crate::wire::errors::{init_error, init_error_with_details, js_value_to_string, InitErrorCode};

#[wasm_bindgen(js_name = registerFsProvider)]
pub fn register_fs_provider(bindings: JsValue) -> Result<(), JsValue> {
    install_fs_provider_value(bindings).map_err(|err| {
        init_error_with_details(
            InitErrorCode::FilesystemProvider,
            "Failed to register filesystem provider",
            Some(err),
        )
    })
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
    if let Some(level) = parsed_opts.log_level.as_deref() {
        set_log_filter_override(level);
    }
    init_logging_once();
    #[cfg(target_arch = "wasm32")]
    ensure_getrandom_js();
    #[cfg(target_arch = "wasm32")]
    ensure_figure_event_bridge();
    install_fs_provider_from_options(&options).map_err(|err| {
        init_error_with_details(
            InitErrorCode::FilesystemProvider,
            "Failed to install filesystem provider",
            Some(err),
        )
    })?;
    #[cfg(target_arch = "wasm32")]
    {
        if !options.is_null() && !options.is_undefined() {
            if let Ok(stream_value) =
                js_sys::Reflect::get(&options, &JsValue::from_str("snapshotStream"))
            {
                if !stream_value.is_null() && !stream_value.is_undefined() {
                    parsed_opts.snapshot_stream = Some(stream_value);
                }
            }
            if let Ok(emitter_value) =
                js_sys::Reflect::get(&options, &JsValue::from_str("telemetryEmitter"))
            {
                if !emitter_value.is_null() && !emitter_value.is_undefined() {
                    parsed_opts.telemetry_emitter = Some(emitter_value);
                }
            }
        }
    }

    apply_plotting_overrides(&parsed_opts);
    wasm_registry::register_all();
    ensure_internal_builtins();
    let builtin_count = runmat_builtins::builtin_functions().len();
    log::info!("RunMat wasm: builtins registered ({builtin_count})");
    #[cfg(target_arch = "wasm32")]
    web_sys::console::log_1(&format!("RunMat wasm: builtins registered ({builtin_count})").into());

    let config = SessionConfig::from_options(&parsed_opts);
    config.apply_env_overrides();
    let snapshot_seed = resolve_snapshot_bytes(&parsed_opts).await.map_err(|err| {
        let message = js_value_to_string(err.clone());
        init_error_with_details(InitErrorCode::SnapshotResolution, message, Some(err))
    })?;

    let mut session = runmat_core::RunMatSession::with_snapshot_bytes(
        config.enable_jit,
        config.verbose,
        snapshot_seed.as_deref(),
    )
    .map_err(|err| {
        init_error(
            InitErrorCode::SessionCreation,
            format!("Failed to initialize RunMat session: {err}"),
        )
    })?;
    session.set_telemetry_consent(config.telemetry_consent);
    if let Some(cid) = config.telemetry_client_id.clone() {
        session.set_telemetry_client_id(Some(cid));
    }
    session.set_emit_fusion_plan(config.emit_fusion_plan);
    session.set_compat_mode(config.language_compat);
    session.set_callstack_limit(config.callstack_limit);
    session.set_error_namespace(config.error_namespace.clone());
    session.set_source_name_override(Some("<wasm>".to_string()));

    let mut gpu_status = GpuStatus {
        requested: config.enable_gpu,
        active: false,
        error: None,
        adapter: None,
    };

    if config.enable_gpu {
        match initialize_gpu_provider(&config).await {
            Ok(_) => {
                gpu_status.active = true;
                gpu_status.adapter = capture_gpu_adapter_info();
                #[cfg(target_arch = "wasm32")]
                {
                    if let Err(err) =
                        runmat_runtime::builtins::plotting::context::ensure_context_from_provider()
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
                let message = js_value_to_string(err.clone());
                log::warn!(
                    "RunMat wasm: GPU initialization failed (falling back to CPU): {message}"
                );
                gpu_status.error = Some(message);
                install_cpu_provider(&config);
            }
        }
    } else {
        install_cpu_provider(&config);
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

    Ok(RunMatWasm::new(
        session,
        snapshot_seed,
        config,
        gpu_status,
        telemetry_sink,
    ))
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn install_fs_provider_from_options(options: &JsValue) -> Result<(), JsValue> {
    if options.is_null() || options.is_undefined() || !options.is_object() {
        return Ok(());
    }
    let value = js_sys::Reflect::get(options, &JsValue::from_str("fsProvider"))?;
    if value.is_null() || value.is_undefined() {
        return Ok(());
    }
    install_js_fs_provider(&value)
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn install_fs_provider_from_options(_options: &JsValue) -> Result<(), JsValue> {
    Ok(())
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn install_fs_provider_value(bindings: JsValue) -> Result<(), JsValue> {
    install_js_fs_provider(&bindings)
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn install_fs_provider_value(_bindings: JsValue) -> Result<(), JsValue> {
    Err(crate::wire::errors::js_error(
        "registerFsProvider is only available when targeting wasm32",
    ))
}

#[cfg(target_arch = "wasm32")]
fn ensure_getrandom_js() {
    let mut buf = [0u8; 1];
    if let Err(err) = getrandom::getrandom(&mut buf) {
        log::warn!("RunMat wasm: failed to initialize JS randomness source: {err}");
    }
}

fn ensure_internal_builtins() {
    if runmat_builtins::builtin_function_by_name("make_handle").is_none() {
        register_make_handle_fallback();
    }
    if runmat_builtins::builtin_function_by_name("make_anon").is_none() {
        register_make_anon_fallback();
    }
}

fn register_make_handle_fallback() {
    use runmat_builtins::wasm_registry::submit_builtin_function;
    use runmat_builtins::{BuiltinFunction, Type};

    fn make_handle_impl(args: &[Value]) -> runmat_builtins::BuiltinFuture {
        let args = args.to_vec();
        Box::pin(async move {
            if args.len() != 1 {
                return Err(build_runtime_error("make_handle: expected 1 input")
                    .with_identifier("RunMat:make_handle:InvalidInput")
                    .build());
            }
            let name: String = std::convert::TryInto::try_into(&args[0]).map_err(|err| {
                build_runtime_error(format!("make_handle: {err}"))
                    .with_identifier("RunMat:make_handle:InvalidInput")
                    .build()
            })?;
            Ok(Value::FunctionHandle(name))
        })
    }

    let builtin = BuiltinFunction::new(
        "make_handle",
        "",
        "",
        "",
        "",
        vec![Type::String],
        Type::Unknown,
        None,
        make_handle_impl,
        &[],
        false,
        true,
    );
    submit_builtin_function(builtin);
}

fn register_make_anon_fallback() {
    use runmat_builtins::wasm_registry::submit_builtin_function;
    use runmat_builtins::BuiltinFunction;

    fn make_anon_impl(args: &[Value]) -> runmat_builtins::BuiltinFuture {
        let args = args.to_vec();
        Box::pin(async move {
            if args.len() != 2 {
                return Err(build_runtime_error("make_anon: expected 2 inputs")
                    .with_identifier("RunMat:make_anon:InvalidInput")
                    .build());
            }
            let params: String = std::convert::TryInto::try_into(&args[0]).map_err(|err| {
                build_runtime_error(format!("make_anon: {err}"))
                    .with_identifier("RunMat:make_anon:InvalidInput")
                    .build()
            })?;
            let body: String = std::convert::TryInto::try_into(&args[1]).map_err(|err| {
                build_runtime_error(format!("make_anon: {err}"))
                    .with_identifier("RunMat:make_anon:InvalidInput")
                    .build()
            })?;
            Ok(Value::String(format!("@anon({params}) {body}")))
        })
    }

    let builtin = BuiltinFunction::new(
        "make_anon",
        "",
        "",
        "",
        "",
        vec![runmat_builtins::Type::String, runmat_builtins::Type::String],
        runmat_builtins::Type::Unknown,
        None,
        make_anon_impl,
        &[],
        false,
        true,
    );
    submit_builtin_function(builtin);
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
