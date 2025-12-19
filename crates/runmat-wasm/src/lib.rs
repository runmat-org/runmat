#![cfg(target_arch = "wasm32")]

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, OnceLock};
use uuid::Uuid;

#[cfg(target_arch = "wasm32")]
use js_sys::{Array, Error as JsError, Reflect, Uint8Array};
use log::warn;
use runmat_accelerate::{
    initialize_acceleration_provider_with, AccelPowerPreference, AccelerateInitOptions,
    AccelerateProviderPreference, AutoOffloadOptions,
};
use runmat_accelerate_api::ProviderPrecision;
#[cfg(target_arch = "wasm32")]
use runmat_accelerate_api::{AccelContextHandle, AccelContextKind};
use runmat_builtins::{NumericDType, ObjectInstance, StructValue, Value};
use runmat_core::{
    matlab_class_name, value_shape, ExecutionProfiling, ExecutionResult, ExecutionStreamEntry,
    ExecutionStreamKind, FusionPlanDecision, FusionPlanEdge, FusionPlanNode, FusionPlanShader,
    FusionPlanSnapshot, InputHandlerAction, InputRequest, InputRequestKind, InputResponse,
    MaterializedVariable, PendingInput, RunMatSession, StdinEvent, StdinEventKind, WorkspaceEntry,
    WorkspaceMaterializeOptions, WorkspaceMaterializeTarget, WorkspacePreview, WorkspaceSnapshot,
};
use runmat_runtime::builtins::{
    plotting::{set_scatter_target_points, set_surface_vertex_budget},
    wasm_registry,
};
use runmat_runtime::warning_store::RuntimeWarning;
use runmat_logging::{init_logging, set_runtime_log_hook, LoggingOptions, RuntimeLogRecord};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map as JsonMap, Value as JsonValue};
use std::backtrace::Backtrace;
use serde_wasm_bindgen;
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::JsFuture;

#[cfg(target_arch = "wasm32")]
mod fs;
#[cfg(target_arch = "wasm32")]
use crate::fs::install_js_fs_provider;
#[cfg(target_arch = "wasm32")]
use runmat_plot::{
    plots::{LegendEntry, PlotType},
    web::{WebRenderer, WebRendererOptions},
    SharedWgpuContext,
};
#[cfg(target_arch = "wasm32")]
use runmat_runtime::builtins::plotting::web::{
    detach_default_web_renderer as runtime_detach_default_renderer,
    detach_web_renderer as runtime_detach_web_renderer,
};
#[cfg(target_arch = "wasm32")]
use runmat_runtime::builtins::plotting::{
    clear_figure as runtime_clear_figure, close_figure as runtime_close_figure,
    configure_subplot as runtime_configure_subplot, context as plotting_context,
    current_axes_state as runtime_current_axes_state,
    current_figure_handle as runtime_current_figure_handle,
    install_figure_observer as runtime_install_figure_observer,
    install_web_renderer as runtime_install_web_renderer,
    install_web_renderer_for_handle as runtime_install_web_renderer_for_handle,
    new_figure_handle as runtime_new_figure_handle,
    render_figure_snapshot as runtime_render_figure_snapshot,
    select_figure as runtime_select_figure, set_hold as runtime_set_hold,
    web_renderer_ready as runtime_plot_renderer_ready, FigureAxesState, FigureError,
    FigureEventKind, FigureEventView, FigureHandle, HoldMode,
};

const MAX_DATA_PREVIEW: usize = 4096;
const MAX_STRUCT_FIELDS: usize = 64;
const MAX_OBJECT_FIELDS: usize = 64;

#[derive(Clone, Copy)]
enum InitErrorCode {
    InvalidOptions,
    SnapshotResolution,
    FilesystemProvider,
    SessionCreation,
    PlotCanvas,
}

impl InitErrorCode {
    fn as_str(&self) -> &'static str {
        match self {
            InitErrorCode::InvalidOptions => "InvalidOptions",
            InitErrorCode::SnapshotResolution => "SnapshotResolution",
            InitErrorCode::FilesystemProvider => "FilesystemProvider",
            InitErrorCode::SessionCreation => "SessionCreation",
            InitErrorCode::PlotCanvas => "PlotCanvas",
        }
    }
}
const MAX_RECURSION_DEPTH: usize = 2;

#[derive(Debug, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct InitOptions {
    #[serde(default)]
    enable_jit: Option<bool>,
    #[serde(default)]
    verbose: Option<bool>,
    #[serde(default)]
    enable_gpu: Option<bool>,
    #[serde(default)]
    snapshot_url: Option<String>,
    #[serde(default)]
    snapshot_bytes: Option<Vec<u8>>,
    #[serde(default)]
    telemetry_consent: Option<bool>,
    #[serde(default)]
    wgpu_power_preference: Option<String>,
    #[serde(default)]
    wgpu_force_fallback_adapter: Option<bool>,
    #[cfg(target_arch = "wasm32")]
    #[serde(skip)]
    snapshot_stream: Option<JsValue>,
    #[serde(default)]
    scatter_target_points: Option<u32>,
    #[serde(default)]
    surface_vertex_budget: Option<u64>,
    #[serde(default)]
    telemetry_id: Option<String>,
    #[serde(default)]
    emit_fusion_plan: Option<bool>,
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    static FIGURE_EVENT_CALLBACK: RefCell<Option<js_sys::Function>> = RefCell::new(None);
}
#[cfg(target_arch = "wasm32")]
static FIGURE_EVENT_OBSERVER: OnceLock<()> = OnceLock::new();
#[cfg(target_arch = "wasm32")]
thread_local! {
    static JS_STDIN_HANDLER: RefCell<Option<js_sys::Function>> = RefCell::new(None);
}
#[cfg(target_arch = "wasm32")]
thread_local! {
    static STDOUT_SUBSCRIBERS: RefCell<HashMap<u32, js_sys::Function>> =
        RefCell::new(HashMap::new());
}
#[cfg(target_arch = "wasm32")]
static STDOUT_FORWARDER: OnceLock<
    Arc<dyn Fn(&runmat_runtime::console::ConsoleEntry) + Send + Sync + 'static>,
> = OnceLock::new();
static STDOUT_NEXT_ID: AtomicU32 = AtomicU32::new(1);

#[cfg(target_arch = "wasm32")]
thread_local! {
    static RUNTIME_LOG_SUBSCRIBERS: RefCell<HashMap<u32, js_sys::Function>> =
        RefCell::new(HashMap::new());
}
#[cfg(target_arch = "wasm32")]
static RUNTIME_LOG_FORWARDER: OnceLock<Arc<dyn Fn(&runmat_logging::RuntimeLogRecord) + Send + Sync + 'static>> =
    OnceLock::new();
static RUNTIME_LOG_NEXT_ID: AtomicU32 = AtomicU32::new(1);

#[derive(Clone)]
struct SessionConfig {
    enable_jit: bool,
    verbose: bool,
    telemetry_consent: bool,
    telemetry_client_id: Option<String>,
    enable_gpu: bool,
    wgpu_power_preference: AccelPowerPreference,
    wgpu_force_fallback_adapter: bool,
    auto_offload: AutoOffloadOptions,
    emit_fusion_plan: bool,
}

impl SessionConfig {
    fn from_options(opts: &InitOptions) -> Self {
        Self {
            enable_jit: opts.enable_jit.unwrap_or(false) && cfg!(feature = "jit"),
            verbose: opts.verbose.unwrap_or(false),
            telemetry_consent: opts.telemetry_consent.unwrap_or(true),
            telemetry_client_id: opts.telemetry_id.clone(),
            enable_gpu: opts.enable_gpu.unwrap_or(true),
            wgpu_power_preference: parse_power_preference(opts.wgpu_power_preference.as_deref()),
            wgpu_force_fallback_adapter: opts.wgpu_force_fallback_adapter.unwrap_or(false),
            auto_offload: AutoOffloadOptions::default(),
            emit_fusion_plan: opts.emit_fusion_plan.unwrap_or(false),
        }
    }

    fn to_accel_options(&self) -> AccelerateInitOptions {
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
}

#[wasm_bindgen]
pub struct RunMatWasm {
    session: RefCell<RunMatSession>,
    snapshot_seed: Option<Vec<u8>>,
    config: RefCell<SessionConfig>,
    gpu_status: GpuStatus,
    disposed: Cell<bool>,
}

#[wasm_bindgen]
impl RunMatWasm {
    #[wasm_bindgen(js_name = execute)]
    pub fn execute(&self, source: String) -> Result<JsValue, JsValue> {
        let mut session = self.session.borrow_mut();
        let result = session
            .execute(&source)
            .map_err(|err| js_error(&format!("RunMat execution failed: {err}")))?;
        let payload = ExecutionPayload::from(result);
        serde_wasm_bindgen::to_value(&payload)
            .map_err(|err| js_error(&format!("Failed to serialize execution result: {err}")))
    }

    #[wasm_bindgen(js_name = resetSession)]
    pub fn reset_session(&self) -> Result<(), JsValue> {
        wasm_registry::register_all();
        let builtin_count = runmat_builtins::builtin_functions().len();
        log::warn!("RunMat wasm: builtins registered ({builtin_count})");
        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(
            &format!("RunMat wasm: builtins registered ({builtin_count})").into(),
        );
        let config = self.config.borrow();
        let consent = config.telemetry_consent;
        let mut session = RunMatSession::with_snapshot_bytes(
            config.enable_jit,
            config.verbose,
            self.snapshot_seed.as_deref(),
        )
        .map_err(|err| js_error(&format!("Failed to reset session: {err}")))?;
        configure_session_input_handler(&mut session);
        session.set_telemetry_consent(consent);
        if let Some(cid) = config.telemetry_client_id.clone() {
            session.set_telemetry_client_id(Some(cid));
        } else {
            session.set_telemetry_client_id(None);
        }
        session.set_emit_fusion_plan(config.emit_fusion_plan);
        let mut slot = self.session.borrow_mut();
        *slot = session;
        Ok(())
    }

    #[wasm_bindgen(js_name = cancelExecution)]
    pub fn cancel_execution(&self) {
        self.session.borrow().cancel_execution();
    }

    #[wasm_bindgen(js_name = setInputHandler)]
    pub fn set_input_handler(&self, handler: JsValue) -> Result<(), JsValue> {
        #[cfg(target_arch = "wasm32")]
        {
            if handler.is_null() || handler.is_undefined() {
                set_js_stdin_handler(None);
                return Ok(());
            }
            let func = handler
                .dyn_into::<js_sys::Function>()
                .map_err(|_| js_error("setInputHandler expects a Function or null"))?;
            set_js_stdin_handler(Some(func));
            configure_session_input_handler(&mut self.session.borrow_mut());
            return Ok(());
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = handler;
            Err(js_error(
                "setInputHandler is only available when targeting wasm32",
            ))
        }
    }

    #[wasm_bindgen(js_name = resumeInput)]
    pub fn resume_input(&self, request_id: String, value: JsValue) -> Result<JsValue, JsValue> {
        #[cfg(target_arch = "wasm32")]
        {
            let uuid = Uuid::parse_str(request_id.trim())
                .map_err(|_| js_error("resumeInput: invalid request id"))?;
            let response = coerce_resume_payload(value)?;
            let mut session = self.session.borrow_mut();
            let result = session
                .resume_input(uuid, response)
                .map_err(|err| js_error(&format!("RunMat resumeInput failed: {err}")))?;
            let payload = ExecutionPayload::from(result);
            return serde_wasm_bindgen::to_value(&payload)
                .map_err(|err| js_error(&format!("Failed to serialize execution result: {err}")));
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = (request_id, value);
            Err(js_error(
                "resumeInput is only available when targeting wasm32",
            ))
        }
    }

    #[wasm_bindgen(js_name = pendingStdinRequests)]
    pub fn pending_stdin_requests(&self) -> Result<JsValue, JsValue> {
        #[cfg(target_arch = "wasm32")]
        {
            let session = self.session.borrow();
            let pending: Vec<PendingInputPayload> = session
                .pending_requests()
                .into_iter()
                .map(PendingInputPayload::from)
                .collect();
            return serde_wasm_bindgen::to_value(&pending).map_err(|err| {
                js_error(&format!(
                    "Failed to serialize pending stdin requests: {err}"
                ))
            });
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            Err(js_error(
                "pendingStdinRequests is only available when targeting wasm32",
            ))
        }
    }

    #[wasm_bindgen(js_name = setFusionPlanEnabled)]
    pub fn set_fusion_plan_enabled(&self, enabled: bool) {
        self.session.borrow_mut().set_emit_fusion_plan(enabled);
        if let Ok(mut cfg) = self.config.try_borrow_mut() {
            cfg.emit_fusion_plan = enabled;
        }
    }

    #[wasm_bindgen(js_name = materializeVariable)]
    pub fn materialize_variable(
        &self,
        selector: JsValue,
        options: JsValue,
    ) -> Result<JsValue, JsValue> {
        #[cfg(target_arch = "wasm32")]
        {
            let target = parse_materialize_target(selector)?;
            let opts = parse_materialize_options(options)?;
            let mut session = self.session.borrow_mut();
            let value = session
                .materialize_variable(target, opts)
                .map_err(|err| js_error(&format!("materializeVariable failed: {err}")))?;
            let payload = MaterializedVariablePayload::from(value);
            return serde_wasm_bindgen::to_value(&payload).map_err(|err| {
                js_error(&format!(
                    "Failed to serialize materialized workspace value: {err}"
                ))
            });
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = (selector, options);
            Err(js_error(
                "materializeVariable is only available when targeting wasm32",
            ))
        }
    }

    #[wasm_bindgen(js_name = stats)]
    pub fn stats(&self) -> Result<JsValue, JsValue> {
        let payload = {
            let session = self.session.borrow();
            StatsPayload::from(session.stats())
        };
        serde_wasm_bindgen::to_value(&payload)
            .map_err(|err| js_error(&format!("Failed to serialize stats: {err}")))
    }

    #[wasm_bindgen(js_name = clearWorkspace)]
    pub fn clear_workspace(&self) {
        self.session.borrow_mut().clear_variables();
    }

    #[wasm_bindgen(js_name = telemetryConsent)]
    pub fn telemetry_consent(&self) -> bool {
        self.session.borrow().telemetry_consent()
    }

    #[wasm_bindgen(js_name = gpuStatus)]
    pub fn gpu_status(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.gpu_status)
            .map_err(|err| js_error(&format!("Failed to serialize GPU status: {err}")))
    }

    #[wasm_bindgen(js_name = telemetryClientId)]
    pub fn telemetry_client_id(&self) -> Option<String> {
        self.session
            .borrow()
            .telemetry_client_id()
            .map(|value| value.to_string())
    }

    #[wasm_bindgen(js_name = memoryUsage)]
    pub fn memory_usage(&self) -> Result<JsValue, JsValue> {
        let stats = capture_memory_usage().map_err(|err| {
            js_error(&format!(
                "Failed to capture wasm memory usage: {}",
                js_value_to_string(err.clone())
            ))
        })?;
        serde_wasm_bindgen::to_value(&stats)
            .map_err(|err| js_error(&format!("Failed to serialize memory stats: {err}")))
    }

    #[wasm_bindgen(js_name = dispose)]
    pub fn dispose(&self) {
        if self.disposed.replace(true) {
            return;
        }
        {
            let mut session = self.session.borrow_mut();
            session.cancel_execution();
            session.cancel_all_pending_requests();
            session.clear_variables();
        }
        #[cfg(target_arch = "wasm32")]
        {
            set_js_stdin_handler(None);
            FIGURE_EVENT_CALLBACK.with(|slot| {
                slot.replace(None);
            });
        }
    }
}

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

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = registerPlotCanvas)]
pub async fn register_plot_canvas(canvas: web_sys::HtmlCanvasElement) -> Result<(), JsValue> {
    install_canvas_renderer(None, canvas).await.map_err(|err| {
        init_error_with_details(
            InitErrorCode::PlotCanvas,
            "Failed to register plot canvas",
            Some(err),
        )
    })
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = registerFigureCanvas)]
pub async fn register_figure_canvas(
    handle: u32,
    canvas: web_sys::HtmlCanvasElement,
) -> Result<(), JsValue> {
    install_canvas_renderer(Some(handle), canvas)
        .await
        .map_err(|err| {
            init_error_with_details(
                InitErrorCode::PlotCanvas,
                "Failed to register figure canvas",
                Some(err),
            )
        })
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = deregisterPlotCanvas)]
pub fn deregister_plot_canvas() {
    runtime_detach_default_renderer();
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = deregisterFigureCanvas)]
pub fn deregister_figure_canvas(handle: u32) {
    runtime_detach_web_renderer(handle);
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = newFigureHandle)]
pub fn wasm_new_figure_handle() -> u32 {
    runtime_new_figure_handle().as_u32()
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = selectFigure)]
pub fn wasm_select_figure(handle: u32) {
    runtime_select_figure(FigureHandle::from(handle));
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = currentFigureHandle)]
pub fn wasm_current_figure_handle() -> u32 {
    runtime_current_figure_handle().as_u32()
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = configureSubplot)]
pub fn wasm_configure_subplot(rows: u32, cols: u32, index: u32) -> Result<(), JsValue> {
    runtime_configure_subplot(rows as usize, cols as usize, index as usize)
        .map_err(figure_error_to_js)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = setHoldMode)]
pub fn wasm_set_hold_mode(mode: JsValue) -> Result<bool, JsValue> {
    let parsed = parse_hold_mode(mode)?;
    Ok(runtime_set_hold(parsed))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = clearFigure)]
pub fn wasm_clear_figure(handle: JsValue) -> Result<u32, JsValue> {
    let target = parse_optional_handle(handle)?;
    let cleared = runtime_clear_figure(target).map_err(figure_error_to_js)?;
    Ok(cleared.as_u32())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = closeFigure)]
pub fn wasm_close_figure(handle: JsValue) -> Result<u32, JsValue> {
    let target = parse_optional_handle(handle)?;
    let closed = runtime_close_figure(target).map_err(figure_error_to_js)?;
    Ok(closed.as_u32())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = currentAxesInfo)]
pub fn wasm_current_axes_info() -> JsValue {
    axes_state_to_js(runtime_current_axes_state())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = renderFigureImage)]
pub async fn wasm_render_figure_image(
    handle: JsValue,
    width: Option<u32>,
    height: Option<u32>,
) -> Result<Uint8Array, JsValue> {
    let target = parse_optional_handle(handle)?.unwrap_or_else(runtime_current_figure_handle);
    let bytes = runtime_render_figure_snapshot(target, width.unwrap_or(0), height.unwrap_or(0))
        .await
        .map_err(figure_error_to_js)?;
    Ok(Uint8Array::from(bytes.as_slice()))
}

#[cfg(target_arch = "wasm32")]
fn shared_webgpu_context() -> Option<SharedWgpuContext> {
    if let Some(ctx) = runmat_plot::shared_wgpu_context() {
        return Some(ctx);
    }

    let handle = runmat_accelerate_api::export_context(AccelContextKind::Plotting)?;
    match handle {
        AccelContextHandle::Wgpu(ctx) => {
            let shared = SharedWgpuContext {
                instance: ctx.instance.clone(),
                device: ctx.device,
                queue: ctx.queue,
                adapter: ctx.adapter,
                adapter_info: ctx.adapter_info.clone(),
                limits: ctx.limits,
                features: ctx.features,
            };
            runmat_plot::install_shared_wgpu_context(shared.clone());
            Some(shared)
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = plotRendererReady)]
pub fn plot_renderer_ready() -> bool {
    runtime_plot_renderer_ready()
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = onFigureEvent)]
pub fn on_figure_event(callback: JsValue) -> Result<(), JsValue> {
    ensure_figure_event_bridge();
    if callback.is_null() || callback.is_undefined() {
        FIGURE_EVENT_CALLBACK.with(|slot| {
            slot.borrow_mut().take();
        });
        return Ok(());
    }
    let func = callback
        .dyn_ref::<js_sys::Function>()
        .ok_or_else(|| js_error("onFigureEvent expects a Function or null"))?
        .clone();
    FIGURE_EVENT_CALLBACK.with(|slot| {
        *slot.borrow_mut() = Some(func);
    });
    Ok(())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = subscribeStdout)]
pub fn subscribe_stdout(callback: JsValue) -> Result<u32, JsValue> {
    init_logging_once();
    let function = callback
        .dyn_into::<js_sys::Function>()
        .map_err(|_| js_error("subscribeStdout expects a Function"))?;
    ensure_stdout_forwarder_installed();
    let id = STDOUT_NEXT_ID.fetch_add(1, Ordering::Relaxed);
    STDOUT_SUBSCRIBERS.with(|cell| {
        cell.borrow_mut().insert(id, function);
    });
    Ok(id)
}

#[cfg(not(target_arch = "wasm32"))]
#[wasm_bindgen(js_name = subscribeStdout)]
pub fn subscribe_stdout(_callback: JsValue) -> Result<u32, JsValue> {
    Err(js_error(
        "subscribeStdout is only available when targeting wasm32",
    ))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = unsubscribeStdout)]
pub fn unsubscribe_stdout(id: u32) {
    let is_empty = STDOUT_SUBSCRIBERS.with(|cell| {
        let mut map = cell.borrow_mut();
        map.remove(&id);
        map.is_empty()
    });
    if is_empty {
        runmat_runtime::console::install_forwarder(None);
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[wasm_bindgen(js_name = unsubscribeStdout)]
pub fn unsubscribe_stdout(_id: u32) {}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = subscribeRuntimeLog)]
pub fn subscribe_runtime_log(callback: JsValue) -> Result<u32, JsValue> {
    init_logging_once();
    let function = callback
        .dyn_into::<js_sys::Function>()
        .map_err(|_| js_error("subscribeRuntimeLog expects a Function"))?;
    ensure_runtime_log_forwarder_installed();
    let id = RUNTIME_LOG_NEXT_ID.fetch_add(1, Ordering::Relaxed);
    RUNTIME_LOG_SUBSCRIBERS.with(|cell| {
        cell.borrow_mut().insert(id, function);
    });
    Ok(id)
}

#[cfg(not(target_arch = "wasm32"))]
#[wasm_bindgen(js_name = subscribeRuntimeLog)]
pub fn subscribe_runtime_log(_callback: JsValue) -> Result<u32, JsValue> {
    Err(js_error(
        "subscribeRuntimeLog is only available when targeting wasm32",
    ))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = unsubscribeRuntimeLog)]
pub fn unsubscribe_runtime_log(id: u32) {
    RUNTIME_LOG_SUBSCRIBERS.with(|cell| {
        cell.borrow_mut().remove(&id);
    });
}

#[cfg(not(target_arch = "wasm32"))]
#[wasm_bindgen(js_name = unsubscribeRuntimeLog)]
pub fn unsubscribe_runtime_log(_id: u32) {}

#[cfg(target_arch = "wasm32")]
fn set_js_stdin_handler(handler: Option<js_sys::Function>) {
    JS_STDIN_HANDLER.with(|slot| *slot.borrow_mut() = handler);
}

#[cfg(target_arch = "wasm32")]
fn invoke_js_stdin_handler(request: &InputRequest) -> InputHandlerAction {
    let handler = match JS_STDIN_HANDLER.with(|slot| slot.borrow().clone()) {
        Some(func) => func,
        None => return InputHandlerAction::Pending,
    };
    let js_request = js_sys::Object::new();
    if let Err(err) = Reflect::set(
        &js_request,
        &JsValue::from_str("prompt"),
        &JsValue::from_str(&request.prompt),
    ) {
        log::warn!(
            "stdin handler: failed to serialize prompt: {}",
            js_value_to_string(err)
        );
    }
    match request.kind {
        InputRequestKind::Line { echo } => {
            Reflect::set(
                &js_request,
                &JsValue::from_str("kind"),
                &JsValue::from_str("line"),
            )
            .unwrap_or_default();
            Reflect::set(
                &js_request,
                &JsValue::from_str("echo"),
                &JsValue::from_bool(echo),
            )
            .unwrap_or_default();
            let value = match handler.call1(&JsValue::NULL, &js_request) {
                Ok(v) => v,
                Err(err) => {
                    return InputHandlerAction::Respond(Err(js_value_to_string(err)));
                }
            };
            if value_should_pending(&value) {
                return InputHandlerAction::Pending;
            }
            if let Some(err) = extract_error_message(&value) {
                return InputHandlerAction::Respond(Err(err));
            }
            if let Some(text) = extract_line_value(&value) {
                return InputHandlerAction::Respond(Ok(InputResponse::Line(text)));
            }
            InputHandlerAction::Respond(Err(
                "stdin handler must return a string for line input".to_string()
            ))
        }
        InputRequestKind::KeyPress => {
            Reflect::set(
                &js_request,
                &JsValue::from_str("kind"),
                &JsValue::from_str("keyPress"),
            )
            .unwrap_or_default();
            let value = match handler.call1(&JsValue::NULL, &js_request) {
                Ok(v) => v,
                Err(err) => {
                    return InputHandlerAction::Respond(Err(js_value_to_string(err)));
                }
            };
            if value_should_pending(&value) {
                return InputHandlerAction::Pending;
            }
            if let Some(err) = extract_error_message(&value) {
                return InputHandlerAction::Respond(Err(err));
            }
            InputHandlerAction::Respond(Ok(InputResponse::KeyPress))
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn value_should_pending(value: &JsValue) -> bool {
    if value.is_null() || value.is_undefined() {
        return true;
    }
    if value.is_instance_of::<js_sys::Promise>() {
        return true;
    }
    if value.is_object() {
        let obj = js_sys::Object::from(value.clone());
        if let Ok(flag) = Reflect::get(&obj, &JsValue::from_str("pending")) {
            if flag.as_bool().unwrap_or(false) {
                return true;
            }
        }
    }
    false
}

#[cfg(target_arch = "wasm32")]
fn extract_error_message(value: &JsValue) -> Option<String> {
    if !value.is_object() {
        return None;
    }
    let obj = js_sys::Object::from(value.clone());
    Reflect::get(&obj, &JsValue::from_str("error"))
        .ok()
        .and_then(|val| val.as_string())
}

#[cfg(target_arch = "wasm32")]
fn extract_line_value(value: &JsValue) -> Option<String> {
    if let Some(text) = value.as_string() {
        return Some(text);
    }
    if !value.is_object() {
        return None;
    }
    let obj = js_sys::Object::from(value.clone());
    if let Ok(raw) = Reflect::get(&obj, &JsValue::from_str("value")) {
        if let Some(text) = raw.as_string() {
            return Some(text);
        }
    }
    if let Ok(raw) = Reflect::get(&obj, &JsValue::from_str("line")) {
        if let Some(text) = raw.as_string() {
            return Some(text);
        }
    }
    None
}

#[cfg(target_arch = "wasm32")]
fn coerce_resume_payload(value: JsValue) -> Result<Result<InputResponse, String>, JsValue> {
    if value.is_object() {
        let obj = js_sys::Object::from(value.clone());
        if let Ok(err) = Reflect::get(&obj, &JsValue::from_str("error")) {
            if let Some(text) = err.as_string() {
                return Ok(Err(text));
            }
        }
        if let Ok(kind) = Reflect::get(&obj, &JsValue::from_str("kind")) {
            if let Some(text) = kind.as_string() {
                match text.as_str() {
                    "keyPress" => return Ok(Ok(InputResponse::KeyPress)),
                    "line" => {}
                    other => {
                        return Err(js_error(&format!(
                        "resumeInput: unsupported kind '{other}'. Expected 'line' or 'keyPress'."
                    )))
                    }
                }
            }
        }
    }
    if let Some(text) = extract_line_value(&value) {
        return Ok(Ok(InputResponse::Line(text)));
    }
    if value.is_null() || value.is_undefined() {
        return Ok(Ok(InputResponse::Line(String::new())));
    }
    if let Some(num) = value.as_f64() {
        if num.is_finite() {
            return Ok(Ok(InputResponse::Line(num.to_string())));
        }
    }
    if let Some(flag) = value.as_bool() {
        return Ok(Ok(InputResponse::Line(if flag {
            "1".to_string()
        } else {
            "0".to_string()
        })));
    }
    Err(js_error(
        "resumeInput expects a string, number, boolean, or { value, kind } payload",
    ))
}

#[cfg(target_arch = "wasm32")]
fn configure_session_input_handler(session: &mut RunMatSession) {
    session.install_input_handler(js_input_bridge);
}

#[cfg(not(target_arch = "wasm32"))]
fn configure_session_input_handler(_session: &mut RunMatSession) {}

#[cfg(target_arch = "wasm32")]
fn js_input_bridge(request: &InputRequest) -> InputHandlerAction {
    invoke_js_stdin_handler(request)
}

#[cfg(target_arch = "wasm32")]
async fn install_canvas_renderer(
    handle: Option<u32>,
    canvas: web_sys::HtmlCanvasElement,
) -> Result<(), JsValue> {
    init_logging_once();
    let options = WebRendererOptions::default();
    let renderer = match shared_webgpu_context() {
        Some(shared) => {
            WebRenderer::with_shared_context(canvas.clone(), options.clone(), shared).await
        }
        None => WebRenderer::new(canvas, options).await,
    }
    .map_err(|err| js_error(&format!("Failed to initialize plot renderer: {err}")))?;
    match handle {
        Some(id) => {
            runtime_detach_web_renderer(id);
            runtime_install_web_renderer_for_handle(id, renderer)
        }
        .map_err(|err| js_error(&format!("Failed to register plot renderer: {err}")))?,
        None => {
            runtime_detach_default_renderer();
            runtime_install_web_renderer(renderer)
        }
        .map_err(|err| js_error(&format!("Failed to register plot renderer: {err}")))?,
    };
    Ok(())
}

#[wasm_bindgen(js_name = initRunMat)]
pub async fn init_runmat(options: JsValue) -> Result<RunMatWasm, JsValue> {
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
    #[cfg(target_arch = "wasm32")]
    {
        if !options.is_null() && !options.is_undefined() {
            if let Ok(stream_value) = Reflect::get(&options, &JsValue::from_str("snapshotStream")) {
                if !stream_value.is_null() && !stream_value.is_undefined() {
                    parsed_opts.snapshot_stream = Some(stream_value);
                }
            }
        }
    }

    apply_plotting_overrides(&parsed_opts);
    wasm_registry::register_all();
    let builtin_count = runmat_builtins::builtin_functions().len();
    log::info!("RunMat wasm: builtins registered ({builtin_count})");
    let builtin_count = runmat_builtins::builtin_functions().len();
    log::info!("RunMat wasm: builtins registered ({builtin_count})");
    #[cfg(target_arch = "wasm32")]
    web_sys::console::log_1(&format!("RunMat wasm: builtins registered ({builtin_count})").into());

    let config = SessionConfig::from_options(&parsed_opts);
    let snapshot_seed = resolve_snapshot_bytes(&parsed_opts).await.map_err(|err| {
        let message = js_value_to_string(err.clone());
        init_error_with_details(InitErrorCode::SnapshotResolution, message, Some(err))
    })?;

    let mut session = RunMatSession::with_snapshot_bytes(
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
    configure_session_input_handler(&mut session);
    session.set_telemetry_consent(config.telemetry_consent);
    if let Some(cid) = config.telemetry_client_id.clone() {
        session.set_telemetry_client_id(Some(cid));
    }
    session.set_emit_fusion_plan(config.emit_fusion_plan);

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
                    if let Err(err) = plotting_context::ensure_context_from_provider() {
                        warn!("RunMat wasm: unable to install shared plotting context: {err}");
                        gpu_status.error = Some(err.to_string());
                    }
                }
            }
            Err(err) => {
                let message = js_value_to_string(err.clone());
                warn!("RunMat wasm: GPU initialization failed (falling back to CPU): {message}");
                gpu_status.error = Some(message);
                install_cpu_provider(&config);
            }
        }
    } else {
        install_cpu_provider(&config);
    }

    Ok(RunMatWasm {
        session: RefCell::new(session),
        snapshot_seed,
        config: RefCell::new(config),
        gpu_status,
        disposed: Cell::new(false),
    })
}

#[cfg(target_arch = "wasm32")]
fn ensure_getrandom_js() {
    let mut buf = [0u8; 1];
    if let Err(err) = getrandom::getrandom(&mut buf) {
        warn!("RunMat wasm: failed to initialize JS randomness source: {err}");
    }
}

fn apply_plotting_overrides(opts: &InitOptions) {
    if let Some(points) = opts.scatter_target_points {
        set_scatter_target_points(points);
    }
    if let Some(budget) = opts.surface_vertex_budget {
        set_surface_vertex_budget(budget);
    }
}

fn install_cpu_provider(config: &SessionConfig) {
    let mut options = config.to_accel_options();
    options.enabled = false;
    options.provider = AccelerateProviderPreference::InProcess;
    initialize_acceleration_provider_with(&options);
}

#[cfg(target_arch = "wasm32")]
fn ensure_figure_event_bridge() {
    FIGURE_EVENT_OBSERVER.get_or_init(|| {
        let observer: Arc<dyn for<'a> Fn(FigureEventView<'a>) + Send + Sync> =
            Arc::new(|event| emit_js_figure_event(event));
        let _ = runtime_install_figure_observer(observer);
    });
}

#[cfg(target_arch = "wasm32")]
fn ensure_runtime_log_forwarder_installed() {
    RUNTIME_LOG_FORWARDER.get_or_init(|| {
        let forwarder: Arc<dyn Fn(&RuntimeLogRecord) + Send + Sync> =
            Arc::new(|record: &RuntimeLogRecord| {
                let js_value = serde_wasm_bindgen::to_value(record).unwrap_or(JsValue::NULL);
                RUNTIME_LOG_SUBSCRIBERS.with(|cell| {
                    for cb in cell.borrow().values() {
                        let _ = cb.call1(&JsValue::NULL, &js_value);
                    }
                });
            });
        let hook = forwarder.clone();
        set_runtime_log_hook(move |rec| {
            (hook)(rec);
        });
        forwarder
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn ensure_figure_event_bridge() {}

#[cfg(target_arch = "wasm32")]
fn parse_hold_mode(value: JsValue) -> Result<HoldMode, JsValue> {
    if value.is_null() || value.is_undefined() {
        return Ok(HoldMode::Toggle);
    }
    if let Some(flag) = value.as_bool() {
        return Ok(if flag { HoldMode::On } else { HoldMode::Off });
    }
    if let Some(text) = value.as_string() {
        let normalized = text.trim().to_ascii_lowercase();
        return match normalized.as_str() {
            "on" | "holdon" => Ok(HoldMode::On),
            "off" | "holdoff" => Ok(HoldMode::Off),
            "toggle" | "switch" => Ok(HoldMode::Toggle),
            other => Err(js_error(&format!(
                "Unsupported hold mode '{other}'. Expected 'on', 'off', or 'toggle'."
            ))),
        };
    }
    Err(js_error(
        "setHoldMode expects a string ('on'|'off'|'toggle') or a boolean",
    ))
}

#[cfg(target_arch = "wasm32")]
fn parse_optional_handle(value: JsValue) -> Result<Option<FigureHandle>, JsValue> {
    if value.is_null() || value.is_undefined() {
        return Ok(None);
    }
    if let Some(num) = value.as_f64() {
        if !num.is_finite() || num <= 0.0 {
            return Err(js_error("Figure handles must be positive numbers"));
        }
        return Ok(Some(FigureHandle::from(num.round() as u32)));
    }
    Err(js_error(
        "Figure handles must be numeric or left undefined for the active figure",
    ))
}

#[cfg(target_arch = "wasm32")]
fn figure_error_to_js(err: FigureError) -> JsValue {
    let payload = js_sys::Object::new();
    let message = err.to_string();
    let code = match err {
        FigureError::InvalidHandle(handle) => {
            let _ = Reflect::set(
                &payload,
                &JsValue::from_str("handle"),
                &JsValue::from(handle),
            );
            "InvalidHandle"
        }
        FigureError::InvalidSubplotGrid { rows, cols } => {
            let _ = Reflect::set(
                &payload,
                &JsValue::from_str("rows"),
                &JsValue::from(rows as u32),
            );
            let _ = Reflect::set(
                &payload,
                &JsValue::from_str("cols"),
                &JsValue::from(cols as u32),
            );
            "InvalidSubplotGrid"
        }
        FigureError::InvalidSubplotIndex { rows, cols, index } => {
            let _ = Reflect::set(
                &payload,
                &JsValue::from_str("rows"),
                &JsValue::from(rows as u32),
            );
            let _ = Reflect::set(
                &payload,
                &JsValue::from_str("cols"),
                &JsValue::from(cols as u32),
            );
            let _ = Reflect::set(
                &payload,
                &JsValue::from_str("index"),
                &JsValue::from(index as u32),
            );
            "InvalidSubplotIndex"
        }
        FigureError::InvalidAxesHandle => "InvalidAxesHandle",
        FigureError::RenderFailure(details) => {
            let _ = Reflect::set(
                &payload,
                &JsValue::from_str("details"),
                &JsValue::from_str(details.as_str()),
            );
            "RenderFailure"
        }
    };
    let _ = Reflect::set(
        &payload,
        &JsValue::from_str("code"),
        &JsValue::from_str(code),
    );
    let _ = Reflect::set(
        &payload,
        &JsValue::from_str("message"),
        &JsValue::from_str(&message),
    );
    JsValue::from(payload)
}

#[cfg(target_arch = "wasm32")]
fn axes_state_to_js(state: FigureAxesState) -> JsValue {
    let payload = js_sys::Object::new();
    let _ = Reflect::set(
        &payload,
        &JsValue::from_str("handle"),
        &JsValue::from(state.handle.as_u32()),
    );
    let _ = Reflect::set(
        &payload,
        &JsValue::from_str("axesRows"),
        &JsValue::from(state.rows as u32),
    );
    let _ = Reflect::set(
        &payload,
        &JsValue::from_str("axesCols"),
        &JsValue::from(state.cols as u32),
    );
    let _ = Reflect::set(
        &payload,
        &JsValue::from_str("activeIndex"),
        &JsValue::from(state.active_index as u32),
    );
    payload.into()
}

#[cfg(target_arch = "wasm32")]
fn emit_js_figure_event(event: FigureEventView<'_>) {
    if let FigureEventKind::Closed = event.kind {
        runtime_detach_web_renderer(event.handle.as_u32());
    }
    FIGURE_EVENT_CALLBACK.with(|slot| {
        if let Some(cb) = slot.borrow().as_ref() {
            let payload = js_sys::Object::new();
            let _ = Reflect::set(
                &payload,
                &JsValue::from_str("handle"),
                &JsValue::from(event.handle.as_u32()),
            );
            let _ = Reflect::set(
                &payload,
                &JsValue::from_str("kind"),
                &JsValue::from_str(match event.kind {
                    FigureEventKind::Created => "created",
                    FigureEventKind::Updated => "updated",
                    FigureEventKind::Cleared => "cleared",
                    FigureEventKind::Closed => "closed",
                }),
            );

            if let Some(figure) = event.figure {
                let (rows, cols) = figure.axes_grid();
                let _ = Reflect::set(
                    &payload,
                    &JsValue::from_str("axesRows"),
                    &JsValue::from(rows as u32),
                );
                let _ = Reflect::set(
                    &payload,
                    &JsValue::from_str("axesCols"),
                    &JsValue::from(cols as u32),
                );
                let plot_count = figure.plot_axes_indices().len() as u32;
                let _ = Reflect::set(
                    &payload,
                    &JsValue::from_str("plotCount"),
                    &JsValue::from(plot_count),
                );
                let indices = Array::new();
                for idx in figure.plot_axes_indices() {
                    indices.push(&JsValue::from(*idx as u32));
                }
                let _ = Reflect::set(
                    &payload,
                    &JsValue::from_str("axesIndices"),
                    &JsValue::from(indices),
                );
                if let Some(title) = figure.title.as_ref() {
                    let _ = Reflect::set(
                        &payload,
                        &JsValue::from_str("title"),
                        &JsValue::from_str(title),
                    );
                }
                let _ = Reflect::set(
                    &payload,
                    &JsValue::from_str("gridEnabled"),
                    &JsValue::from_bool(figure.grid_enabled),
                );
                if let Some(x_label) = figure.x_label.as_ref() {
                    let _ = Reflect::set(
                        &payload,
                        &JsValue::from_str("xLabel"),
                        &JsValue::from_str(x_label),
                    );
                }
                if let Some(y_label) = figure.y_label.as_ref() {
                    let _ = Reflect::set(
                        &payload,
                        &JsValue::from_str("yLabel"),
                        &JsValue::from_str(y_label),
                    );
                }
                let legend_entries = figure.legend_entries();
                let _ = Reflect::set(
                    &payload,
                    &JsValue::from_str("legendEnabled"),
                    &JsValue::from_bool(figure.legend_enabled && !legend_entries.is_empty()),
                );
                let _ = Reflect::set(
                    &payload,
                    &JsValue::from_str("legendEntries"),
                    &legend_entries_to_js(&legend_entries),
                );
            } else {
                let _ = Reflect::set(
                    &payload,
                    &JsValue::from_str("axesRows"),
                    &JsValue::from(0u32),
                );
                let _ = Reflect::set(
                    &payload,
                    &JsValue::from_str("axesCols"),
                    &JsValue::from(0u32),
                );
                let _ = Reflect::set(
                    &payload,
                    &JsValue::from_str("plotCount"),
                    &JsValue::from(0u32),
                );
                let _ = Reflect::set(
                    &payload,
                    &JsValue::from_str("axesIndices"),
                    &JsValue::from(Array::new()),
                );
            }

            let _ = cb.call1(&JsValue::NULL, &payload);
        }
    });
}

#[cfg(target_arch = "wasm32")]
fn legend_entries_to_js(entries: &[LegendEntry]) -> JsValue {
    let array = Array::new();
    for entry in entries {
        let item = js_sys::Object::new();
        let _ = Reflect::set(
            &item,
            &JsValue::from_str("label"),
            &JsValue::from_str(&entry.label),
        );
        let _ = Reflect::set(
            &item,
            &JsValue::from_str("plotType"),
            &JsValue::from_str(plot_type_to_str(entry.plot_type)),
        );
        let color = Array::new();
        color.push(&JsValue::from_f64(entry.color.x as f64));
        color.push(&JsValue::from_f64(entry.color.y as f64));
        color.push(&JsValue::from_f64(entry.color.z as f64));
        color.push(&JsValue::from_f64(entry.color.w as f64));
        let _ = Reflect::set(&item, &JsValue::from_str("color"), &color.into());
        array.push(&item.into());
    }
    array.into()
}

#[cfg(target_arch = "wasm32")]
fn plot_type_to_str(plot_type: PlotType) -> &'static str {
    match plot_type {
        PlotType::Line => "line",
        PlotType::Scatter => "scatter",
        PlotType::Bar => "bar",
        PlotType::ErrorBar => "errorbar",
        PlotType::Stairs => "stairs",
        PlotType::Stem => "stem",
        PlotType::Area => "area",
        PlotType::Quiver => "quiver",
        PlotType::Pie => "pie",
        PlotType::Image => "image",
        PlotType::Scatter3 => "scatter3",
        PlotType::Contour => "contour",
        PlotType::ContourFill => "contourf",
    }
}

#[cfg(target_arch = "wasm32")]
fn ensure_stdout_forwarder_installed() {
    let forwarder = STDOUT_FORWARDER.get_or_init(|| {
        Arc::new(dispatch_stdout_entry as fn(&runmat_runtime::console::ConsoleEntry))
            as Arc<dyn Fn(&runmat_runtime::console::ConsoleEntry) + Send + Sync + 'static>
    });
    runmat_runtime::console::install_forwarder(Some(forwarder.clone()));
}

#[cfg(target_arch = "wasm32")]
fn dispatch_stdout_entry(entry: &runmat_runtime::console::ConsoleEntry) {
    let handlers: Vec<js_sys::Function> =
        STDOUT_SUBSCRIBERS.with(|cell| cell.borrow().values().cloned().collect());
    if handlers.is_empty() {
        return;
    }
    if let Ok(payload) =
        serde_wasm_bindgen::to_value(&ConsoleStreamPayload::from_console_entry(entry))
    {
        for handler in handlers {
            let _ = handler.call1(&JsValue::NULL, &payload);
        }
    }
}

#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
async fn initialize_gpu_provider(config: &SessionConfig) -> Result<(), JsValue> {
    use runmat_accelerate::initialize_wgpu_provider_async;

    let options = config.to_accel_options();
    initialize_wgpu_provider_async(&options)
        .await
        .map_err(|err| js_error(&format!("Failed to initialize WebGPU provider: {err}")))
}

#[cfg(not(all(feature = "gpu", target_arch = "wasm32")))]
async fn initialize_gpu_provider(_config: &SessionConfig) -> Result<(), JsValue> {
    Err(js_error(
        "runmat-wasm was built without GPU support or on a non-wasm32 target",
    ))
}

async fn resolve_snapshot_bytes(opts: &InitOptions) -> Result<Option<Vec<u8>>, JsValue> {
    if let Some(bytes) = opts.snapshot_bytes.clone() {
        return Ok(Some(bytes));
    }

    #[cfg(target_arch = "wasm32")]
    {
        if let Some(stream_value) = opts.snapshot_stream.as_ref() {
            if !stream_value.is_null() && !stream_value.is_undefined() {
                let stream: web_sys::ReadableStream = stream_value
                    .clone()
                    .dyn_into()
                    .map_err(|_| js_error("snapshotStream must be a ReadableStream"))?;
                let bytes = read_stream(stream).await?;
                return Ok(Some(bytes));
            }
        }
    }

    if let Some(url) = &opts.snapshot_url {
        let bytes = fetch_snapshot_bytes(url).await?;
        return Ok(Some(bytes));
    }

    Ok(None)
}

#[cfg(target_arch = "wasm32")]
async fn fetch_snapshot_bytes(url: &str) -> Result<Vec<u8>, JsValue> {
    use js_sys::Uint8Array;

    let window = web_sys::window().ok_or_else(|| js_error("window is unavailable"))?;
    let resp_value = JsFuture::from(window.fetch_with_str(url)).await?;
    let resp: web_sys::Response = resp_value
        .dyn_into()
        .map_err(|_| js_error("Fetch response is not a Response"))?;

    if !resp.ok() {
        return Err(js_error(&format!(
            "Failed to fetch snapshot from {url}: status {}",
            resp.status()
        )));
    }

    if let Some(body) = resp.body() {
        return read_stream(body).await;
    }

    let buffer = JsFuture::from(resp.array_buffer()?).await?;
    let array = Uint8Array::new(&buffer);
    Ok(array.to_vec())
}

#[cfg(target_arch = "wasm32")]
async fn read_stream(stream: web_sys::ReadableStream) -> Result<Vec<u8>, JsValue> {
    use js_sys::{Reflect, Uint8Array};
    use web_sys::ReadableStreamDefaultReader;

    let reader_value = stream.get_reader();
    let reader: ReadableStreamDefaultReader = reader_value
        .dyn_into()
        .map_err(|_| js_error("Failed to cast reader"))?;

    let mut chunks: Vec<Vec<u8>> = Vec::new();
    let mut total = 0usize;
    loop {
        let promise = reader.read();
        let result = JsFuture::from(promise).await?;
        let done = Reflect::get(&result, &JsValue::from_str("done"))?
            .as_bool()
            .unwrap_or(false);
        if done {
            break;
        }
        let value = Reflect::get(&result, &JsValue::from_str("value"))?;
        if value.is_undefined() || value.is_null() {
            continue;
        }
        let chunk = Uint8Array::new(&value);
        let mut buffer = vec![0u8; chunk.length() as usize];
        chunk.copy_to(&mut buffer[..]);
        total += buffer.len();
        chunks.push(buffer);
    }

    let mut merged = Vec::with_capacity(total);
    for chunk in chunks {
        merged.extend_from_slice(&chunk);
    }
    Ok(merged)
}

#[cfg(not(target_arch = "wasm32"))]
async fn fetch_snapshot_bytes(_url: &str) -> Result<Vec<u8>, JsValue> {
    Err(js_error(
        "Snapshot fetching is only supported on wasm32 targets",
    ))
}

fn parse_power_preference(input: Option<&str>) -> AccelPowerPreference {
    match input.map(|s| s.to_ascii_lowercase()) {
        Some(ref value) if value.contains("low") => AccelPowerPreference::LowPower,
        Some(ref value) if value.contains("high") => AccelPowerPreference::HighPerformance,
        _ => AccelPowerPreference::Auto,
    }
}

fn js_error(message: &str) -> JsValue {
    JsValue::from_str(message)
}

#[cfg(target_arch = "wasm32")]
fn capture_memory_usage() -> Result<MemoryUsagePayload, JsValue> {
    let memory = wasm_bindgen::memory()
        .dyn_into::<js_sys::WebAssembly::Memory>()
        .map_err(|_| js_error("Active wasm memory handle unavailable"))?;
    let buffer: js_sys::ArrayBuffer = memory
        .buffer()
        .dyn_into()
        .map_err(|_| js_error("Active wasm memory buffer unavailable"))?;
    let bytes = buffer.byte_length() as u64;
    let pages = (bytes / 65_536) as u32;
    Ok(MemoryUsagePayload { bytes, pages })
}

#[cfg(not(target_arch = "wasm32"))]
fn capture_memory_usage() -> Result<MemoryUsagePayload, JsValue> {
    Ok(MemoryUsagePayload { bytes: 0, pages: 0 })
}

fn init_error(code: InitErrorCode, message: impl Into<String>) -> JsValue {
    init_error_with_details(code, message, None)
}

fn init_error_with_details(
    code: InitErrorCode,
    message: impl Into<String>,
    details: Option<JsValue>,
) -> JsValue {
    #[cfg(target_arch = "wasm32")]
    {
        let msg = message.into();
        let error = JsError::new(&msg);
        let _ = Reflect::set(
            error.as_ref(),
            &JsValue::from_str("code"),
            &JsValue::from_str(code.as_str()),
        );
        if let Some(detail) = details {
            let _ = Reflect::set(error.as_ref(), &JsValue::from_str("details"), &detail);
        }
        error.into()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let msg = message.into();
        JsValue::from_str(&format!("{}: {msg}", code.as_str()))
    }
}

fn init_logging_once() {
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| {
        #[cfg(target_arch = "wasm32")]
        {
            std::panic::set_hook(Box::new(|info| {
                let _ = web_sys::console::error_1(&JsValue::from_str(
                    "RunMat panic hook invoked; forwarding to console_error_panic_hook",
                ));
                console_error_panic_hook::hook(info);
                let bt = Backtrace::force_capture();
                let _ = web_sys::console::error_1(&JsValue::from_str(&format!(
                    "RunMat panic backtrace:\n{bt:?}"
                )));
            }));
            let _ = init_logging(LoggingOptions { enable_otlp: false });
            ensure_runtime_log_forwarder_installed();
        }
    });
}

fn js_value_to_string(value: JsValue) -> String {
    value.as_string().unwrap_or_else(|| format!("{value:?}"))
}

#[cfg(target_arch = "wasm32")]
fn install_fs_provider_from_options(options: &JsValue) -> Result<(), JsValue> {
    if options.is_null() || options.is_undefined() || !options.is_object() {
        return Ok(());
    }
    let value = Reflect::get(options, &JsValue::from_str("fsProvider"))?;
    if value.is_null() || value.is_undefined() {
        return Ok(());
    }
    install_js_fs_provider(&value)
}

#[cfg(not(target_arch = "wasm32"))]
fn install_fs_provider_from_options(_options: &JsValue) -> Result<(), JsValue> {
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn install_fs_provider_value(bindings: JsValue) -> Result<(), JsValue> {
    install_js_fs_provider(&bindings)
}

#[cfg(not(target_arch = "wasm32"))]
fn install_fs_provider_value(_bindings: JsValue) -> Result<(), JsValue> {
    Err(js_error(
        "registerFsProvider is only available when targeting wasm32",
    ))
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PendingInputPayload {
    id: String,
    request: InputRequestPayload,
    waiting_ms: u64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MaterializeSelectorPayload {
    name: Option<String>,
    #[serde(alias = "previewToken")]
    token: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MaterializeOptionsPayload {
    limit: Option<usize>,
    slices: Option<JsonValue>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct InputRequestPayload {
    prompt: String,
    kind: &'static str,
    echo: bool,
}

impl From<PendingInput> for PendingInputPayload {
    fn from(input: PendingInput) -> Self {
        let (kind, echo) = match input.request.kind {
            InputRequestKind::Line { echo } => ("line", echo),
            InputRequestKind::KeyPress => ("keyPress", false),
        };
        Self {
            id: input.id.to_string(),
            request: InputRequestPayload {
                prompt: input.request.prompt,
                kind,
                echo,
            },
            waiting_ms: input.waiting_ms,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ExecutionPayload {
    value_text: Option<String>,
    value_json: Option<JsonValue>,
    type_info: Option<String>,
    execution_time_ms: u64,
    used_jit: bool,
    error: Option<String>,
    stdout: Vec<ConsoleStreamPayload>,
    workspace: WorkspacePayload,
    figures_touched: Vec<u32>,
    warnings: Vec<WarningPayload>,
    stdin_events: Vec<StdinEventPayload>,
    profiling: Option<ProfilingPayload>,
    fusion_plan: Option<FusionPlanPayload>,
    stdin_requested: Option<PendingInputPayload>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct MemoryUsagePayload {
    bytes: u64,
    pages: u32,
}

impl From<ExecutionResult> for ExecutionPayload {
    fn from(result: ExecutionResult) -> Self {
        let value_text = result.value.as_ref().map(|v| v.to_string());
        let value_json = result.value.as_ref().map(|v| value_to_json(v, 0));
        Self {
            value_text,
            value_json,
            type_info: result.type_info,
            execution_time_ms: result.execution_time_ms,
            used_jit: result.used_jit,
            error: result.error,
            stdout: result
                .streams
                .into_iter()
                .map(ConsoleStreamPayload::from)
                .collect(),
            workspace: WorkspacePayload::from(result.workspace),
            figures_touched: result.figures_touched,
            warnings: result
                .warnings
                .into_iter()
                .map(WarningPayload::from)
                .collect(),
            stdin_events: result
                .stdin_events
                .into_iter()
                .map(StdinEventPayload::from)
                .collect(),
            profiling: result.profiling.map(ProfilingPayload::from),
            fusion_plan: result.fusion_plan.map(FusionPlanPayload::from),
            stdin_requested: result.stdin_requested.map(PendingInputPayload::from),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ConsoleStreamPayload {
    stream: &'static str,
    text: String,
    timestamp_ms: u64,
}

impl From<ExecutionStreamEntry> for ConsoleStreamPayload {
    fn from(entry: ExecutionStreamEntry) -> Self {
        let stream = match entry.stream {
            ExecutionStreamKind::Stdout => "stdout",
            ExecutionStreamKind::Stderr => "stderr",
        };
        Self {
            stream,
            text: entry.text,
            timestamp_ms: entry.timestamp_ms,
        }
    }
}

impl ConsoleStreamPayload {
    fn from_console_entry(entry: &runmat_runtime::console::ConsoleEntry) -> Self {
        let stream = match entry.stream {
            runmat_runtime::console::ConsoleStream::Stdout => "stdout",
            runmat_runtime::console::ConsoleStream::Stderr => "stderr",
        };
        Self {
            stream,
            text: entry.text.clone(),
            timestamp_ms: entry.timestamp_ms,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct WarningPayload {
    identifier: String,
    message: String,
}

impl From<RuntimeWarning> for WarningPayload {
    fn from(warning: RuntimeWarning) -> Self {
        Self {
            identifier: warning.identifier,
            message: warning.message,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct StdinEventPayload {
    prompt: String,
    kind: &'static str,
    echo: bool,
    value: Option<String>,
    error: Option<String>,
}

impl From<StdinEvent> for StdinEventPayload {
    fn from(event: StdinEvent) -> Self {
        let kind = match event.kind {
            StdinEventKind::Line => "line",
            StdinEventKind::KeyPress => "keyPress",
        };
        Self {
            prompt: event.prompt,
            kind,
            echo: event.echo,
            value: event.value,
            error: event.error,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct WorkspacePayload {
    full: bool,
    version: u64,
    values: Vec<WorkspaceEntryPayload>,
}

impl From<WorkspaceSnapshot> for WorkspacePayload {
    fn from(snapshot: WorkspaceSnapshot) -> Self {
        Self {
            full: snapshot.full,
            version: snapshot.version,
            values: snapshot
                .values
                .into_iter()
                .map(WorkspaceEntryPayload::from)
                .collect(),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct WorkspaceEntryPayload {
    name: String,
    class_name: String,
    dtype: Option<String>,
    shape: Vec<usize>,
    is_gpu: bool,
    size_bytes: Option<u64>,
    preview: Option<WorkspacePreviewPayload>,
    residency: &'static str,
    preview_token: Option<String>,
}

impl From<WorkspaceEntry> for WorkspaceEntryPayload {
    fn from(entry: WorkspaceEntry) -> Self {
        Self {
            name: entry.name,
            class_name: entry.class_name,
            dtype: entry.dtype,
            shape: entry.shape,
            is_gpu: entry.is_gpu,
            size_bytes: entry.size_bytes,
            preview: entry.preview.map(WorkspacePreviewPayload::from),
            residency: entry.residency.as_str(),
            preview_token: entry.preview_token.map(|id| id.to_string()),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct WorkspacePreviewPayload {
    values: Vec<f64>,
    truncated: bool,
}

impl From<WorkspacePreview> for WorkspacePreviewPayload {
    fn from(preview: WorkspacePreview) -> Self {
        Self {
            values: preview.values,
            truncated: preview.truncated,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct MaterializedVariablePayload {
    name: String,
    class_name: String,
    dtype: Option<String>,
    shape: Vec<usize>,
    is_gpu: bool,
    residency: &'static str,
    size_bytes: Option<u64>,
    preview: Option<WorkspacePreviewPayload>,
    value_text: String,
    value_json: JsonValue,
}

impl From<MaterializedVariable> for MaterializedVariablePayload {
    fn from(value: MaterializedVariable) -> Self {
        Self {
            name: value.name,
            class_name: value.class_name,
            dtype: value.dtype,
            shape: value.shape,
            is_gpu: value.is_gpu,
            residency: value.residency.as_str(),
            size_bytes: value.size_bytes,
            preview: value.preview.map(WorkspacePreviewPayload::from),
            value_text: value.value.to_string(),
            value_json: value_to_json(&value.value, 0),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ProfilingPayload {
    total_ms: u64,
    cpu_ms: Option<u64>,
    gpu_ms: Option<u64>,
    kernel_count: Option<u32>,
}

impl From<ExecutionProfiling> for ProfilingPayload {
    fn from(profiling: ExecutionProfiling) -> Self {
        Self {
            total_ms: profiling.total_ms,
            cpu_ms: profiling.cpu_ms,
            gpu_ms: profiling.gpu_ms,
            kernel_count: profiling.kernel_count,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FusionPlanPayload {
    nodes: Vec<FusionPlanNodePayload>,
    edges: Vec<FusionPlanEdgePayload>,
    shaders: Vec<FusionPlanShaderPayload>,
    decisions: Vec<FusionPlanDecisionPayload>,
}

impl From<FusionPlanSnapshot> for FusionPlanPayload {
    fn from(snapshot: FusionPlanSnapshot) -> Self {
        Self {
            nodes: snapshot
                .nodes
                .into_iter()
                .map(FusionPlanNodePayload::from)
                .collect(),
            edges: snapshot
                .edges
                .into_iter()
                .map(FusionPlanEdgePayload::from)
                .collect(),
            shaders: snapshot
                .shaders
                .into_iter()
                .map(FusionPlanShaderPayload::from)
                .collect(),
            decisions: snapshot
                .decisions
                .into_iter()
                .map(FusionPlanDecisionPayload::from)
                .collect(),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FusionPlanNodePayload {
    id: String,
    kind: String,
    label: String,
    shape: Vec<usize>,
    residency: Option<String>,
}

impl From<FusionPlanNode> for FusionPlanNodePayload {
    fn from(node: FusionPlanNode) -> Self {
        Self {
            id: node.id,
            kind: node.kind,
            label: node.label,
            shape: node.shape,
            residency: node.residency,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FusionPlanEdgePayload {
    from: String,
    to: String,
    reason: Option<String>,
}

impl From<FusionPlanEdge> for FusionPlanEdgePayload {
    fn from(edge: FusionPlanEdge) -> Self {
        Self {
            from: edge.from,
            to: edge.to,
            reason: edge.reason,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FusionPlanShaderPayload {
    name: String,
    stage: String,
    workgroup_size: Option<[u32; 3]>,
    source_hash: Option<String>,
}

impl From<FusionPlanShader> for FusionPlanShaderPayload {
    fn from(shader: FusionPlanShader) -> Self {
        Self {
            name: shader.name,
            stage: shader.stage,
            workgroup_size: shader.workgroup_size,
            source_hash: shader.source_hash,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FusionPlanDecisionPayload {
    node_id: String,
    fused: bool,
    reason: Option<String>,
    thresholds: Option<String>,
}

impl From<FusionPlanDecision> for FusionPlanDecisionPayload {
    fn from(decision: FusionPlanDecision) -> Self {
        Self {
            node_id: decision.node_id,
            fused: decision.fused,
            reason: decision.reason,
            thresholds: decision.thresholds,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct StatsPayload {
    total_executions: usize,
    jit_compiled: usize,
    interpreter_fallback: usize,
    total_execution_time_ms: u64,
    average_execution_time_ms: f64,
}

impl From<&runmat_core::ExecutionStats> for StatsPayload {
    fn from(stats: &runmat_core::ExecutionStats) -> Self {
        Self {
            total_executions: stats.total_executions,
            jit_compiled: stats.jit_compiled,
            interpreter_fallback: stats.interpreter_fallback,
            total_execution_time_ms: stats.total_execution_time_ms,
            average_execution_time_ms: stats.average_execution_time_ms,
        }
    }
}

fn parse_materialize_target(value: JsValue) -> Result<WorkspaceMaterializeTarget, JsValue> {
    if value.is_undefined() || value.is_null() {
        return Err(js_error(
            "materializeVariable requires a selector (name or previewToken)",
        ));
    }
    if let Some(token) = value.as_string() {
        return parse_materialize_token_str(&token);
    }
    let payload: MaterializeSelectorPayload = serde_wasm_bindgen::from_value(value.clone())
        .map_err(|err| js_error(&format!("materializeVariable selector: {err}")))?;
    if let Some(token) = payload.token {
        return parse_materialize_token_str(&token);
    }
    if let Some(name) = payload.name {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err(js_error(
                "materializeVariable selector.name must not be empty",
            ));
        }
        return Ok(WorkspaceMaterializeTarget::Name(trimmed.to_string()));
    }
    Err(js_error(
        "materializeVariable selector must include a name or previewToken",
    ))
}

fn parse_materialize_token_str(token: &str) -> Result<WorkspaceMaterializeTarget, JsValue> {
    let parsed = Uuid::parse_str(token.trim())
        .map_err(|_| js_error("materializeVariable previewToken must be a UUID string"))?;
    Ok(WorkspaceMaterializeTarget::Token(parsed))
}

fn parse_materialize_options(value: JsValue) -> Result<WorkspaceMaterializeOptions, JsValue> {
    if value.is_undefined() || value.is_null() {
        return Ok(WorkspaceMaterializeOptions::default());
    }
    if !value.is_object() {
        return Err(js_error("materializeVariable options must be an object"));
    }
    let payload: MaterializeOptionsPayload =
        serde_wasm_bindgen::from_value(value).map_err(|err| {
            js_error(&format!(
                "materializeVariable options could not be parsed: {err}"
            ))
        })?;
    if payload.slices.is_some() {
        return Err(js_error(
            "materializeVariable slices are not supported yet; omit the 'slices' option",
        ));
    }
    let mut opts = WorkspaceMaterializeOptions::default();
    if let Some(limit) = payload.limit {
        opts.max_elements = limit.max(1).min(MAX_DATA_PREVIEW);
    }
    Ok(opts)
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct GpuStatus {
    requested: bool,
    active: bool,
    error: Option<String>,
    adapter: Option<GpuAdapterInfo>,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct GpuAdapterInfo {
    device_id: u32,
    name: String,
    vendor: String,
    backend: Option<String>,
    memory_bytes: Option<u64>,
    precision: Option<String>,
}

fn value_to_json(value: &Value, depth: usize) -> JsonValue {
    if depth >= MAX_RECURSION_DEPTH {
        return json!({
            "kind": "display",
            "className": matlab_class_name(value),
            "shape": value_shape(value),
            "value": value.to_string(),
        });
    }

    match value {
        Value::Int(iv) => json!({
            "kind": "int",
            "className": iv.class_name(),
            "value": iv.to_i64(),
            "shape": scalar_shape(),
        }),
        Value::Num(n) => json!({
            "kind": "double",
            "value": n,
            "shape": scalar_shape(),
        }),
        Value::Complex(re, im) => json!({
            "kind": "complex",
            "real": re,
            "imag": im,
            "shape": scalar_shape(),
        }),
        Value::Bool(b) => json!({
            "kind": "logical",
            "value": b,
            "shape": scalar_shape(),
        }),
        Value::LogicalArray(arr) => {
            let (preview, truncated) = preview_slice(&arr.data, MAX_DATA_PREVIEW);
            let rows = arr.shape.get(0).copied().unwrap_or(0);
            let cols = arr.shape.get(1).copied().unwrap_or(0);
            json!({
                "kind": "logical-array",
                "shape": arr.shape,
                "rows": rows,
                "cols": cols,
                "preview": preview.iter().map(|v| *v != 0).collect::<Vec<bool>>(),
                "length": arr.data.len(),
                "truncated": truncated,
            })
        }
        Value::String(s) => json!({
            "kind": "string",
            "value": s,
            "shape": vec![1, s.chars().count()],
        }),
        Value::StringArray(sa) => {
            let (preview, truncated) = preview_slice(&sa.data, MAX_DATA_PREVIEW);
            json!({
                "kind": "string-array",
                "shape": sa.shape,
                "rows": sa.rows,
                "cols": sa.cols,
                "preview": preview,
                "length": sa.data.len(),
                "truncated": truncated,
            })
        }
        Value::CharArray(ca) => {
            let s: String = ca.data.iter().copied().collect();
            json!({
                "kind": "char-array",
                "rows": ca.rows,
                "cols": ca.cols,
                "shape": vec![ca.rows, ca.cols],
                "text": s,
            })
        }
        Value::Tensor(t) => {
            let (preview, truncated) = preview_slice(&t.data, MAX_DATA_PREVIEW);
            json!({
                "kind": "tensor",
                "shape": t.shape,
                "rows": t.rows,
                "cols": t.cols,
                "dtype": match t.dtype {
                    NumericDType::F32 => "single",
                    NumericDType::F64 => "double",
                },
                "preview": preview,
                "length": t.data.len(),
                "truncated": truncated,
            })
        }
        Value::ComplexTensor(t) => {
            let (preview, truncated) = preview_slice(&t.data, MAX_DATA_PREVIEW);
            let preview: Vec<JsonValue> = preview
                .into_iter()
                .map(|(re, im)| json!({ "real": re, "imag": im }))
                .collect();
            json!({
                "kind": "complex-tensor",
                "shape": t.shape,
                "rows": t.rows,
                "cols": t.cols,
                "preview": preview,
                "length": t.data.len(),
                "truncated": truncated,
            })
        }
        Value::Cell(ca) => json!({
            "kind": "cell",
            "shape": ca.shape,
            "rows": ca.rows,
            "cols": ca.cols,
            "length": ca.data.len(),
        }),
        Value::Struct(st) => struct_to_json(st, depth + 1),
        Value::GpuTensor(handle) => {
            let (rows, cols) = rows_cols_from_shape(&handle.shape);
            json!({
                "kind": "gpu-tensor",
                "shape": handle.shape,
                "rows": rows,
                "cols": cols,
                "deviceId": handle.device_id,
                "bufferId": handle.buffer_id,
            })
        }
        Value::Object(obj) => object_to_json(obj, depth + 1),
        Value::HandleObject(handle) => json!({
            "kind": "handle",
            "className": handle.class_name,
            "valid": handle.valid,
        }),
        Value::Listener(listener) => json!({
            "kind": "listener",
            "id": listener.id,
            "event": listener.event_name,
            "enabled": listener.enabled,
            "valid": listener.valid,
        }),
        Value::FunctionHandle(name) => json!({
            "kind": "function",
            "name": name,
        }),
        Value::Closure(closure) => json!({
            "kind": "closure",
            "functionName": closure.function_name,
            "captureCount": closure.captures.len(),
        }),
        Value::ClassRef(name) => json!({
            "kind": "class-ref",
            "name": name,
        }),
        Value::MException(ex) => json!({
            "kind": "exception",
            "identifier": ex.identifier,
            "message": ex.message,
            "stack": ex.stack,
        }),
    }
}

fn struct_to_json(st: &StructValue, depth: usize) -> JsonValue {
    let mut fields = JsonMap::new();
    let mut truncated = false;
    for (idx, (name, field_value)) in st.fields.iter().enumerate() {
        if idx >= MAX_STRUCT_FIELDS {
            truncated = true;
            break;
        }
        fields.insert(name.clone(), value_to_json(field_value, depth));
    }
    json!({
        "kind": "struct",
        "fieldOrder": st.field_names().cloned().take(MAX_STRUCT_FIELDS).collect::<Vec<_>>(),
        "fields": fields,
        "totalFields": st.fields.len(),
        "truncated": truncated,
    })
}

fn object_to_json(obj: &ObjectInstance, depth: usize) -> JsonValue {
    let mut fields = JsonMap::new();
    let mut truncated = false;
    for (idx, (name, value)) in obj.properties.iter().enumerate() {
        if idx >= MAX_OBJECT_FIELDS {
            truncated = true;
            break;
        }
        fields.insert(name.clone(), value_to_json(value, depth));
    }
    json!({
        "kind": "object",
        "className": obj.class_name,
        "properties": fields,
        "propertyCount": obj.properties.len(),
        "truncated": truncated,
    })
}

fn scalar_shape() -> Vec<usize> {
    vec![1, 1]
}

fn rows_cols_from_shape(shape: &[usize]) -> (usize, usize) {
    let rows = shape.get(0).copied().unwrap_or(0);
    let cols = if shape.len() >= 2 {
        shape[1]
    } else if rows == 0 {
        0
    } else {
        1
    };
    (rows, cols)
}

fn preview_slice<T: Clone>(data: &[T], limit: usize) -> (Vec<T>, bool) {
    if data.len() > limit {
        (data[..limit].to_vec(), true)
    } else {
        (data.to_vec(), false)
    }
}

fn capture_gpu_adapter_info() -> Option<GpuAdapterInfo> {
    let provider = runmat_accelerate_api::provider()?;
    let info = provider.device_info_struct();
    let precision = provider_precision_label(provider.precision()).map(str::to_string);
    Some(GpuAdapterInfo {
        device_id: info.device_id,
        name: info.name,
        vendor: info.vendor,
        backend: info.backend,
        memory_bytes: info.memory_bytes,
        precision,
    })
}

fn provider_precision_label(precision: ProviderPrecision) -> Option<&'static str> {
    match precision {
        ProviderPrecision::F32 => Some("single"),
        ProviderPrecision::F64 => Some("double"),
    }
}
