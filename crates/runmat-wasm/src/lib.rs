#![cfg(target_arch = "wasm32")]

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, OnceLock};
use uuid::Uuid;

use js_sys::{Error as JsError, Reflect, Uint8Array};
use log::warn;
use miette::{SourceOffset, SourceSpan};
use runmat_accelerate::{
    initialize_acceleration_provider_with, AccelPowerPreference, AccelerateInitOptions,
    AccelerateProviderPreference, AutoOffloadOptions,
};
use runmat_accelerate_api::{AccelContextHandle, AccelContextKind, ProviderPrecision};
use runmat_builtins::{NumericDType, ObjectInstance, StructValue, Value};
use runmat_core::{
    matlab_class_name, value_shape, CompatMode, ExecutionProfiling, ExecutionResult,
    ExecutionStreamEntry, ExecutionStreamKind, FusionPlanDecision, FusionPlanEdge, FusionPlanNode,
    FusionPlanShader, FusionPlanSnapshot, InputRequest, InputRequestKind, InputResponse,
    MaterializedVariable, RunError, RunMatSession, StdinEvent, StdinEventKind, WorkspaceEntry,
    WorkspaceMaterializeOptions, WorkspaceMaterializeTarget, WorkspacePreview,
    WorkspaceSliceOptions, WorkspaceSnapshot,
};
use runmat_logging::{
    init_logging, set_runtime_log_hook, LoggingGuard, LoggingOptions, RuntimeLogRecord,
};
use runmat_runtime::build_runtime_error;
use runmat_thread_local::runmat_thread_local;
use serde_json::{json, Map as JsonMap, Value as JsonValue};
use std::backtrace::Backtrace;
use tracing::{info, info_span};
use runmat_core::{TelemetryPlatformInfo, TelemetryRunConfig, TelemetryRunFinish, TelemetrySink};
use runmat_telemetry::TelemetryRunKind;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use glam::Vec2;

#[cfg(target_arch = "wasm32")]
mod fs;
#[cfg(target_arch = "wasm32")]
use crate::fs::install_js_fs_provider;
#[cfg(target_arch = "wasm32")]
use runmat_plot::{
    event::{
        FigureEvent as PlotFigureEvent, FigureEventKind as PlotFigureEventKind, FigureSnapshot,
    },
    web::{WebCanvas, WebRenderer, WebRendererOptions},
    SharedWgpuContext,
};
#[cfg(target_arch = "wasm32")]
use runmat_runtime::builtins::plotting::{
    bind_surface_to_figure as runtime_bind_surface_to_figure, clear_figure as runtime_clear_figure,
    close_figure as runtime_close_figure, configure_subplot as runtime_configure_subplot,
    context as plotting_context, current_axes_state as runtime_current_axes_state,
    current_figure_handle as runtime_current_figure_handle,
    detach_surface as runtime_detach_surface, figure_handles as runtime_figure_handles,
    install_figure_observer as runtime_install_figure_observer,
    install_surface as runtime_install_surface, new_figure_handle as runtime_new_figure_handle,
    present_figure_on_surface as runtime_present_figure_on_surface,
    present_surface as runtime_present_surface,
    render_current_scene as runtime_render_current_scene,
    render_figure_snapshot as runtime_render_figure_snapshot,
    reset_hold_state_for_run as runtime_reset_hold_state_for_run,
    resize_surface as runtime_resize_surface, select_figure as runtime_select_figure,
    set_hold as runtime_set_hold, web_renderer_ready as runtime_plot_renderer_ready,
    handle_plot_surface_event as runtime_handle_plot_surface_event,
    fit_surface_extents as runtime_fit_surface_extents,
    reset_surface_camera as runtime_reset_surface_camera,
    FigureAxesState, FigureError, FigureEventKind, FigureEventView, FigureHandle, HoldMode,
};
#[cfg(target_arch = "wasm32")]
use runmat_runtime::builtins::{
    plotting::{set_scatter_target_points, set_surface_vertex_budget},
    wasm_registry,
};
use runmat_runtime::warning_store::RuntimeWarning;
#[cfg(target_arch = "wasm32")]
use runmat_runtime::RuntimeError;
use serde::{Deserialize, Serialize};

const MAX_DATA_PREVIEW: usize = 4096;
const MAX_STRUCT_FIELDS: usize = 64;
const MAX_OBJECT_FIELDS: usize = 64;
const MAX_OUTPUT_LIST_ITEMS: usize = 64;

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
    #[serde(default, rename = "telemetryRunKind")]
    telemetry_run_kind: Option<String>,
    #[serde(default)]
    emit_fusion_plan: Option<bool>,
    #[serde(default)]
    language_compat: Option<String>,
    #[serde(default)]
    log_level: Option<String>,
    #[serde(default)]
    gpu_buffer_pool_max_per_key: Option<u32>,
    #[serde(default)]
    callstack_limit: Option<usize>,
    #[serde(default)]
    error_namespace: Option<String>,
    #[cfg(target_arch = "wasm32")]
    #[serde(skip)]
    telemetry_emitter: Option<JsValue>,
}

runmat_thread_local! {
    static FIGURE_EVENT_CALLBACK: RefCell<Option<js_sys::Function>> = RefCell::new(None);
}
static FIGURE_EVENT_OBSERVER: OnceLock<()> = OnceLock::new();
runmat_thread_local! {
    static JS_STDIN_HANDLER: RefCell<Option<js_sys::Function>> = RefCell::new(None);
}
runmat_thread_local! {
    static STDOUT_SUBSCRIBERS: RefCell<HashMap<u32, js_sys::Function>> =
        RefCell::new(HashMap::new());
}
type StdoutForwarder = Arc<dyn Fn(&runmat_runtime::console::ConsoleEntry) + Send + Sync + 'static>;
type RuntimeLogForwarder = Arc<dyn Fn(&runmat_logging::RuntimeLogRecord) + Send + Sync + 'static>;
type TraceForwarder = Arc<dyn Fn(&[runmat_logging::TraceEvent]) + Send + Sync + 'static>;

static STDOUT_FORWARDER: OnceLock<StdoutForwarder> = OnceLock::new();
static STDOUT_NEXT_ID: AtomicU32 = AtomicU32::new(1);

runmat_thread_local! {
    static RUNTIME_LOG_SUBSCRIBERS: RefCell<HashMap<u32, js_sys::Function>> =
        RefCell::new(HashMap::new());
}
static RUNTIME_LOG_FORWARDER: OnceLock<RuntimeLogForwarder> = OnceLock::new();
static RUNTIME_LOG_NEXT_ID: AtomicU32 = AtomicU32::new(1);

runmat_thread_local! {
    static TRACE_SUBSCRIBERS: RefCell<HashMap<u32, js_sys::Function>> =
        RefCell::new(HashMap::new());
}
static TRACE_FORWARDER: OnceLock<TraceForwarder> = OnceLock::new();
static TRACE_NEXT_ID: AtomicU32 = AtomicU32::new(1);
static LOGGING_GUARD: OnceLock<LoggingGuard> = OnceLock::new();
static LOG_FILTER_OVERRIDE: OnceLock<String> = OnceLock::new();

// Plot surface registry for web backends.
static PLOT_SURFACE_NEXT_ID: AtomicU32 = AtomicU32::new(1);
runmat_thread_local! {
    // Back-compat helpers: old API keyed by handle/plot canvas, implemented on top of surface ids.
    static LEGACY_PLOT_SURFACE_ID: RefCell<Option<u32>> = RefCell::new(None);
    static LEGACY_FIGURE_SURFACES: RefCell<HashMap<u32, u32>> = RefCell::new(HashMap::new());
}

#[derive(Clone)]
struct SessionConfig {
    enable_jit: bool,
    verbose: bool,
    telemetry_consent: bool,
    telemetry_client_id: Option<String>,
    telemetry_run_kind: TelemetryRunKind,
    enable_gpu: bool,
    wgpu_power_preference: AccelPowerPreference,
    wgpu_force_fallback_adapter: bool,
    auto_offload: AutoOffloadOptions,
    emit_fusion_plan: bool,
    language_compat: CompatMode,
    gpu_buffer_pool_max_per_key: Option<u32>,
    callstack_limit: usize,
    error_namespace: String,
}

impl SessionConfig {
    fn from_options(opts: &InitOptions) -> Self {
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
                .unwrap_or(runmat_ignition::DEFAULT_CALLSTACK_LIMIT),
            error_namespace: opts
                .error_namespace
                .clone()
                .unwrap_or_else(|| runmat_ignition::DEFAULT_ERROR_NAMESPACE.to_string()),
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

    fn apply_env_overrides(&self) {
        if let Some(max) = self.gpu_buffer_pool_max_per_key {
            log::info!("RunMat wasm: setting RUNMAT_WGPU_POOL_MAX_PER_KEY={}", max);
            std::env::set_var("RUNMAT_WGPU_POOL_MAX_PER_KEY", max.to_string());
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
    active_interrupt: RefCell<Option<Arc<std::sync::atomic::AtomicBool>>>,
    telemetry_sink: Option<Arc<dyn TelemetrySink>>,
}

#[wasm_bindgen]
impl RunMatWasm {
    #[wasm_bindgen(js_name = execute)]
    pub async fn execute(&self, source: String) -> Result<JsValue, JsValue> {
        init_logging_once();
        let exec_span = info_span!(
            "runmat.execute",
            source_len = source.len() as u64,
            disposed = self.disposed.get()
        );
        let _enter = exec_span.enter();
        info!(target = "runmat.runtime", "Execution started");
        // Capture figure handles before execution so we can close stale figures after a successful run.
        let figures_before: Vec<u32> = runtime_figure_handles()
            .into_iter()
            .map(|handle| handle.as_u32())
            .collect();
        // Reset hold state so a previous `hold on` doesn't cause subsequent runs to keep appending.
        runtime_reset_hold_state_for_run();

        let telemetry_run = {
            if self.disposed.get() {
                None
            } else {
                let cfg = self.config.borrow();
                if !cfg.telemetry_consent {
                    None
                } else {
                    self.session.borrow().telemetry_run(TelemetryRunConfig {
                        kind: cfg.telemetry_run_kind.clone(),
                        jit_enabled: cfg.enable_jit,
                        accelerate_enabled: self.gpu_status.active,
                    })
                }
            }
        };

        // Ensure `cancelExecution()` can interrupt the *active* session even while we temporarily
        // move it out of `self.session` to avoid holding a RefCell borrow across `.await`.
        let interrupt_handle = self.session.borrow().interrupt_handle();
        self.active_interrupt
            .borrow_mut()
            .replace(Arc::clone(&interrupt_handle));
        struct ActiveInterruptGuard<'a> {
            slot: &'a RefCell<Option<Arc<std::sync::atomic::AtomicBool>>>,
        }
        impl Drop for ActiveInterruptGuard<'_> {
            fn drop(&mut self) {
                self.slot.borrow_mut().take();
            }
        }
        let _active_interrupt_guard = ActiveInterruptGuard {
            slot: &self.active_interrupt,
        };

        // We must not hold a RefCell borrow across `.await`, so temporarily move the session out.
        let mut session = {
            let mut slot = self.session.borrow_mut();
            std::mem::take(&mut *slot)
        };

        let exec_result = session.execute(&source).await;
        // Always restore the session, even if execution fails (e.g. parse/compile error).
        *self.session.borrow_mut() = session;
        let payload = match exec_result {
            Ok(result) => {
                // When a run succeeds, close figures that existed before the run but were
                // not touched during it. This avoids stale plots lingering across runs while preserving
                // GPU surfaces for the figures that remain active.
                if result.error.is_none() {
                    let touched: std::collections::HashSet<u32> =
                        result.figures_touched.iter().copied().collect();
                    for handle in figures_before {
                        if !touched.contains(&handle) {
                            let _ = runtime_close_figure(Some(FigureHandle::from(handle)));
                        }
                    }
                }
                ExecutionPayload::from_result(result, &source)
            }
            Err(err) => ExecutionPayload {
                value_text: None,
                value_json: None,
                type_info: None,
                execution_time_ms: 0,
                used_jit: false,
                error: Some(run_error_payload(&err, &source)),
                stdout: Vec::new(),
                workspace: WorkspacePayload {
                    full: false,
                    version: 0,
                    values: Vec::new(),
                },
                figures_touched: Vec::new(),
                warnings: Vec::new(),
                stdin_events: Vec::new(),
                profiling: None,
                fusion_plan: None,
            },
        };
        info!(
            target = "runmat.runtime",
            workspace_entries = payload.workspace.values.len(),
            stdout_entries = payload.stdout.len(),
            figures_touched = payload.figures_touched.len(),
            used_jit = payload.used_jit,
            error = payload
                .error
                .as_ref()
                .map(|err| err.message.as_str())
                .unwrap_or(""),
            "Execution finished"
        );
        if let Some(run) = telemetry_run {
            let duration = std::time::Duration::from_millis(payload.execution_time_ms);
            let (success, error) = match payload.error.as_ref() {
                None => (true, None),
                Some(err) => (
                    false,
                    err.identifier
                        .as_deref()
                        .map(|s| s.to_string())
                        .or_else(|| Some("runtime_error".to_string())),
                ),
            };
            run.finish(TelemetryRunFinish {
                duration: Some(duration),
                success,
                jit_used: payload.used_jit,
                error,
                counters: None,
                provider: None,
            });
        }
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
        session.set_telemetry_consent(consent);
        if let Some(cid) = config.telemetry_client_id.clone() {
            session.set_telemetry_client_id(Some(cid));
        } else {
            session.set_telemetry_client_id(None);
        }
        session.set_emit_fusion_plan(config.emit_fusion_plan);
        session.set_compat_mode(config.language_compat);
        session.set_callstack_limit(config.callstack_limit);
        session.set_error_namespace(config.error_namespace.clone());
        session.set_source_name_override(Some("<wasm>".to_string()));
        if self.telemetry_sink.is_some() {
            session.set_telemetry_platform_info(TelemetryPlatformInfo {
                os: Some("web".to_string()),
                arch: Some("wasm32".to_string()),
            });
            session.set_telemetry_sink(self.telemetry_sink.clone());
        }
        let mut slot = self.session.borrow_mut();
        *slot = session;
        Ok(())
    }

    #[wasm_bindgen(js_name = cancelExecution)]
    pub fn cancel_execution(&self) {
        if let Some(flag) = self.active_interrupt.borrow().as_ref() {
            flag.store(true, std::sync::atomic::Ordering::Relaxed);
            return;
        }
        self.session.borrow().cancel_execution();
    }

    #[wasm_bindgen(js_name = cancelPendingRequests)]
    pub fn cancel_pending_requests(&self) {
        // Phase 2: stdin + internal suspensions are awaited inside `ExecuteFuture`.
        // Keep this API as a no-op for now to avoid breaking older hosts.
    }

    #[wasm_bindgen(js_name = "setLanguageCompat")]
    pub fn set_language_compat(&self, mode: String) {
        if self.disposed.get() {
            return;
        }
        if let Some(parsed) = parse_language_compat_from_str(&mode) {
            {
                let mut config = self.config.borrow_mut();
                config.language_compat = parsed;
            }
            self.session.borrow_mut().set_compat_mode(parsed);
        } else {
            warn!("RunMat wasm: ignoring unknown language compat mode '{mode}'");
        }
    }

    #[wasm_bindgen(js_name = setInputHandler)]
    pub fn set_input_handler(&self, handler: JsValue) -> Result<(), JsValue> {
        #[cfg(target_arch = "wasm32")]
        {
            if handler.is_null() || handler.is_undefined() {
                set_js_stdin_handler(None);
                let mut session = self.session.borrow_mut();
                session.clear_async_input_handler();
            } else {
                let func = handler
                    .dyn_into::<js_sys::Function>()
                    .map_err(|_| js_error("setInputHandler expects a Function or null"))?;
                set_js_stdin_handler(Some(func));
                let mut session = self.session.borrow_mut();
                session
                    .install_async_input_handler(|req| async move { js_input_request(req).await });
            }
            Ok(())
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = handler;
            Err(js_error(
                "setInputHandler is only available when targeting wasm32",
            ))
        }
    }

    // Pending/resume plumbing has been removed in favor of poll-driven `ExecuteFuture`.

    #[wasm_bindgen(js_name = setFusionPlanEnabled)]
    pub fn set_fusion_plan_enabled(&self, enabled: bool) {
        self.session.borrow_mut().set_emit_fusion_plan(enabled);
        if let Ok(mut cfg) = self.config.try_borrow_mut() {
            cfg.emit_fusion_plan = enabled;
        }
    }

    /// Compile-only fusion plan snapshot (no execution).
    #[wasm_bindgen(js_name = fusionPlanForSource)]
    pub fn fusion_plan_for_source(&self, source: String) -> Result<JsValue, JsValue> {
        let mut session = self.session.borrow_mut();
        let snapshot = session
            .compile_fusion_plan(&source)
            .map_err(|err| run_error_to_js(&err, &source))?;
        match snapshot {
            Some(plan) => serde_wasm_bindgen::to_value(&FusionPlanPayload::from(plan))
                .map_err(|err| js_error(&format!("Failed to serialize fusion plan: {err}"))),
            None => Ok(JsValue::NULL),
        }
    }

    #[wasm_bindgen(js_name = materializeVariable)]
    pub async fn materialize_variable(
        &self,
        selector: JsValue,
        options: JsValue,
    ) -> Result<JsValue, JsValue> {
        #[cfg(target_arch = "wasm32")]
        {
            let target = parse_materialize_target(selector)?;
            let opts = parse_materialize_options(options)?;
            // We must not hold a RefCell borrow across `.await`, so temporarily move the session out.
            let mut session = {
                let mut slot = self.session.borrow_mut();
                std::mem::take(&mut *slot)
            };

            let materialize_result = session.materialize_variable(target, opts).await;
            // Always restore the session, even if materialization fails.
            *self.session.borrow_mut() = session;

            let value = materialize_result
                .map_err(|err| js_error(&format!("materializeVariable failed: {err}")))?;
            let payload = MaterializedVariablePayload::from(value);
            serde_wasm_bindgen::to_value(&payload).map_err(|err| {
                js_error(&format!(
                    "Failed to serialize materialized workspace value: {err}"
                ))
            })
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
pub async fn register_plot_canvas(canvas: JsValue) -> Result<(), JsValue> {
    let canvas = parse_web_canvas(canvas)?;
    let surface_id = PLOT_SURFACE_NEXT_ID.fetch_add(1, Ordering::Relaxed);
    install_surface_renderer(surface_id, canvas)
        .await
        .map_err(|err| {
            init_error_with_details(
                InitErrorCode::PlotCanvas,
                "Failed to register plot canvas",
                Some(err),
            )
        })?;
    LEGACY_PLOT_SURFACE_ID.with(|slot| {
        slot.borrow_mut().replace(surface_id);
    });
    Ok(())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = registerFigureCanvas)]
pub async fn register_figure_canvas(handle: u32, canvas: JsValue) -> Result<(), JsValue> {
    let canvas = parse_web_canvas(canvas)?;
    let surface_id = PLOT_SURFACE_NEXT_ID.fetch_add(1, Ordering::Relaxed);
    install_surface_renderer(surface_id, canvas)
        .await
        .map_err(|err| {
            init_error_with_details(
                InitErrorCode::PlotCanvas,
                "Failed to register figure canvas",
                Some(err),
            )
        })?;
    runtime_bind_surface_to_figure(surface_id, handle).map_err(|err| js_error(err.message()))?;
    LEGACY_FIGURE_SURFACES.with(|slot| {
        slot.borrow_mut().insert(handle, surface_id);
    });
    // Prime immediately if the figure already exists.
    let _ = runtime_render_current_scene(handle);
    Ok(())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = deregisterPlotCanvas)]
pub fn deregister_plot_canvas() {
    let surface_id = LEGACY_PLOT_SURFACE_ID.with(|slot| slot.borrow_mut().take());
    if let Some(id) = surface_id {
        runtime_detach_surface(id);
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = deregisterFigureCanvas)]
pub fn deregister_figure_canvas(handle: u32) {
    let surface_id = LEGACY_FIGURE_SURFACES.with(|slot| slot.borrow_mut().remove(&handle));
    if let Some(id) = surface_id {
        runtime_detach_surface(id);
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = resizeFigureCanvas)]
pub fn resize_figure_canvas(handle: u32, width: u32, height: u32) -> Result<(), JsValue> {
    let surface_id = LEGACY_FIGURE_SURFACES.with(|slot| slot.borrow().get(&handle).copied());
    let Some(surface_id) = surface_id else {
        return Err(js_error("Figure canvas not registered"));
    };
    // Legacy API has no access to devicePixelRatio; assume 1.0.
    runtime_resize_surface(surface_id, width.max(1), height.max(1), 1.0)
        .map_err(|err| js_error(err.message()))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = renderCurrentFigureScene)]
pub fn render_current_figure_scene(handle: u32) -> Result<(), JsValue> {
    runtime_render_current_scene(handle).map_err(|err| js_error(err.message()))
}

// New surface-id based API.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = createPlotSurface)]
pub async fn create_plot_surface(canvas: JsValue) -> Result<u32, JsValue> {
    let canvas = parse_web_canvas(canvas)?;
    let surface_id = PLOT_SURFACE_NEXT_ID.fetch_add(1, Ordering::Relaxed);
    install_surface_renderer(surface_id, canvas).await?;
    Ok(surface_id)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = destroyPlotSurface)]
pub fn destroy_plot_surface(surface_id: u32) {
    runtime_detach_surface(surface_id);
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = resizePlotSurface)]
pub fn resize_plot_surface(
    surface_id: u32,
    width: u32,
    height: u32,
    pixels_per_point: f32,
) -> Result<(), JsValue> {
    runtime_resize_surface(surface_id, width.max(1), height.max(1), pixels_per_point)
        .map_err(|err| js_error(err.message()))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = bindSurfaceToFigure)]
pub fn bind_surface_to_figure(surface_id: u32, handle: u32) -> Result<(), JsValue> {
    runtime_bind_surface_to_figure(surface_id, handle).map_err(|err| js_error(err.message()))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = presentSurface)]
pub fn present_surface(surface_id: u32) -> Result<(), JsValue> {
    runtime_present_surface(surface_id).map_err(|err| js_error(err.message()))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = presentFigureOnSurface)]
pub fn present_figure_on_surface(surface_id: u32, handle: u32) -> Result<(), JsValue> {
    runtime_present_figure_on_surface(surface_id, handle).map_err(|err| js_error(err.message()))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PlotSurfaceEventPayload {
    kind: String,
    x: f32,
    y: f32,
    #[serde(default)]
    dx: f32,
    #[serde(default)]
    dy: f32,
    #[serde(default)]
    button: i32,
    #[serde(default)]
    wheel_delta_x: f32,
    #[serde(default)]
    wheel_delta_y: f32,
    #[serde(default)]
    wheel_delta_mode: u32,
    #[serde(default)]
    shift_key: bool,
    #[serde(default)]
    ctrl_key: bool,
    #[serde(default)]
    alt_key: bool,
    #[serde(default)]
    meta_key: bool,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = handlePlotSurfaceEvent)]
pub fn handle_plot_surface_event(surface_id: u32, event: JsValue) -> Result<(), JsValue> {
    use runmat_plot::core::interaction::MouseButton as PlotMouseButton;
    use runmat_plot::core::interaction::Modifiers as PlotModifiers;
    use runmat_plot::core::PlotEvent;

    let payload: PlotSurfaceEventPayload =
        serde_wasm_bindgen::from_value(event).map_err(|err| js_error(&err.to_string()))?;
    let position = Vec2::new(payload.x, payload.y);
    let delta = Vec2::new(payload.dx, payload.dy);
    let button = match payload.button {
        2 => PlotMouseButton::Right,
        1 => PlotMouseButton::Middle,
        _ => PlotMouseButton::Left,
    };
    let modifiers = PlotModifiers {
        shift: payload.shift_key,
        ctrl: payload.ctrl_key,
        alt: payload.alt_key,
        meta: payload.meta_key,
    };

    let event = match payload.kind.as_str() {
        "mouseDown" => PlotEvent::MousePress {
            position,
            button,
            modifiers,
        },
        "mouseUp" => PlotEvent::MouseRelease {
            position,
            button,
            modifiers,
        },
        "mouseMove" => PlotEvent::MouseMove {
            position,
            delta,
            modifiers,
        },
        "wheel" => {
            // Normalize DOM wheel delta (pixel/line/page) into a roughly "lines" scale.
            // - Pixel deltas (trackpads) tend to be large; scale them down.
            // - Page deltas are rare; scale them up.
            let mut wheel_delta_x = payload.wheel_delta_x;
            let mut wheel_delta_y = payload.wheel_delta_y;
            match payload.wheel_delta_mode {
                0 => {
                    // pixels
                    wheel_delta_x /= 100.0;
                    wheel_delta_y /= 100.0;
                }
                1 => {
                    // lines
                }
                2 => {
                    // pages
                    wheel_delta_x *= 10.0;
                    wheel_delta_y *= 10.0;
                }
                _ => {}
            }
            // Align with the native path + common CAD conventions:
            // positive delta = zoom in (wheel up / away from user).
            wheel_delta_x = -wheel_delta_x;
            wheel_delta_y = -wheel_delta_y;
            PlotEvent::MouseWheel {
                position,
                delta: Vec2::new(wheel_delta_x, wheel_delta_y),
                modifiers,
            }
        }
        other => {
            let message = format!("Unknown plot event kind '{other}'");
            return Err(js_error(&message));
        }
    };

    runtime_handle_plot_surface_event(surface_id, event).map_err(|err| js_error(err.message()))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = "fitPlotSurfaceExtents")]
pub fn fit_plot_surface_extents(surface_id: u32) -> Result<(), JsValue> {
    runtime_fit_surface_extents(surface_id).map_err(|err| js_error(err.message()))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = "resetPlotSurfaceCamera")]
pub fn reset_plot_surface_camera(surface_id: u32) -> Result<(), JsValue> {
    runtime_reset_surface_camera(surface_id).map_err(|err| js_error(err.message()))
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
        .map_err(runtime_flow_to_js)?;
    Ok(Uint8Array::from(bytes.as_slice()))
}

#[cfg(target_arch = "wasm32")]
fn shared_webgpu_context() -> Option<SharedWgpuContext> {
    if let Some(ctx) = runmat_plot::shared_wgpu_context() {
        log::debug!("plot-web: shared_webgpu_context: using existing runmat_plot shared context");
        return Some(ctx);
    }

    let api_provider_present = runmat_accelerate_api::provider().is_some();
    log::debug!(
        "plot-web: shared_webgpu_context: no plot context installed yet (api_provider_present={})",
        api_provider_present
    );

    let handle = match runmat_accelerate_api::export_context(AccelContextKind::Plotting) {
        Some(handle) => handle,
        None => {
            log::debug!(
                "plot-web: shared_webgpu_context: export_context(Plotting) returned None (api_provider_present={})",
                api_provider_present
            );
            return None;
        }
    };
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
            log::debug!("plot-web: shared_webgpu_context: installed shared context from accelerate provider (adapter={:?})", shared.adapter_info);
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
#[wasm_bindgen(js_name = setLogFilter)]
pub fn set_log_filter(filter: &str) -> Result<(), JsValue> {
    init_logging_once();
    runmat_logging::update_log_filter(filter).map_err(|err| js_error(&err))
}

#[cfg(not(target_arch = "wasm32"))]
#[wasm_bindgen(js_name = setLogFilter)]
pub fn set_log_filter(_filter: &str) -> Result<(), JsValue> {
    Err(js_error(
        "setLogFilter is only available when targeting wasm32",
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
#[wasm_bindgen(js_name = subscribeTraceEvents)]
pub fn subscribe_trace_events(callback: JsValue) -> Result<u32, JsValue> {
    init_logging_once();
    let function = callback
        .dyn_into::<js_sys::Function>()
        .map_err(|_| js_error("subscribeTraceEvents expects a Function"))?;
    ensure_trace_forwarder_installed();
    let id = TRACE_NEXT_ID.fetch_add(1, Ordering::Relaxed);
    TRACE_SUBSCRIBERS.with(|cell| {
        cell.borrow_mut().insert(id, function);
    });
    Ok(id)
}

#[cfg(not(target_arch = "wasm32"))]
#[wasm_bindgen(js_name = subscribeTraceEvents)]
pub fn subscribe_trace_events(_callback: JsValue) -> Result<u32, JsValue> {
    Err(js_error(
        "subscribeTraceEvents is only available when targeting wasm32",
    ))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = unsubscribeTraceEvents)]
pub fn unsubscribe_trace_events(id: u32) {
    TRACE_SUBSCRIBERS.with(|cell| {
        cell.borrow_mut().remove(&id);
    });
}

#[cfg(not(target_arch = "wasm32"))]
#[wasm_bindgen(js_name = unsubscribeTraceEvents)]
pub fn unsubscribe_trace_events(_id: u32) {}

#[cfg(target_arch = "wasm32")]
fn set_js_stdin_handler(handler: Option<js_sys::Function>) {
    JS_STDIN_HANDLER.with(|slot| *slot.borrow_mut() = handler);
}

#[cfg(target_arch = "wasm32")]
async fn js_input_request(request: InputRequest) -> Result<InputResponse, String> {
    let handler = JS_STDIN_HANDLER.with(|slot| slot.borrow().clone());
    let Some(handler) = handler else {
        return Err("stdin requested but no input handler is installed".to_string());
    };

    let js_request = js_sys::Object::new();
    Reflect::set(
        &js_request,
        &JsValue::from_str("prompt"),
        &JsValue::from_str(&request.prompt),
    )
    .map_err(js_value_to_string)?;

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
        }
        InputRequestKind::KeyPress => {
            Reflect::set(
                &js_request,
                &JsValue::from_str("kind"),
                &JsValue::from_str("keyPress"),
            )
            .unwrap_or_default();
        }
    }

    let mut value = handler
        .call1(&JsValue::NULL, &js_request)
        .map_err(js_value_to_string)?;

    if value.is_instance_of::<js_sys::Promise>() {
        value = JsFuture::from(js_sys::Promise::from(value))
            .await
            .map_err(js_value_to_string)?;
    }

    if let Some(err) = extract_error_message(&value) {
        return Err(err);
    }

    match request.kind {
        InputRequestKind::Line { .. } => {
            // Accept null/undefined as an empty line.
            if value.is_null() || value.is_undefined() {
                return Ok(InputResponse::Line(String::new()));
            }
            if let Some(text) = extract_line_value(&value) {
                return Ok(InputResponse::Line(text));
            }
            Err(
                "stdin handler must return a string (or Promise of a string) for line input"
                    .to_string(),
            )
        }
        InputRequestKind::KeyPress => Ok(InputResponse::KeyPress),
    }
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
async fn install_surface_renderer(surface_id: u32, canvas: WebCanvas) -> Result<(), JsValue> {
    init_logging_once();
    let options = WebRendererOptions::default();
    let canvas_kind = match &canvas {
        WebCanvas::Html(_) => "html",
        WebCanvas::Offscreen(_) => "offscreen",
    };
    log::debug!(
        "plot-web: install_surface_renderer(surface_id={surface_id}, canvas_kind={})",
        canvas_kind
    );
    let renderer = match shared_webgpu_context() {
        Some(shared) => {
            WebRenderer::with_shared_context(canvas.clone(), options.clone(), shared).await
        }
        None => WebRenderer::new(canvas, options).await,
    }
    .map_err(|err| js_error(&format!("Failed to initialize plot renderer: {err}")))?;
    runtime_install_surface(surface_id, renderer)
        .map_err(|err| js_error(&format!("Failed to register plot surface: {err}")))?;
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn parse_web_canvas(canvas: JsValue) -> Result<WebCanvas, JsValue> {
    if canvas.is_null() || canvas.is_undefined() {
        return Err(js_error("Canvas is required"));
    }
    if let Ok(html) = canvas.clone().dyn_into::<web_sys::HtmlCanvasElement>() {
        return Ok(WebCanvas::Html(html));
    }
    if let Ok(offscreen) = canvas.clone().dyn_into::<web_sys::OffscreenCanvas>() {
        return Ok(WebCanvas::Offscreen(offscreen));
    }
    Err(js_error("Expected an HTMLCanvasElement or OffscreenCanvas"))
}

// Figure priming is handled by `presentFigureOnSurface` / `renderCurrentFigureScene` which
// load the current figure snapshot based on the bound handle.

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
            if let Ok(stream_value) = Reflect::get(&options, &JsValue::from_str("snapshotStream")) {
                if !stream_value.is_null() && !stream_value.is_undefined() {
                    parsed_opts.snapshot_stream = Some(stream_value);
                }
            }
            if let Ok(emitter_value) = Reflect::get(&options, &JsValue::from_str("telemetryEmitter")) {
                if !emitter_value.is_null() && !emitter_value.is_undefined() {
                    parsed_opts.telemetry_emitter = Some(emitter_value);
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
    config.apply_env_overrides();
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
                    if let Err(err) = plotting_context::ensure_context_from_provider() {
                        let message = err.message().to_string();
                        warn!("RunMat wasm: unable to install shared plotting context: {message}");
                        gpu_status.error = Some(message);
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

    let instance = RunMatWasm {
        session: RefCell::new(session),
        snapshot_seed,
        config: RefCell::new(config),
        gpu_status,
        disposed: Cell::new(false),
        active_interrupt: RefCell::new(None),
        telemetry_sink,
    };
    Ok(instance)
}

#[cfg(target_arch = "wasm32")]
fn ensure_getrandom_js() {
    let mut buf = [0u8; 1];
    if let Err(err) = getrandom::getrandom(&mut buf) {
        warn!("RunMat wasm: failed to initialize JS randomness source: {err}");
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

fn apply_plotting_overrides(opts: &InitOptions) {
    if let Some(points) = opts.scatter_target_points {
        set_scatter_target_points(points);
    }
    if let Some(budget) = opts.surface_vertex_budget {
        set_surface_vertex_budget(budget);
    }
}

fn parse_language_compat(input: Option<&str>) -> CompatMode {
    input
        .and_then(parse_language_compat_from_str)
        .unwrap_or(CompatMode::Matlab)
}

fn parse_telemetry_run_kind(value: Option<&str>) -> TelemetryRunKind {
    match value.unwrap_or("repl").trim().to_ascii_lowercase().as_str() {
        "script" => TelemetryRunKind::Script,
        "benchmark" => TelemetryRunKind::Benchmark,
        "install" => TelemetryRunKind::Install,
        _ => TelemetryRunKind::Repl,
    }
}

fn parse_language_compat_from_str(value: &str) -> Option<CompatMode> {
    if value.eq_ignore_ascii_case("strict") {
        Some(CompatMode::Strict)
    } else if value.eq_ignore_ascii_case("matlab") {
        Some(CompatMode::Matlab)
    } else {
        None
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
            Arc::new(emit_js_figure_event);
        let _ = runtime_install_figure_observer(observer);
    });
}

#[cfg(target_arch = "wasm32")]
fn ensure_runtime_log_forwarder_installed() {
    RUNTIME_LOG_FORWARDER.get_or_init(|| {
        let forwarder: RuntimeLogForwarder = Arc::new(|record: &RuntimeLogRecord| {
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

#[cfg(target_arch = "wasm32")]
fn ensure_trace_forwarder_installed() {
    use runmat_logging::{set_trace_hook, TraceEvent};

    TRACE_FORWARDER.get_or_init(|| {
        let forwarder: TraceForwarder = Arc::new(|events: &[TraceEvent]| {
            let js_value = serde_wasm_bindgen::to_value(events).unwrap_or(JsValue::NULL);
            TRACE_SUBSCRIBERS.with(|cell| {
                for cb in cell.borrow().values() {
                    let _ = cb.call1(&JsValue::NULL, &js_value);
                }
            });
        });
        let hook = forwarder.clone();
        set_trace_hook(move |events| {
            (hook)(events);
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
        FigureError::RenderFailure { source } => {
            let details = source.to_string();
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
fn runtime_flow_to_js(err: RuntimeError) -> JsValue {
    serde_wasm_bindgen::to_value(&runtime_error_payload(&err, None))
        .unwrap_or_else(|_| js_error(err.message()))
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
        let handle = event.handle.as_u32();
        // Legacy API cleanup: if a figure-specific canvas was registered via the old handle-based
        // API, detach its surface when the figure is closed.
        let surface_id = LEGACY_FIGURE_SURFACES.with(|slot| slot.borrow_mut().remove(&handle));
        if let Some(id) = surface_id {
            runtime_detach_surface(id);
        }
    }
    FIGURE_EVENT_CALLBACK.with(|slot| {
        if let Some(cb) = slot.borrow().as_ref() {
            let payload = convert_event_view(event);
            let js_value = serde_wasm_bindgen::to_value(&payload).unwrap_or(JsValue::UNDEFINED);
            let _ = cb.call1(&JsValue::NULL, &js_value);
        }
    });
}

#[cfg(target_arch = "wasm32")]
fn convert_event_view(view: FigureEventView<'_>) -> PlotFigureEvent {
    PlotFigureEvent {
        handle: view.handle.as_u32(),
        kind: match view.kind {
            FigureEventKind::Created => PlotFigureEventKind::Created,
            FigureEventKind::Updated => PlotFigureEventKind::Updated,
            FigureEventKind::Cleared => PlotFigureEventKind::Cleared,
            FigureEventKind::Closed => PlotFigureEventKind::Closed,
        },
        figure: view.figure.map(FigureSnapshot::capture),
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

fn line_col_from_offset(source: &str, offset: usize) -> (usize, usize) {
    let mut line = 1;
    let mut line_start = 0;
    for (idx, ch) in source.char_indices() {
        if idx >= offset {
            break;
        }
        if ch == '\n' {
            line += 1;
            line_start = idx + 1;
        }
    }
    let col = offset.saturating_sub(line_start) + 1;
    (line, col)
}

fn span_payload_from_source(source: &str, start: usize, end: usize) -> RunMatErrorSpanPayload {
    let (line, column) = line_col_from_offset(source, start);
    RunMatErrorSpanPayload {
        start,
        end,
        line,
        column,
    }
}

fn format_run_error(err: &RunError, source: &str) -> String {
    match err {
        RunError::Syntax(err) => {
            let mut message = err.message.clone();
            if let Some(expected) = &err.expected {
                message = format!("{message} (expected {expected})");
            }
            if let Some(found) = &err.found_token {
                message = format!("{message} (found '{found}')");
            }
            let span = SourceSpan::new(SourceOffset::from(err.position), 1);
            build_runtime_error(message)
                .with_identifier("RunMat:SyntaxError")
                .with_span(span)
                .build()
                .format_diagnostic_with_source(Some("<wasm>"), Some(source))
        }
        RunError::Semantic(err) => {
            let span = err.span.map(|span| {
                SourceSpan::new(
                    SourceOffset::from(span.start),
                    span.end.saturating_sub(span.start).max(1),
                )
            });
            let mut builder = build_runtime_error(err.message.clone());
            if let Some(identifier) = err.identifier.as_deref() {
                builder = builder.with_identifier(identifier);
            }
            if let Some(span) = span {
                builder = builder.with_span(span);
            }
            builder
                .build()
                .format_diagnostic_with_source(Some("<wasm>"), Some(source))
        }
        RunError::Compile(err) => {
            let span = err.span.map(|span| {
                SourceSpan::new(
                    SourceOffset::from(span.start),
                    span.end.saturating_sub(span.start).max(1),
                )
            });
            let mut builder = build_runtime_error(err.message.clone());
            if let Some(identifier) = err.identifier.as_deref() {
                builder = builder.with_identifier(identifier);
            }
            if let Some(span) = span {
                builder = builder.with_span(span);
            }
            builder
                .build()
                .format_diagnostic_with_source(Some("<wasm>"), Some(source))
        }
        RunError::Runtime(err) => err.format_diagnostic_with_source(Some("<wasm>"), Some(source)),
    }
}

fn runtime_error_payload(err: &RuntimeError, source: Option<&str>) -> RunMatErrorPayload {
    let identifier = err.identifier().map(|id| id.to_string());
    let span = match (source, err.span.as_ref()) {
        (Some(source), Some(span)) => {
            let start = span.offset();
            let end = start + span.len();
            Some(span_payload_from_source(source, start, end))
        }
        _ => None,
    };
    let diagnostic = match source {
        Some(source) => err.format_diagnostic_with_source(Some("<wasm>"), Some(source)),
        None => err.format_diagnostic(),
    };
    let callstack = if !err.context.call_stack.is_empty() {
        err.context.call_stack.clone()
    } else {
        err.context
            .call_frames
            .iter()
            .map(|frame| frame.function.clone())
            .collect()
    };
    RunMatErrorPayload {
        kind: RunMatErrorKind::Runtime,
        message: err.message().to_string(),
        identifier,
        diagnostic,
        span,
        callstack,
        callstack_elided: err.context.call_frames_elided,
    }
}

fn run_error_payload(err: &RunError, source: &str) -> RunMatErrorPayload {
    let diagnostic = format_run_error(err, source);
    match err {
        RunError::Syntax(err) => {
            let mut message = err.message.clone();
            if let Some(expected) = &err.expected {
                message = format!("{message} (expected {expected})");
            }
            if let Some(found) = &err.found_token {
                message = format!("{message} (found '{found}')");
            }
            let span = span_payload_from_source(source, err.position, err.position + 1);
            RunMatErrorPayload {
                kind: RunMatErrorKind::Syntax,
                message,
                identifier: Some("RunMat:SyntaxError".to_string()),
                diagnostic,
                span: Some(span),
                callstack: Vec::new(),
                callstack_elided: 0,
            }
        }
        RunError::Semantic(err) => RunMatErrorPayload {
            kind: RunMatErrorKind::Semantic,
            message: err.message.clone(),
            identifier: err.identifier.clone(),
            diagnostic,
            span: err.span.map(|span| {
                let end = span.end.max(span.start + 1);
                span_payload_from_source(source, span.start, end)
            }),
            callstack: Vec::new(),
            callstack_elided: 0,
        },
        RunError::Compile(err) => RunMatErrorPayload {
            kind: RunMatErrorKind::Compile,
            message: err.message.clone(),
            identifier: err.identifier.clone(),
            diagnostic,
            span: err.span.map(|span| {
                let end = span.end.max(span.start + 1);
                span_payload_from_source(source, span.start, end)
            }),
            callstack: Vec::new(),
            callstack_elided: 0,
        },
        RunError::Runtime(err) => runtime_error_payload(err, Some(source)),
    }
}

fn run_error_to_js(err: &RunError, source: &str) -> JsValue {
    serde_wasm_bindgen::to_value(&run_error_payload(err, source))
        .unwrap_or_else(|_| JsValue::from_str("RunMat error"))
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
                web_sys::console::error_1(&JsValue::from_str(
                    "RunMat panic hook invoked; forwarding to console_error_panic_hook",
                ));
                console_error_panic_hook::hook(info);
                let bt = Backtrace::force_capture();
                web_sys::console::error_1(&JsValue::from_str(&format!(
                    "RunMat panic backtrace:\n{bt:?}"
                )));
            }));
            let guard = init_logging(LoggingOptions {
                enable_otlp: false,
                enable_traces: true,
                pid: 1,
                default_filter: Some(
                    LOG_FILTER_OVERRIDE
                        .get()
                        .cloned()
                        .unwrap_or_else(|| "debug".to_string()),
                ),
            });
            let _ = LOGGING_GUARD.set(guard);
            ensure_runtime_log_forwarder_installed();
            ensure_trace_forwarder_installed();
        }
    });
}

fn set_log_filter_override(level: &str) {
    let normalized = level.trim();
    if normalized.is_empty() {
        return;
    }
    if LOGGING_GUARD.get().is_none() {
        let _ = LOG_FILTER_OVERRIDE.set(normalized.to_string());
    }
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
    slice: Option<MaterializeSlicePayload>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MaterializeSlicePayload {
    start: Vec<usize>,
    shape: Vec<usize>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
enum RunMatErrorKind {
    Syntax,
    Semantic,
    Compile,
    Runtime,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RunMatErrorSpanPayload {
    start: usize,
    end: usize,
    line: usize,
    column: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RunMatErrorPayload {
    kind: RunMatErrorKind,
    message: String,
    identifier: Option<String>,
    diagnostic: String,
    span: Option<RunMatErrorSpanPayload>,
    callstack: Vec<String>,
    callstack_elided: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ExecutionPayload {
    value_text: Option<String>,
    value_json: Option<JsonValue>,
    type_info: Option<String>,
    execution_time_ms: u64,
    used_jit: bool,
    error: Option<RunMatErrorPayload>,
    stdout: Vec<ConsoleStreamPayload>,
    workspace: WorkspacePayload,
    figures_touched: Vec<u32>,
    warnings: Vec<WarningPayload>,
    stdin_events: Vec<StdinEventPayload>,
    profiling: Option<ProfilingPayload>,
    fusion_plan: Option<FusionPlanPayload>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct MemoryUsagePayload {
    bytes: u64,
    pages: u32,
}

impl ExecutionPayload {
    fn from_result(result: ExecutionResult, source: &str) -> Self {
        let value_text = result.value.as_ref().map(|v| v.to_string());
        let value_json = result.value.as_ref().map(|v| value_to_json(v, 0));
        let error = result
            .error
            .as_ref()
            .map(|err| runtime_error_payload(err, Some(source)));
        Self {
            value_text,
            value_json,
            type_info: result.type_info,
            execution_time_ms: result.execution_time_ms,
            used_jit: result.used_jit,
            error,
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
    let mut opts = WorkspaceMaterializeOptions::default();
    if let Some(limit) = payload.limit {
        opts.max_elements = limit.clamp(1, MAX_DATA_PREVIEW);
    }
    if let Some(slice) = payload.slice {
        opts.slice = Some(WorkspaceSliceOptions {
            start: slice.start,
            shape: slice.shape,
        });
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
            let rows = arr.shape.first().copied().unwrap_or(0);
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
        Value::OutputList(values) => {
            let truncated = values.len() > MAX_OUTPUT_LIST_ITEMS;
            let items: Vec<JsonValue> = values
                .iter()
                .take(MAX_OUTPUT_LIST_ITEMS)
                .map(|v| value_to_json(v, depth + 1))
                .collect();
            json!({
                "kind": "output-list",
                "length": values.len(),
                "items": items,
                "truncated": truncated,
            })
        }
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
        "fieldOrder": st.field_names().take(MAX_STRUCT_FIELDS).cloned().collect::<Vec<_>>(),
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
    let rows = shape.first().copied().unwrap_or(0);
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
