use std::cell::RefCell;
use std::sync::{Arc, OnceLock};

#[cfg(target_arch = "wasm32")]
use js_sys::{Array, Reflect};
use log::warn;
use runmat_accelerate::{
    initialize_acceleration_provider_with, AccelPowerPreference, AccelerateInitOptions,
    AccelerateProviderPreference, AutoOffloadOptions,
};
use runmat_accelerate_api::ProviderPrecision;
#[cfg(target_arch = "wasm32")]
use runmat_accelerate_api::{AccelContextHandle, AccelContextKind};
use runmat_builtins::{NumericDType, ObjectInstance, StructValue, Value};
use runmat_core::{ExecutionResult, RunMatSession};
use runmat_runtime::builtins::plotting::{set_scatter_target_points, set_surface_vertex_budget};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map as JsonMap, Value as JsonValue};
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
    web::{WebRenderer, WebRendererOptions},
    SharedWgpuContext,
};
#[cfg(target_arch = "wasm32")]
use runmat_runtime::builtins::plotting::{
    configure_subplot as runtime_configure_subplot,
    context as plotting_context,
    current_figure_handle as runtime_current_figure_handle,
    install_figure_observer as runtime_install_figure_observer,
    install_web_renderer as runtime_install_web_renderer,
    install_web_renderer_for_handle as runtime_install_web_renderer_for_handle,
    new_figure_handle as runtime_new_figure_handle,
    select_figure as runtime_select_figure,
    set_hold as runtime_set_hold,
    web_renderer_ready as runtime_plot_renderer_ready,
    FigureHandle, HoldMode,
};

const MAX_DATA_PREVIEW: usize = 4096;
const MAX_STRUCT_FIELDS: usize = 64;
const MAX_OBJECT_FIELDS: usize = 64;
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
    #[serde(default)]
    snapshot_stream: Option<JsValue>,
    #[serde(default)]
    scatter_target_points: Option<u32>,
    #[serde(default)]
    surface_vertex_budget: Option<u64>,
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    static FIGURE_EVENT_CALLBACK: RefCell<Option<js_sys::Function>> = RefCell::new(None);
}
#[cfg(target_arch = "wasm32")]
static FIGURE_EVENT_OBSERVER: OnceLock<()> = OnceLock::new();

#[derive(Clone)]
struct SessionConfig {
    enable_jit: bool,
    verbose: bool,
    telemetry_consent: bool,
    enable_gpu: bool,
    wgpu_power_preference: AccelPowerPreference,
    wgpu_force_fallback_adapter: bool,
    auto_offload: AutoOffloadOptions,
}

impl SessionConfig {
    fn from_options(opts: &InitOptions) -> Self {
        Self {
            enable_jit: opts.enable_jit.unwrap_or(false) && cfg!(feature = "jit"),
            verbose: opts.verbose.unwrap_or(false),
            telemetry_consent: opts.telemetry_consent.unwrap_or(true),
            enable_gpu: opts.enable_gpu.unwrap_or(true),
            wgpu_power_preference: parse_power_preference(opts.wgpu_power_preference.as_deref()),
            wgpu_force_fallback_adapter: opts.wgpu_force_fallback_adapter.unwrap_or(false),
            auto_offload: AutoOffloadOptions::default(),
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
    config: SessionConfig,
    gpu_status: GpuStatus,
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
        let session = RunMatSession::with_snapshot_bytes(
            self.config.enable_jit,
            self.config.verbose,
            self.snapshot_seed.as_deref(),
        )
        .map_err(|err| js_error(&format!("Failed to reset session: {err}")))?;
        let mut slot = self.session.borrow_mut();
        *slot = session;
        Ok(())
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
        self.config.telemetry_consent
    }

    #[wasm_bindgen(js_name = gpuStatus)]
    pub fn gpu_status(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.gpu_status)
            .map_err(|err| js_error(&format!("Failed to serialize GPU status: {err}")))
    }
}

#[wasm_bindgen(js_name = registerFsProvider)]
pub fn register_fs_provider(bindings: JsValue) -> Result<(), JsValue> {
    install_fs_provider_value(bindings)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = registerPlotCanvas)]
pub async fn register_plot_canvas(canvas: web_sys::HtmlCanvasElement) -> Result<(), JsValue> {
    install_canvas_renderer(None, canvas).await
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = registerFigureCanvas)]
pub async fn register_figure_canvas(
    handle: u32,
    canvas: web_sys::HtmlCanvasElement,
) -> Result<(), JsValue> {
    install_canvas_renderer(Some(handle), canvas).await
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
    if rows == 0 || cols == 0 {
        return Err(js_error(
            "configureSubplot requires rows and cols to be at least 1",
        ));
    }
    runtime_configure_subplot(rows as usize, cols as usize, index as usize);
    Ok(())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = setHoldMode)]
pub fn wasm_set_hold_mode(mode: JsValue) -> Result<bool, JsValue> {
    let parsed = parse_hold_mode(mode)?;
    Ok(runtime_set_hold(parsed))
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
async fn install_canvas_renderer(
    handle: Option<u32>,
    canvas: web_sys::HtmlCanvasElement,
) -> Result<(), JsValue> {
    init_logging_once();
    let options = WebRendererOptions::default();
    let renderer_future = match shared_webgpu_context() {
        Some(shared) => WebRenderer::with_shared_context(canvas.clone(), options.clone(), shared),
        None => WebRenderer::new(canvas, options),
    };
    let renderer = renderer_future
        .await
        .map_err(|err| js_error(&format!("Failed to initialize plot renderer: {err}")))?;
    match handle {
        Some(id) => runtime_install_web_renderer_for_handle(id, renderer)
            .map_err(|err| js_error(&format!("Failed to register plot renderer: {err}")))?,
        None => runtime_install_web_renderer(renderer)
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
    install_fs_provider_from_options(&options)?;
    let parsed_opts: InitOptions = if options.is_null() || options.is_undefined() {
        InitOptions::default()
    } else {
        serde_wasm_bindgen::from_value(options)
            .map_err(|err| js_error(&format!("Invalid init options: {err}")))?
    };

    apply_plotting_overrides(&parsed_opts);

    let config = SessionConfig::from_options(&parsed_opts);
    let snapshot_seed = resolve_snapshot_bytes(&parsed_opts).await?;

    let session = RunMatSession::with_snapshot_bytes(
        config.enable_jit,
        config.verbose,
        snapshot_seed.as_deref(),
    )
    .map_err(|err| js_error(&format!("Failed to initialize RunMat session: {err}")))?;

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
                let message = js_value_to_string(err);
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
        config,
        gpu_status,
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
    use runmat_plot::plots::Figure;
    FIGURE_EVENT_OBSERVER.get_or_init(|| {
        let observer: Arc<dyn Fn(u32, &Figure) + Send + Sync> =
            Arc::new(|handle, figure| emit_js_figure_event(handle, figure));
        let _ = runtime_install_figure_observer(observer);
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
fn emit_js_figure_event(handle: u32, figure: &runmat_plot::plots::Figure) {
    FIGURE_EVENT_CALLBACK.with(|slot| {
        if let Some(cb) = slot.borrow().as_ref() {
            let payload = js_sys::Object::new();
            let _ = Reflect::set(
                &payload,
                &JsValue::from_str("handle"),
                &JsValue::from(handle),
            );
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
            let _ = cb.call1(&JsValue::NULL, &payload);
        }
    });
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
    use js_sys::{Reflect, Uint8Array};
    use web_sys::{ReadableStream, ReadableStreamDefaultReader};

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
async fn read_stream(stream: ReadableStream) -> Result<Vec<u8>, JsValue> {
    use js_sys::{Reflect, Uint8Array};
    use web_sys::ReadableStreamDefaultReader;

    let reader_value = stream
        .get_reader()
        .map_err(|_| js_error("ReadableStream.getReader() failed"))?;
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

fn init_logging_once() {
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| {
        #[cfg(target_arch = "wasm32")]
        {
            console_error_panic_hook::set_once();
            let _ = wasm_logger::init(wasm_logger::Config::default());
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
struct ExecutionPayload {
    value_text: Option<String>,
    value_json: Option<JsonValue>,
    type_info: Option<String>,
    execution_time_ms: u64,
    used_jit: bool,
    error: Option<String>,
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

fn matlab_class_name(value: &Value) -> String {
    match value {
        Value::Num(_) | Value::Tensor(_) | Value::ComplexTensor(_) | Value::Complex(_, _) => {
            "double".to_string()
        }
        Value::Int(iv) => iv.class_name().to_string(),
        Value::Bool(_) | Value::LogicalArray(_) => "logical".to_string(),
        Value::String(_) | Value::StringArray(_) => "string".to_string(),
        Value::CharArray(_) => "char".to_string(),
        Value::Cell(_) => "cell".to_string(),
        Value::Struct(_) => "struct".to_string(),
        Value::GpuTensor(_) => "gpuArray".to_string(),
        Value::FunctionHandle(_) | Value::Closure(_) => "function_handle".to_string(),
        Value::HandleObject(handle) => {
            if handle.class_name.is_empty() {
                "handle".to_string()
            } else {
                handle.class_name.clone()
            }
        }
        Value::Listener(_) => "event.listener".to_string(),
        Value::Object(obj) => obj.class_name.clone(),
        Value::ClassRef(_) => "meta.class".to_string(),
        Value::MException(_) => "MException".to_string(),
    }
}

fn value_shape(value: &Value) -> Option<Vec<usize>> {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => {
            Some(scalar_shape())
        }
        Value::LogicalArray(arr) => Some(arr.shape.clone()),
        Value::StringArray(sa) => Some(sa.shape.clone()),
        Value::String(s) => Some(vec![1, s.chars().count()]),
        Value::CharArray(ca) => Some(vec![ca.rows, ca.cols]),
        Value::Tensor(t) => Some(t.shape.clone()),
        Value::ComplexTensor(t) => Some(t.shape.clone()),
        Value::Cell(ca) => Some(ca.shape.clone()),
        Value::GpuTensor(handle) => Some(handle.shape.clone()),
        _ => None,
    }
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
