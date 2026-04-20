use std::cell::{Cell, RefCell};
use std::sync::Arc;

use log::warn;
use runmat_core::RunMatSession;
use runmat_core::{
    TelemetryFailureInfo, TelemetryHost, TelemetryPlatformInfo, TelemetryRunConfig,
    TelemetryRunFinish, TelemetrySink, WorkspaceEntry, WorkspaceResidency,
};
use runmat_runtime::builtins::plotting::{
    close_figure as runtime_close_figure, current_figure_handle as runtime_current_figure_handle,
    figure_handles as runtime_figure_handles,
    invalidate_surface_revisions as runtime_invalidate_surface_revisions,
    reset_hold_state_for_run as runtime_reset_hold_state_for_run,
    reset_plot_state as runtime_reset_plot_state, FigureHandle,
};
use runmat_runtime::builtins::wasm_registry;
use runmat_runtime::data::{
    dataset_root, read_array_payload_async, read_array_slice_payload_async, read_manifest_async,
};
use runmat_runtime::{
    runtime_plot_export_figure_scene, runtime_plot_import_figure_scene_async,
    runtime_plot_import_figure_scene_from_path_async, ReplayErrorKind,
};
use serde_json::Value as JsonValue;
use tracing::{info, info_span};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use crate::api::init::install_fs_provider_value;
use crate::api::streams::js_input_request;
use crate::runtime::config::{
    parse_language_compat_from_str, parse_workspace_export_mode, SessionConfig,
};
use crate::runtime::gpu::{capture_memory_usage, GpuStatus};
use crate::runtime::logging::init_logging_once;
use crate::runtime::state::{
    clear_figure_event_callback as clear_figure_event_callback_state, set_js_stdin_handler,
    JS_STDIN_HANDLER,
};
use crate::wire::errors::{
    js_error, js_value_to_string, run_error_payload, run_error_to_js, runtime_error_to_js,
    RunMatErrorKind,
};
use crate::wire::payloads::{
    compute_data_preview_slice, estimate_data_array_bytes, infer_dataset_class_name,
    parse_data_materialize_options_wire, parse_materialize_options, parse_materialize_target,
    DataMaterializedVariablePayload, ExecutionPayload, FusionPlanPayload,
    MaterializedVariablePayload, MemoryUsagePayload, StatsPayload, WorkspaceEntryPayload,
    WorkspacePayload, WorkspacePreviewPayload,
};
use crate::wire::value::MAX_DATA_PREVIEW;

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

impl RunMatWasm {
    pub(crate) fn new(
        session: RunMatSession,
        snapshot_seed: Option<Vec<u8>>,
        config: SessionConfig,
        gpu_status: GpuStatus,
        telemetry_sink: Option<Arc<dyn TelemetrySink>>,
    ) -> Self {
        Self {
            session: RefCell::new(session),
            snapshot_seed,
            config: RefCell::new(config),
            gpu_status,
            disposed: Cell::new(false),
            active_interrupt: RefCell::new(None),
            telemetry_sink,
        }
    }

    fn ensure_not_disposed(&self) -> Result<(), JsValue> {
        if self.disposed.get() {
            return Err(js_error("RunMat session has been disposed"));
        }
        Ok(())
    }
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
        let figures_before: Vec<u32> = runtime_figure_handles()
            .into_iter()
            .map(|handle| handle.as_u32())
            .collect();
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

        let mut session = {
            let mut slot = self.session.borrow_mut();
            std::mem::take(&mut *slot)
        };

        let exec_result = session.execute(&source).await;
        *self.session.borrow_mut() = session;
        let payload = match exec_result {
            Ok(result) => {
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
            let failure = payload.error.as_ref().map(|err| TelemetryFailureInfo {
                stage: match &err.kind {
                    RunMatErrorKind::Syntax => "parser",
                    RunMatErrorKind::Semantic => "hir",
                    RunMatErrorKind::Compile => "compile",
                    RunMatErrorKind::Runtime => "runtime",
                }
                .to_string(),
                code: err
                    .identifier
                    .clone()
                    .unwrap_or_else(|| "RunMat:RuntimeError".to_string()),
                has_span: err.span.is_some(),
                component: None,
            });
            run.finish(TelemetryRunFinish {
                duration: Some(duration),
                success,
                jit_used: payload.used_jit,
                error,
                failure,
                host: Some(TelemetryHost::Wasm),
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
        let has_stdin_handler = JS_STDIN_HANDLER.with(|s| s.borrow().is_some());
        if has_stdin_handler {
            slot.install_async_input_handler(|req| async move { js_input_request(req).await });
        }
        runtime_reset_plot_state();
        runtime_invalidate_surface_revisions();
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
    pub fn cancel_pending_requests(&self) {}

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

    #[wasm_bindgen(js_name = setFusionPlanEnabled)]
    pub fn set_fusion_plan_enabled(&self, enabled: bool) {
        self.session.borrow_mut().set_emit_fusion_plan(enabled);
        if let Ok(mut cfg) = self.config.try_borrow_mut() {
            cfg.emit_fusion_plan = enabled;
        }
    }

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
            let mut session = {
                let mut slot = self.session.borrow_mut();
                std::mem::take(&mut *slot)
            };

            let materialize_result = session.materialize_variable(target, opts).await;
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

    #[wasm_bindgen(js_name = exportWorkspaceState)]
    pub async fn export_workspace_state(
        &self,
        include_variables: Option<String>,
    ) -> Result<Option<Vec<u8>>, JsValue> {
        self.ensure_not_disposed()?;
        let mode = parse_workspace_export_mode(include_variables.as_deref());

        let mut session = {
            let mut slot = self.session.borrow_mut();
            std::mem::take(&mut *slot)
        };
        let result = session.export_workspace_state(mode).await;
        *self.session.borrow_mut() = session;

        result.map_err(|err| js_error(&format!("Failed to export workspace state: {err}")))
    }

    #[wasm_bindgen(js_name = importWorkspaceState)]
    pub fn import_workspace_state(&self, state: &[u8]) -> Result<bool, JsValue> {
        self.ensure_not_disposed()?;
        let mut session = self.session.borrow_mut();
        match session.import_workspace_state(state) {
            Ok(()) => Ok(true),
            Err(err) => {
                warn!("RunMat wasm: workspace import rejected: {err}");
                Ok(false)
            }
        }
    }

    #[wasm_bindgen(js_name = workspaceSnapshot)]
    pub fn workspace_snapshot(&self) -> Result<JsValue, JsValue> {
        self.ensure_not_disposed()?;
        let payload = {
            let mut session = self.session.borrow_mut();
            WorkspacePayload::from(session.workspace_snapshot())
        };
        serde_wasm_bindgen::to_value(&payload)
            .map_err(|err| js_error(&format!("Failed to serialize workspace snapshot: {err}")))
    }

    #[wasm_bindgen(js_name = inspectDataFile)]
    pub async fn inspect_data_file(&self, path: &str) -> Result<JsValue, JsValue> {
        self.ensure_not_disposed()?;
        let root = dataset_root(path);
        let manifest = read_manifest_async(&root)
            .await
            .map_err(|err| js_error(&format!("inspectDataFile failed: {err}")))?;
        let entries: Vec<WorkspaceEntryPayload> = manifest
            .arrays
            .iter()
            .map(|(name, meta)| {
                WorkspaceEntryPayload::from(WorkspaceEntry {
                    name: name.clone(),
                    class_name: infer_dataset_class_name(meta.shape.len()).to_string(),
                    dtype: Some(meta.dtype.clone()),
                    shape: meta.shape.clone(),
                    is_gpu: false,
                    size_bytes: Some(estimate_data_array_bytes(&meta.shape, &meta.dtype)),
                    preview: None,
                    residency: WorkspaceResidency::Cpu,
                    preview_token: None,
                })
            })
            .collect();
        serde_wasm_bindgen::to_value(&entries)
            .map_err(|err| js_error(&format!("Failed to serialize data entries: {err}")))
    }

    #[wasm_bindgen(js_name = materializeDataFileVariable)]
    pub async fn materialize_data_file_variable(
        &self,
        path: &str,
        array: &str,
        options: JsValue,
    ) -> Result<JsValue, JsValue> {
        self.ensure_not_disposed()?;
        let wire = parse_data_materialize_options_wire(options)?;
        let root = dataset_root(path);
        let manifest = read_manifest_async(&root).await.map_err(|err| {
            js_error(&format!(
                "materializeDataFileVariable manifest read failed: {err}"
            ))
        })?;
        let meta = manifest
            .arrays
            .get(array)
            .ok_or_else(|| js_error(&format!("Array not found in dataset: {array}")))?;
        let total_elements = meta.shape.iter().copied().product::<usize>().max(1);
        let limit = wire.limit.unwrap_or(MAX_DATA_PREVIEW).max(1);
        let (slice_start, slice_shape) =
            compute_data_preview_slice(&meta.shape, wire.slice.as_ref(), limit);
        let payload =
            match read_array_slice_payload_async(&root, meta, &slice_start, &slice_shape).await {
                Ok(payload) => payload,
                Err(_) => read_array_payload_async(&root, meta).await.map_err(|err| {
                    js_error(&format!(
                        "materializeDataFileVariable payload read failed: {err}"
                    ))
                })?,
            };
        let values: Vec<f64> = payload.values.into_iter().take(limit).collect();
        let response = DataMaterializedVariablePayload {
            name: array.to_string(),
            class_name: infer_dataset_class_name(meta.shape.len()).to_string(),
            dtype: Some(meta.dtype.clone()),
            shape: meta.shape.clone(),
            is_gpu: false,
            residency: WorkspaceResidency::Cpu.as_str(),
            size_bytes: Some(estimate_data_array_bytes(&meta.shape, &meta.dtype)),
            preview: Some(WorkspacePreviewPayload {
                values: values.clone(),
                truncated: values.len() < total_elements,
            }),
            value_text: values
                .iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>()
                .join(", "),
            value_json: JsonValue::Array(values.into_iter().map(JsonValue::from).collect()),
        };
        serde_wasm_bindgen::to_value(&response).map_err(|err| {
            js_error(&format!(
                "Failed to serialize data materialized value: {err}"
            ))
        })
    }

    #[wasm_bindgen(js_name = exportFigureScene)]
    pub fn export_figure_scene(&self, handle: u32) -> Result<Option<Vec<u8>>, JsValue> {
        self.ensure_not_disposed()?;
        log::debug!("RunMat wasm: exportFigureScene start handle={handle}");
        match runtime_plot_export_figure_scene(FigureHandle::from(handle)) {
            Ok(Some(payload)) => {
                log::debug!(
                    "RunMat wasm: exportFigureScene ok handle={} bytes={}",
                    handle,
                    payload.len()
                );
                Ok(Some(payload))
            }
            Ok(None) => {
                log::debug!("RunMat wasm: exportFigureScene empty handle={}", handle);
                let current = runtime_current_figure_handle();
                if current.as_u32() == handle {
                    return Ok(None);
                }
                match runtime_plot_export_figure_scene(current) {
                    Ok(payload) => {
                        log::debug!(
                            "RunMat wasm: exportFigureScene fallback handle={} current_handle={} has_payload={}",
                            handle,
                            current.as_u32(),
                            payload.as_ref().map(|bytes| bytes.len()).unwrap_or(0)
                        );
                        Ok(payload)
                    }
                    Err(err) => {
                        warn!("RunMat wasm: fallback figure scene export rejected: {err}");
                        Ok(None)
                    }
                }
            }
            Err(err) => {
                warn!("RunMat wasm: figure scene export rejected: {err}");
                Ok(None)
            }
        }
    }

    #[wasm_bindgen(js_name = importFigureScene)]
    pub async fn import_figure_scene(&self, scene: &[u8]) -> Result<Option<u32>, JsValue> {
        self.ensure_not_disposed()?;
        log::debug!("RunMat wasm: importFigureScene start bytes={}", scene.len());
        match runtime_plot_import_figure_scene_async(scene).await {
            Ok(Some(handle)) => {
                log::debug!(
                    "RunMat wasm: importFigureScene ok handle={}",
                    handle.as_u32()
                );
                Ok(Some(handle.as_u32()))
            }
            Ok(None) => {
                log::debug!("RunMat wasm: importFigureScene returned none");
                Ok(None)
            }
            Err(err) => {
                if err.identifier() == Some(ReplayErrorKind::DecodeFailed.identifier()) {
                    warn!("RunMat wasm: figure scene decode failed: {err}");
                } else {
                    warn!("RunMat wasm: figure scene import rejected: {err}");
                }
                Err(runtime_error_to_js(&err))
            }
        }
    }

    #[wasm_bindgen(js_name = importFigureSceneFromPath)]
    pub async fn import_figure_scene_from_path(&self, path: &str) -> Result<Option<u32>, JsValue> {
        self.ensure_not_disposed()?;
        match runtime_plot_import_figure_scene_from_path_async(path).await {
            Ok(Some(handle)) => Ok(Some(handle.as_u32())),
            Ok(None) => Ok(None),
            Err(err) => {
                warn!("RunMat wasm: figure scene import-from-path rejected: {err}");
                Err(runtime_error_to_js(&err))
            }
        }
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
        let (bytes, pages) = capture_memory_usage().map_err(|err| {
            js_error(&format!(
                "Failed to capture wasm memory usage: {}",
                js_value_to_string(err.clone())
            ))
        })?;
        let stats = MemoryUsagePayload { bytes, pages };
        serde_wasm_bindgen::to_value(&stats)
            .map_err(|err| js_error(&format!("Failed to serialize memory stats: {err}")))
    }

    #[wasm_bindgen(js_name = setFsProvider)]
    pub fn set_fs_provider(&self, bindings: JsValue) -> Result<(), JsValue> {
        install_fs_provider_value(bindings).map_err(|err| {
            js_error(&format!(
                "Failed to install filesystem provider: {}",
                js_value_to_string(err)
            ))
        })
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
            clear_figure_event_callback_state();
        }
    }
}
