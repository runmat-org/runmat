use std::cell::{Cell, RefCell};
use std::sync::Arc;

use log::warn;
use runmat_core::RunMatSession;
use runmat_core::{
    TelemetryFailureInfo, TelemetryHost, TelemetryPlatformInfo, TelemetryRunConfig,
    TelemetryRunFinish, TelemetrySink, WorkspaceEntry, WorkspaceResidency,
};
use runmat_runtime::analysis::{
    AnalysisFieldPageResult as FeaFieldPayload,
    AnalysisFieldRequestOptions as FeaFieldRequestPayload, AnalysisFieldSelection,
    AnalysisRuntimeCapabilities as FeaCapabilitiesPayload, FeaSourceCheckResult as FeaCheckPayload,
};
use runmat_runtime::builtins::plotting::{
    close_figure as runtime_close_figure, close_geometry_scene as runtime_close_geometry_scene,
    export_geometry_scene as runtime_export_geometry_scene,
    figure_handles as runtime_figure_handles, import_figure as runtime_import_figure,
    import_geometry_scene as runtime_import_geometry_scene,
    import_geometry_scene_payload as runtime_import_geometry_scene_payload,
    invalidate_surface_revisions as runtime_invalidate_surface_revisions,
    reset_hold_state_for_run as runtime_reset_hold_state_for_run,
    reset_plot_state as runtime_reset_plot_state, FigureHandle,
};
use runmat_runtime::builtins::wasm_registry;
use runmat_runtime::data::{
    dataset_root, read_array_payload_async, read_array_slice_payload_async, read_manifest_async,
    DataArrayValues,
};
use runmat_runtime::execution::RuntimeExecutionServices;
use runmat_runtime::geometry::{
    GeometryInspectPayload, GeometryPreviewBudgetPayload, GeometryPreviewBudgetPolicyPayload,
    GeometryPreviewPayload, GeometryStatsPayload,
};
use runmat_runtime::{
    runtime_plot_import_figure_scene_async, runtime_plot_import_figure_scene_from_path_async,
    ReplayErrorKind,
};
use serde::Deserialize;
use serde_json::Value as JsonValue;
use tracing::{info, info_span};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use crate::api::init::install_fs_provider_value;
#[cfg(target_arch = "wasm32")]
use crate::api::streams::JsInputHandler;
use crate::runtime::config::{
    parse_language_compat_from_str, parse_workspace_export_mode, request_error_namespace,
    SessionConfig,
};
use crate::runtime::execution::BrowserExecutionService;
use crate::runtime::gpu::{capture_memory_usage, GpuStatus};
use crate::runtime::logging::init_logging_once;
use crate::runtime::state::clear_figure_event_callback as clear_figure_event_callback_state;
use crate::wire::errors::{
    js_error, js_value_to_string, run_error_payload, run_error_to_js, runtime_error_to_js,
    RunMatErrorKind,
};
use crate::wire::payloads::{
    compute_data_preview_slice, estimate_data_array_bytes, infer_dataset_class_name,
    parse_data_materialize_options_wire, parse_materialize_options, parse_materialize_target,
    DataArrayPreviewPayload, DataMaterializedVariablePayload, ExecutionPayload, FusionPlanPayload,
    MaterializedVariablePayload, MemoryUsagePayload, StatsPayload, WorkspaceEntryPayload,
    WorkspacePayload,
};
use crate::wire::value::{integer_json_value, MAX_DATA_PREVIEW};

#[wasm_bindgen]
pub struct RunMatWasm {
    pub(crate) session: RefCell<RunMatSession>,
    in_flight_session: Cell<bool>,
    #[cfg(target_arch = "wasm32")]
    input_handler: RefCell<Option<JsInputHandler>>,
    pub(crate) fea_selection: RefCell<Option<super::fea_execution::BrowserFeaSelection>>,
    pub(crate) fea_admission: std::rc::Rc<RefCell<Option<JsValue>>>,
    pub(crate) active_fea_lease:
        std::rc::Rc<RefCell<Option<super::fea_run_context::ActiveFeaLease>>>,
    pub(crate) pending_fea_admission: std::rc::Rc<Cell<bool>>,
    clear_workspace_after_flight: Cell<bool>,
    #[cfg(target_arch = "wasm32")]
    local_tasks: RefCell<std::rc::Rc<runmat_runtime::context::BrowserLocalTaskSupervisor>>,
    config: std::rc::Rc<RefCell<SessionConfig>>,
    gpu_status: GpuStatus,
    disposed: Cell<bool>,
    pub(crate) active_interrupt:
        RefCell<Option<runmat_execution::cancellation::CancellationSource>>,
    telemetry_sink: Option<Arc<dyn TelemetrySink>>,
    package_cache: Option<Arc<dyn runmat_package_cache::CacheBackend>>,
    execution_service: std::rc::Rc<BrowserExecutionService>,
    dialog_host: RefCell<Option<std::rc::Rc<dyn runmat_runtime::context::RuntimeHostService>>>,
    application_settings:
        Option<std::rc::Rc<runmat_runtime::context::IsolatedMemoryPreferenceService>>,
}

pub(crate) struct InFlightSession<'a> {
    host: &'a RunMatWasm,
    session: Option<RunMatSession>,
}

impl std::ops::Deref for InFlightSession<'_> {
    type Target = RunMatSession;

    fn deref(&self) -> &Self::Target {
        self.session.as_ref().expect("in-flight session is owned")
    }
}

impl std::ops::DerefMut for InFlightSession<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.session.as_mut().expect("in-flight session is owned")
    }
}

impl Drop for InFlightSession<'_> {
    fn drop(&mut self) {
        if let Some(mut session) = self.session.take() {
            if self.host.disposed.get() {
                session.cancel_execution();
                session.shutdown_local_tasks();
                if let Err(err) = session.clear_variables() {
                    warn!("RunMat wasm: disposed in-flight workspace cleanup failed: {err}");
                }
                // The moved session is terminal. Never replace the disposed
                // placeholder with an owner whose callbacks could resume.
            } else {
                let config = self.host.config.borrow();
                session.set_compat_mode(config.language_compat);
                session.set_error_namespace(config.error_namespace.clone());
                session.set_emit_fusion_plan(config.emit_fusion_plan);
                drop(config);
                if self.host.clear_workspace_after_flight.replace(false) {
                    if let Err(err) = session.clear_variables() {
                        warn!("RunMat wasm: deferred workspace clear failed: {err}");
                    }
                }
                *self.host.session.borrow_mut() = session;
            }
        }
        self.host.in_flight_session.set(false);
    }
}

pub(crate) struct WasmInterruptGuard<'a> {
    slot: &'a RefCell<Option<runmat_execution::cancellation::CancellationSource>>,
    _runtime_guard: runmat_runtime::interrupt::InterruptGuard,
}

impl Drop for WasmInterruptGuard<'_> {
    fn drop(&mut self) {
        self.slot.borrow_mut().take();
    }
}

pub(crate) fn install_wasm_interrupt(
    slot: &RefCell<Option<runmat_execution::cancellation::CancellationSource>>,
    handle: runmat_execution::cancellation::CancellationSource,
) -> WasmInterruptGuard<'_> {
    slot.borrow_mut().replace(handle.clone());
    let runtime_guard = runmat_runtime::interrupt::replace_interrupt(Some(handle.token()));
    WasmInterruptGuard {
        slot,
        _runtime_guard: runtime_guard,
    }
}

/// Await read-only preparation with the request's cancellation generation.
/// Dropping the pending read releases its captured scope and session borrow;
/// it does not cancel or detach publication/execution owned by another request.
pub(crate) async fn await_wasm_preparation<T>(
    cancellation: &runmat_execution::cancellation::CancellationSource,
    preparation: impl std::future::Future<Output = Result<T, JsValue>>,
) -> Result<T, JsValue> {
    use futures::FutureExt;
    let token = cancellation.token();
    let cancelled = token.cancelled().fuse();
    let preparation = preparation.fuse();
    futures::pin_mut!(cancelled, preparation);
    futures::select_biased! {
        _ = cancelled => {
            let error = runmat_core::RunError::Runtime(
                runmat_runtime::build_runtime_error("Execution cancelled by user")
                    .with_identifier("RunMat:ExecutionCancelled")
                    .build(),
            );
            Err(crate::wire::errors::run_error_to_js(&error, None, None))
        },
        result = preparation => result,
    }
}

impl RunMatWasm {
    pub(crate) fn config_for_named_project(
        &self,
        project: runmat_config::runtime::HostConfigDocument,
        user: Option<String>,
    ) -> Result<SessionConfig, String> {
        if self.disposed.get() {
            return Err("RunMat session has been disposed".into());
        }
        self.config.borrow().for_project(project, user)
    }

    /// Session mechanisms and the WASM request adapter consume one admitted
    /// configuration. Startup-owned physical/resource settings cannot silently
    /// change under another project in the same browser host.
    pub(crate) fn admit_named_config(&self, next: &SessionConfig) -> Result<(), String> {
        let current = self.config.borrow();
        let startup = |runtime: &runmat_config::runtime::RunMatRuntimeConfig| {
            serde_json::to_value((
                &runtime.gc,
                &runtime.plotting,
                &runtime.logging,
                &runtime.fea,
                &runtime.accelerate,
            ))
            .map_err(|error| error.to_string())
        };
        if startup(&current.runtime)? != startup(&next.runtime)? {
            return Err("selected project changes browser startup-owned GPU, GC, plotting, logging or FEA storage settings; initialize a host with the selected project settings".into());
        }
        Ok(())
    }

    pub(crate) fn publish_named_config(&self, next: SessionConfig) {
        self.config.replace(next);
    }
}

fn ensure_runtime_not_cancelled(operation: &str) -> Result<(), String> {
    if runmat_runtime::interrupt::is_cancelled() {
        Err(format!("{operation} cancelled by user"))
    } else {
        Ok(())
    }
}

fn operation_error_message(err: runmat_runtime::operations::OperationErrorEnvelope) -> String {
    format!("{}: {}", err.error_code, err.message)
}

struct GeometryInspectWithAsset {
    inspect: GeometryInspectPayload,
    asset: Option<runmat_geometry_core::GeometryAsset>,
}

struct GeometryInputFile {
    byte_count: u64,
    bytes: Option<Vec<u8>>,
}

impl RunMatWasm {
    pub(crate) fn new(
        mut session: RunMatSession,
        config: SessionConfig,
        gpu_status: GpuStatus,
        telemetry_sink: Option<Arc<dyn TelemetrySink>>,
        package_cache: Option<Arc<dyn runmat_package_cache::CacheBackend>>,
        execution_service: std::rc::Rc<BrowserExecutionService>,
        dialog_host: Option<std::rc::Rc<dyn runmat_runtime::context::RuntimeHostService>>,
    ) -> Self {
        let fea_admission = std::rc::Rc::new(RefCell::new(None));
        let active_fea_lease = std::rc::Rc::new(RefCell::new(None));
        let pending_fea_admission = std::rc::Rc::new(Cell::new(false));
        let config = std::rc::Rc::new(RefCell::new(config));
        session.install_analysis_service(std::rc::Rc::new(
            super::fea_analysis_host::BrowserAnalysisService {
                host: fea_admission.clone(),
                active: active_fea_lease.clone(),
                config: config.clone(),
            },
        ));
        session.install_fea_run_admission_service(std::rc::Rc::new(
            super::fea_async_host::BrowserFeaAdmission {
                host: fea_admission.clone(),
                active: active_fea_lease.clone(),
                pending: pending_fea_admission.clone(),
            },
        ));
        #[cfg(target_arch = "wasm32")]
        let local_tasks = RefCell::new(session.browser_local_task_supervisor());
        Self {
            session: RefCell::new(session),
            in_flight_session: Cell::new(false),
            #[cfg(target_arch = "wasm32")]
            input_handler: RefCell::new(None),
            fea_selection: RefCell::new(None),
            fea_admission,
            active_fea_lease,
            pending_fea_admission,
            clear_workspace_after_flight: Cell::new(false),
            #[cfg(target_arch = "wasm32")]
            local_tasks,
            config,
            gpu_status,
            disposed: Cell::new(false),
            active_interrupt: RefCell::new(None),
            telemetry_sink,
            package_cache,
            execution_service,
            dialog_host: RefCell::new(dialog_host),
            application_settings: None,
        }
    }

    pub(crate) fn with_application_settings(
        mut self,
        settings: std::rc::Rc<runmat_runtime::context::IsolatedMemoryPreferenceService>,
    ) -> Result<Self, String> {
        self.session
            .get_mut()
            .install_application_settings(settings.clone())
            .map_err(|error| error.to_string())?;
        self.application_settings = Some(settings);
        Ok(self)
    }

    pub(crate) fn ensure_not_disposed(&self) -> Result<(), JsValue> {
        if self.disposed.get() {
            return Err(js_error("RunMat session has been disposed"));
        }
        if self.in_flight_session.get() {
            return Err(js_error("RunMat session already has an in-flight request"));
        }
        Ok(())
    }

    pub(crate) fn take_session(&self) -> Result<InFlightSession<'_>, JsValue> {
        self.ensure_not_disposed()?;
        if self.in_flight_session.replace(true) {
            return Err(js_error("RunMat session already has an in-flight request"));
        }
        let session = std::mem::take(&mut *self.session.borrow_mut());
        Ok(InFlightSession {
            host: self,
            session: Some(session),
        })
    }
}

async fn inspect_geometry_path_with_asset(
    path: String,
    budget: Option<GeometryPreviewBudgetPayload>,
) -> Result<GeometryInspectWithAsset, String> {
    let deadline = geometry_preview_deadline(budget.as_ref());
    ensure_runtime_not_cancelled("geometry inspection")?;
    ensure_geometry_preview_deadline(deadline, "geometry inspection")?;
    let mut input_file = read_geometry_input_file(&path).await?;
    ensure_runtime_not_cancelled("geometry inspection")?;
    ensure_geometry_preview_deadline(deadline, "geometry inspection")?;
    let byte_count = input_file.byte_count;
    if let Some(max_bytes) = budget.as_ref().and_then(|item| item.max_bytes) {
        if byte_count > max_bytes {
            return Ok(GeometryInspectWithAsset {
                inspect: GeometryInspectPayload {
                    path: path.clone(),
                    format: geometry_format_from_path(&path),
                    byte_count,
                    supported: is_supported_geometry_path(&path),
                    stats: None,
                    geometry_summary: None,
                    regions: Vec::new(),
                    diagnostics: Vec::new(),
                    degraded_reason: Some(format!(
                        "Geometry file is {byte_count} bytes, above the preview budget of {max_bytes} bytes."
                    )),
                },
                asset: None,
            });
        }
    }

    let bytes = match input_file.bytes.take() {
        Some(bytes) => bytes,
        None => runmat_filesystem::read_async(std::path::PathBuf::from(&path))
            .await
            .map_err(|err| format!("failed to read geometry file {path}: {err}"))?,
    };
    ensure_runtime_not_cancelled("geometry inspection")?;
    ensure_geometry_preview_deadline(deadline, "geometry inspection")?;
    let inspect = runmat_runtime::geometry::geometry_inspect_op(
        &path,
        &bytes,
        runmat_runtime::operations::OperationContext::new(None, None),
    )
    .map_err(operation_error_message)?;
    ensure_runtime_not_cancelled("geometry inspection")?;
    ensure_geometry_preview_deadline(deadline, "geometry inspection")?;
    let mut payload = GeometryInspectPayload {
        path: path.clone(),
        format: inspect.data.format,
        byte_count: inspect.data.byte_count as u64,
        supported: false,
        stats: None,
        geometry_summary: None,
        regions: Vec::new(),
        diagnostics: Vec::new(),
        degraded_reason: None,
    };
    payload.supported = payload.format != "unknown";

    if !payload.supported {
        payload.degraded_reason = Some("Unsupported geometry format.".to_string());
        return Ok(GeometryInspectWithAsset {
            inspect: payload,
            asset: None,
        });
    }

    let max_triangles = budget
        .as_ref()
        .and_then(|item| item.max_triangles)
        .or(Some(16_000_000));
    let import_started_ms = js_sys::Date::now();
    info!(
        target: "runmat.geometry.preview",
        path = %path,
        max_triangles = ?max_triangles,
        budget_policy = ?budget.as_ref().and_then(|item| item.budget_policy),
        "geometry preview import started"
    );
    let asset = match runmat_runtime::geometry::geometry_load_with_options_op(
        &path,
        &bytes,
        geometry_import_options_from_budget(budget.as_ref(), max_triangles),
        runmat_runtime::operations::OperationContext::new(None, None),
    ) {
        Ok(envelope) => {
            info!(
                target: "runmat.geometry.preview",
                path = %path,
                elapsed_ms = (js_sys::Date::now() - import_started_ms) as u64,
                "geometry preview import completed"
            );
            envelope.data
        }
        Err(err) => {
            let message = operation_error_message(err);
            warn!(
                target: "runmat.geometry.preview",
                "RunMat wasm: geometry preview import failed path={} elapsed_ms={} error={}",
                path,
                (js_sys::Date::now() - import_started_ms) as u64,
                message
            );
            payload.degraded_reason = Some(message);
            return Ok(GeometryInspectWithAsset {
                inspect: payload,
                asset: None,
            });
        }
    };
    ensure_runtime_not_cancelled("geometry inspection")?;
    ensure_geometry_preview_deadline(deadline, "geometry inspection")?;
    let stats = runmat_runtime::geometry::geometry_compute_stats_op(
        &asset,
        runmat_runtime::operations::OperationContext::new(None, None),
    )
    .map_err(operation_error_message)?
    .data;
    if let Some(max_vertices) = budget.as_ref().and_then(|item| item.max_vertices) {
        if stats.total_vertices > max_vertices {
            payload.degraded_reason = Some(format!(
                "Geometry has {} vertices, above the preview budget of {max_vertices} vertices.",
                stats.total_vertices
            ));
        }
    }
    payload.stats = Some(GeometryStatsPayload {
        mesh_count: stats.mesh_count,
        total_vertices: stats.total_vertices,
        total_elements: stats.total_elements,
        region_count: stats.region_count,
    });
    payload.geometry_summary =
        serde_json::to_value(runmat_runtime::geometry::geometry_asset_summary(&asset)).ok();
    payload.regions = asset
        .regions
        .iter()
        .filter_map(|region| serde_json::to_value(region).ok())
        .collect();
    payload.diagnostics = asset
        .diagnostics
        .iter()
        .filter_map(|diagnostic| serde_json::to_value(diagnostic).ok())
        .collect();
    Ok(GeometryInspectWithAsset {
        inspect: payload,
        asset: Some(asset),
    })
}

async fn read_geometry_input_file(path: &str) -> Result<GeometryInputFile, String> {
    match runmat_filesystem::metadata_async(std::path::PathBuf::from(path)).await {
        Ok(metadata) => Ok(GeometryInputFile {
            byte_count: metadata.len(),
            bytes: None,
        }),
        Err(metadata_err) => {
            let bytes = runmat_filesystem::read_async(std::path::PathBuf::from(path))
                .await
                .map_err(|read_err| {
                    format!(
                        "failed to inspect geometry metadata {path}: {metadata_err}; \
                         also failed to read geometry file through the active filesystem provider: {read_err}"
                    )
                })?;
            Ok(GeometryInputFile {
                byte_count: bytes.len() as u64,
                bytes: Some(bytes),
            })
        }
    }
}

async fn preview_geometry_path(
    path: String,
    budget: Option<GeometryPreviewBudgetPayload>,
) -> Result<GeometryPreviewPayload, String> {
    let deadline = geometry_preview_deadline(budget.as_ref());
    ensure_runtime_not_cancelled("geometry preview")?;
    ensure_geometry_preview_deadline(deadline, "geometry preview")?;
    let figure_options = geometry_preview_figure_options_from_budget(budget.as_ref());
    let xray = budget
        .as_ref()
        .and_then(|budget| budget.xray)
        .unwrap_or(false);
    let inspected = inspect_geometry_path_with_asset(path, budget.clone()).await?;
    let GeometryInspectWithAsset { inspect, asset } = inspected;
    ensure_runtime_not_cancelled("geometry preview")?;
    ensure_geometry_preview_deadline(deadline, "geometry preview")?;
    if asset.is_none() || inspect.stats.is_none() {
        return Ok(GeometryPreviewPayload {
            inspect,
            scene_kind: "unavailable".to_string(),
            figure_handle: None,
            geometry_scene_handle: None,
            truncated: false,
            preview_message: Some("Geometry preview is unavailable for this file.".to_string()),
        });
    }
    let asset = asset.ok_or_else(|| "geometry preview asset was not available".to_string())?;
    let truncated = geometry_asset_preview_truncated(&asset);
    ensure_runtime_not_cancelled("geometry preview")?;
    ensure_geometry_preview_deadline(deadline, "geometry preview")?;
    let title = format!("Geometry Preview: {}", filename_for_display(&inspect.path));
    let scene_options = runmat_runtime::geometry::GeometryPreviewSceneOptions {
        triangles_per_chunk: 24_000,
        presentation: runmat_runtime::geometry::GeometryPreviewPresentation::Cad,
        xray,
        allow_create_fea_study: budget
            .as_ref()
            .and_then(|budget| budget.allow_create_fea_study)
            .unwrap_or(false),
    };
    match runmat_runtime::geometry::geometry_preview_scene(&asset, title.clone(), scene_options) {
        Ok(scene) => {
            ensure_geometry_preview_deadline(deadline, "geometry preview")?;
            let overlay = runmat_runtime::geometry::geometry_preview_scene_overlay(
                &asset,
                Some(filename_for_display(&inspect.path).to_string()),
                if truncated {
                    runmat_plot::GeometrySceneCompleteness::BoundedPreview
                } else {
                    runmat_plot::GeometrySceneCompleteness::Complete
                },
                geometry_preview_quality_label(budget.as_ref(), truncated),
                Some(inspect.format.clone()),
                Some(inspect.byte_count),
                scene_options.allow_create_fea_study,
            );
            let scene = scene.with_overlay(overlay);
            let geometry_scene_handle =
                runtime_import_geometry_scene(scene).map_err(|err| err.to_string())?;
            return Ok(GeometryPreviewPayload {
                inspect,
                scene_kind: "mesh".to_string(),
                figure_handle: None,
                geometry_scene_handle: Some(geometry_scene_handle),
                truncated,
                preview_message: truncated.then(|| {
                    "Preview mesh was limited by the requested CAD preview budget.".to_string()
                }),
            });
        }
        Err(err) => {
            warn!(
                "RunMat wasm: chunked geometry scene generation failed; falling back to figure preview: {err}"
            );
        }
    }
    let figure =
        match runmat_runtime::geometry::geometry_preview_figure(&asset, title, figure_options) {
            Ok(figure) => figure,
            Err(err) => {
                return Ok(GeometryPreviewPayload {
                    inspect,
                    scene_kind: "summary".to_string(),
                    figure_handle: None,
                    geometry_scene_handle: None,
                    truncated,
                    preview_message: Some(err),
                });
            }
        };
    ensure_runtime_not_cancelled("geometry preview")?;
    ensure_geometry_preview_deadline(deadline, "geometry preview")?;
    let figure_handle = runtime_import_figure(figure).as_u32();
    Ok(GeometryPreviewPayload {
        inspect,
        scene_kind: "mesh".to_string(),
        figure_handle: Some(figure_handle),
        geometry_scene_handle: None,
        truncated,
        preview_message: truncated
            .then(|| "Preview mesh was limited by the requested CAD preview budget.".to_string()),
    })
}

fn geometry_asset_preview_truncated(asset: &runmat_geometry_core::GeometryAsset) -> bool {
    asset
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "CAD_IMPORT_TESSELLATION_TRUNCATED")
}

#[derive(Debug, Clone, Copy)]
struct GeometryPreviewDeadline {
    started_ms: f64,
    timeout_ms: u64,
}

fn geometry_preview_deadline(
    budget: Option<&GeometryPreviewBudgetPayload>,
) -> Option<GeometryPreviewDeadline> {
    let timeout_ms = budget.and_then(|item| item.timeout_ms)?;
    if timeout_ms == 0 {
        return None;
    }
    Some(GeometryPreviewDeadline {
        started_ms: js_sys::Date::now(),
        timeout_ms,
    })
}

fn ensure_geometry_preview_deadline(
    deadline: Option<GeometryPreviewDeadline>,
    context: &str,
) -> Result<(), String> {
    let Some(deadline) = deadline else {
        return Ok(());
    };
    let elapsed_ms = js_sys::Date::now() - deadline.started_ms;
    if elapsed_ms > deadline.timeout_ms as f64 {
        return Err(format!(
            "{context} timed out after {} ms",
            deadline.timeout_ms
        ));
    }
    Ok(())
}

fn geometry_import_options_from_budget(
    budget: Option<&GeometryPreviewBudgetPayload>,
    max_triangles: Option<u64>,
) -> runmat_geometry_io::GeometryImportOptions {
    runmat_geometry_io::GeometryImportOptions {
        source_units: None,
        max_triangles,
        budget_policy: match budget.and_then(|budget| budget.budget_policy) {
            Some(GeometryPreviewBudgetPolicyPayload::Strict) => {
                runmat_geometry_io::GeometryImportBudgetPolicy::Strict
            }
            _ => runmat_geometry_io::GeometryImportBudgetPolicy::Truncate,
        },
        units: runmat_geometry_core::LengthUnit::Meter,
        tessellation_profile: tessellation_profile_from_budget(budget),
        relative_deflection: true,
    }
}

fn tessellation_profile_from_budget(
    budget: Option<&GeometryPreviewBudgetPayload>,
) -> runmat_geometry_core::TessellationProfile {
    let Some(profile) = budget.and_then(|budget| budget.tessellation_profile.as_ref()) else {
        return runmat_geometry_core::TessellationProfile::default();
    };
    runmat_geometry_core::TessellationProfile {
        profile_id: profile
            .profile_id
            .clone()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| "preview-v1".to_string()),
        chord_tolerance: finite_positive(profile.chord_tolerance),
        angle_tolerance_deg: finite_positive(profile.angle_tolerance_deg),
        healing_mode: Default::default(),
    }
}

fn geometry_preview_figure_options_from_budget(
    budget: Option<&GeometryPreviewBudgetPayload>,
) -> runmat_runtime::geometry::GeometryPreviewFigureOptions {
    let mut options = runmat_runtime::geometry::GeometryPreviewFigureOptions::cad_preview();
    options.xray = budget.and_then(|budget| budget.xray).unwrap_or(false);
    options
}

fn geometry_preview_quality_label(
    budget: Option<&GeometryPreviewBudgetPayload>,
    truncated: bool,
) -> &'static str {
    match budget.and_then(|budget| budget.budget_policy) {
        Some(GeometryPreviewBudgetPolicyPayload::Strict) if !truncated => "complete tessellation",
        Some(GeometryPreviewBudgetPolicyPayload::Strict) => "strict tessellation",
        _ if truncated => "bounded preview",
        _ => "interactive preview",
    }
}

fn finite_positive(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite() && *value > 0.0)
}

async fn check_fea_path(
    path: String,
    session_slot: &RefCell<RunMatSession>,
) -> Result<FeaCheckPayload, String> {
    let mut session = {
        let mut slot = session_slot.borrow_mut();
        std::mem::take(&mut *slot)
    };
    let result = session
        .check_fea_source_path(&std::path::PathBuf::from(path))
        .await;
    *session_slot.borrow_mut() = session;
    result
}

fn load_fea_field(
    run_id: String,
    field_id: String,
    options: FeaFieldRequestPayload,
) -> Result<FeaFieldPayload, String> {
    Ok(runmat_runtime::analysis::analysis_field_page_by_run_id_op(
        &run_id,
        AnalysisFieldSelection::new(field_id, None),
        options,
        runmat_runtime::operations::OperationContext::new(None, None),
    )
    .map_err(|err| err.message)?
    .data)
}

fn filename_for_display(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
}

fn geometry_format_from_path(path: &str) -> String {
    let ext = std::path::PathBuf::from(path)
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "stl" => "stl",
        "step" | "stp" => "step",
        "obj" => "obj",
        "ply" => "ply",
        "gltf" | "glb" => "gltf",
        _ => "unknown",
    }
    .to_string()
}

fn is_supported_geometry_path(path: &str) -> bool {
    geometry_format_from_path(path) != "unknown"
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
enum ExecuteRequestSourcePayload {
    Text { name: String, text: String },
    Path { path: String },
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ExecuteHostPolicyPayload {
    top_level_await: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ExecuteRequestPayload {
    input_manifest: Option<runmat_filesystem::pinned::InputManifest>,
    source: ExecuteRequestSourcePayload,
    compatibility: Option<String>,
    error_namespace: Option<String>,
    host_policy: Option<ExecuteHostPolicyPayload>,
    requested_outputs: Option<u32>,
    retain_figures: Option<bool>,
}

#[wasm_bindgen]
impl RunMatWasm {
    #[wasm_bindgen(js_name = packageCacheSnapshot)]
    pub async fn package_cache_snapshot_js(&self) -> Result<JsValue, JsValue> {
        self.ensure_not_disposed()?;
        let Some(cache) = self.package_cache.as_ref() else {
            return Ok(JsValue::NULL);
        };
        let snapshot = cache
            .snapshot()
            .await
            .map_err(|error| js_error(&format!("Package cache snapshot failed: {error}")))?;
        serde_wasm_bindgen::to_value(&snapshot).map_err(|error| {
            js_error(&format!(
                "Package cache snapshot serialization failed: {error}"
            ))
        })
    }

    #[wasm_bindgen(js_name = executeRequest)]
    pub async fn execute_request_js(&self, request_value: JsValue) -> Result<JsValue, JsValue> {
        self.ensure_not_disposed()?;
        let request_payload: ExecuteRequestPayload = serde_wasm_bindgen::from_value(request_value)
            .map_err(|err| js_error(&format!("executeRequest payload parse failed: {err}")))?;
        // Admit ownership before telemetry or interrupt authority is changed.
        // A second request must never replace the first request's cancel slot.
        let mut session = self.take_session()?;
        let source_for_telemetry = match &request_payload.source {
            ExecuteRequestSourcePayload::Text { text, .. } => text.clone(),
            ExecuteRequestSourcePayload::Path { path } => path.clone(),
        };
        init_logging_once();
        let exec_span = info_span!(
            "runmat.execute",
            source_len = source_for_telemetry.len() as u64,
            disposed = self.disposed.get()
        );
        let _enter = exec_span.enter();
        info!(target = "runmat.runtime", "Execution started");
        let figures_before: Vec<u32> = runtime_figure_handles()
            .into_iter()
            .map(|handle| handle.as_u32())
            .collect();
        let retain_figures = request_payload.retain_figures.unwrap_or(false);
        runtime_reset_hold_state_for_run();

        let telemetry_run = {
            if self.disposed.get() {
                None
            } else {
                let cfg = self.config.borrow();
                if !cfg.telemetry_consent {
                    None
                } else {
                    session.telemetry_run(TelemetryRunConfig {
                        kind: cfg.telemetry_run_kind.clone(),
                        jit_enabled: cfg.enable_jit,
                        accelerate_enabled: self.gpu_status.active,
                    })
                }
            }
        };

        let interrupt_handle = session.interrupt_handle();
        self.active_interrupt
            .borrow_mut()
            .replace(interrupt_handle.clone());
        struct ActiveInterruptGuard<'a> {
            slot: &'a RefCell<Option<runmat_execution::cancellation::CancellationSource>>,
        }
        impl Drop for ActiveInterruptGuard<'_> {
            fn drop(&mut self) {
                self.slot.borrow_mut().take();
            }
        }
        let _active_interrupt_guard = ActiveInterruptGuard {
            slot: &self.active_interrupt,
        };

        let mut request = runmat_core::abi::ExecutionRequest::for_source(
            match request_payload.source {
                ExecuteRequestSourcePayload::Text { name, text } => {
                    runmat_core::abi::SourceInput::Text { name, text }
                }
                ExecuteRequestSourcePayload::Path { path } => {
                    runmat_core::abi::SourceInput::Path(path)
                }
            },
            self.config.borrow().language_compat,
            runmat_core::abi::HostExecutionPolicy::default(),
            session.workspace_handle(),
        );
        let compatibility_override = request_payload
            .compatibility
            .as_deref()
            .map(|compatibility| {
                if let Some(parsed) = parse_language_compat_from_str(compatibility) {
                    request.compatibility = parsed;
                    Ok(parsed)
                } else {
                    Err(js_error(&format!(
                        "executeRequest compatibility is invalid: {compatibility}"
                    )))
                }
            })
            .transpose()?;
        let error_namespace_override = request_error_namespace(
            compatibility_override,
            request_payload.error_namespace.as_deref(),
        );
        if let Some(host_policy) = request_payload.host_policy {
            if let Some(top_level_await) = host_policy.top_level_await {
                request.host_policy.top_level_await = top_level_await;
            }
        }
        if let Some(requested_outputs) = request_payload.requested_outputs {
            request.requested_outputs = match requested_outputs {
                0 => runmat_hir::RequestedOutputCount::Zero,
                1 => runmat_hir::RequestedOutputCount::One,
                count => runmat_hir::RequestedOutputCount::Exactly(count as usize),
            };
        }
        if let Some(namespace) = error_namespace_override.as_deref() {
            session.set_error_namespace(namespace);
        }
        let exec_response = session
            .execute_request_with_inputs(request, request_payload.input_manifest)
            .await;
        if error_namespace_override.is_some() {
            session.set_error_namespace(self.config.borrow().error_namespace.clone());
        }
        drop(session);
        let mut payload = match exec_response.result {
            Ok(outcome) => {
                if !retain_figures
                    && !outcome.diagnostics.iter().any(|diagnostic| {
                        diagnostic.severity == runmat_core::abi::DiagnosticSeverity::Error
                    })
                {
                    let touched: std::collections::HashSet<u32> =
                        outcome.figures_touched.iter().copied().collect();
                    for handle in figures_before {
                        if !touched.contains(&handle) {
                            let _ = runtime_close_figure(Some(FigureHandle::from(handle)));
                        }
                    }
                }
                outcome.with_owned(|outcome| {
                    ExecutionPayload::from_outcome(outcome, &exec_response.source_context)
                })
            }
            Err(err) => ExecutionPayload {
                dependency_lock: None,
                flow: serde_json::json!({ "kind": "no-value" }),
                value_text: None,
                value_json: None,
                type_info: None,
                execution_time_ms: 0,
                used_jit: false,
                error: Some(run_error_payload(
                    &err,
                    Some(exec_response.source_context.source_name()),
                    exec_response.source_context.source_text(),
                )),
                stdout: Default::default(),
                display_events: Vec::new(),
                workspace: WorkspacePayload {
                    full: false,
                    version: 0,
                    values: Vec::new(),
                    removals: Vec::new(),
                },
                figures_touched: Vec::new(),
                warnings: Vec::new(),
                stdin_events: Vec::new(),
                profiling: None,
                fusion_plan: None,
            },
        };
        payload.dependency_lock = exec_response.dependency_lock;
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

    #[wasm_bindgen(js_name = previewGeometry)]
    pub async fn preview_geometry_js(
        &self,
        path: String,
        budget_value: JsValue,
    ) -> Result<JsValue, JsValue> {
        self.ensure_not_disposed()?;
        let interrupt_handle = self.session.borrow().interrupt_handle();
        let _interrupt_guard = install_wasm_interrupt(&self.active_interrupt, interrupt_handle);
        let budget = if budget_value.is_null() || budget_value.is_undefined() {
            None
        } else {
            Some(
                serde_wasm_bindgen::from_value::<GeometryPreviewBudgetPayload>(budget_value)
                    .map_err(|err| {
                        js_error(&format!("previewGeometry budget parse failed: {err}"))
                    })?,
            )
        };
        let payload = preview_geometry_path(path, budget)
            .await
            .map_err(|err| js_error(&err))?;
        serde_wasm_bindgen::to_value(&payload)
            .map_err(|err| js_error(&format!("Failed to serialize geometry preview: {err}")))
    }

    #[wasm_bindgen(js_name = disposeGeometryPreview)]
    pub fn dispose_geometry_preview_js(
        &self,
        figure_handle: Option<u32>,
        geometry_scene_handle: Option<u32>,
    ) -> Result<(), JsValue> {
        self.ensure_not_disposed()?;
        if let Some(figure_handle) = figure_handle {
            runtime_close_figure(Some(FigureHandle::from(figure_handle)))
                .map_err(|err| js_error(&format!("disposeGeometryPreview failed: {err}")))?;
        }
        if let Some(geometry_scene_handle) = geometry_scene_handle {
            runtime_close_geometry_scene(geometry_scene_handle);
        }
        Ok(())
    }

    #[wasm_bindgen(js_name = feaCapabilities)]
    pub fn fea_capabilities_js(&self) -> Result<JsValue, JsValue> {
        self.ensure_not_disposed()?;
        let owned_host_ready = cfg!(feature = "occt-wasm-host")
            && ["__runmatOcctExecute", "__runmatExecutePreparedFeaStudy"]
                .iter()
                .all(|name| {
                    js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str(name))
                        .is_ok_and(|value| value.is_function())
                });
        let execution = if owned_host_ready {
            runmat_runtime::analysis::AnalysisExecutionCapability::Local {
                live_progress: true,
            }
        } else {
            runmat_runtime::analysis::AnalysisExecutionCapability::CheckOnly
        };
        let payload = FeaCapabilitiesPayload::for_runtime(
            execution,
            [".stl", ".step", ".stp", ".obj", ".ply", ".gltf", ".glb"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
        );
        crate::wire::json_contract_to_js(&payload)
            .map_err(|err| js_error(&format!("Failed to serialize FEA capabilities: {err}")))
    }

    #[wasm_bindgen(js_name = checkFeaSource)]
    pub async fn check_fea_source_js(
        &self,
        path: String,
        inputs: JsValue,
    ) -> Result<JsValue, JsValue> {
        self.ensure_not_disposed()?;
        let (filesystem, path) =
            crate::runtime::filesystem::captured::CapturedFilesystem::simulation(&path, inputs)?;
        let payload = filesystem
            .run(check_fea_path(path, &self.session))
            .await?
            .map_err(|err| js_error(&err))?;
        crate::wire::json_contract_to_js(&payload)
            .map_err(|err| js_error(&format!("Failed to serialize FEA check result: {err}")))
    }

    #[wasm_bindgen(js_name = checkFeaWorkspace)]
    pub async fn check_fea_workspace_js(&self, request: JsValue) -> Result<JsValue, JsValue> {
        self.ensure_not_disposed()?;
        let request: runmat_core::FeaWorkspaceCheckRequest =
            serde_wasm_bindgen::from_value(request)
                .map_err(|error| js_error(&format!("Invalid candidate workspace: {error}")))?;
        let mut session = {
            let mut slot = self.session.borrow_mut();
            std::mem::take(&mut *slot)
        };
        let result = session.check_fea_workspace(request).await;
        *self.session.borrow_mut() = session;
        let result = result.map_err(|error| js_error(&error))?;
        crate::wire::json_contract_to_js(&result)
            .map_err(|error| js_error(&format!("Failed to serialize candidate check: {error}")))
    }

    #[wasm_bindgen(js_name = checkFeaWorkingSource)]
    pub async fn check_fea_working_source_js(&self, request: JsValue) -> Result<JsValue, JsValue> {
        self.ensure_not_disposed()?;
        let request: runmat_core::FeaWorkingCheckRequest = serde_wasm_bindgen::from_value(request)
            .map_err(|error| js_error(&format!("Invalid working source: {error}")))?;
        let mut session = {
            let mut slot = self.session.borrow_mut();
            std::mem::take(&mut *slot)
        };
        let result = session.check_fea_working_source(request).await;
        *self.session.borrow_mut() = session;
        let result = result.map_err(|error| js_error(&error))?;
        crate::wire::json_contract_to_js(&result).map_err(|error| js_error(&error.to_string()))
    }

    #[wasm_bindgen(js_name = inspectFeaChanges)]
    pub async fn inspect_fea_changes_js(&self, id: JsValue) -> Result<JsValue, JsValue> {
        self.ensure_not_disposed()?;
        let id: runmat_fea_source::FeaChangeSetId =
            serde_wasm_bindgen::from_value(id).map_err(|error| js_error(&error.to_string()))?;
        let mut session = {
            let mut slot = self.session.borrow_mut();
            std::mem::take(&mut *slot)
        };
        let result = session.inspect_fea_changes(id).await;
        *self.session.borrow_mut() = session;
        let result = result.map_err(|error| js_error(&error))?;
        crate::wire::json_contract_to_js(&result).map_err(|error| js_error(&error.to_string()))
    }

    #[wasm_bindgen(js_name = reviewFeaChanges)]
    pub async fn review_fea_changes_js(&self, request: JsValue) -> Result<JsValue, JsValue> {
        self.ensure_not_disposed()?;
        let request: runmat_fea_source::FeaChangeSetReviewRequest =
            serde_wasm_bindgen::from_value(request)
                .map_err(|error| js_error(&error.to_string()))?;
        let session = {
            let mut slot = self.session.borrow_mut();
            std::mem::take(&mut *slot)
        };
        let result = session.review_fea_changes(request).await;
        *self.session.borrow_mut() = session;
        let result = result.map_err(|error| js_error(&error))?;
        crate::wire::json_contract_to_js(&result).map_err(|error| js_error(&error.to_string()))
    }

    #[wasm_bindgen(js_name = stageFeaChanges)]
    pub async fn stage_fea_changes_js(&self, request: JsValue) -> Result<JsValue, JsValue> {
        self.ensure_not_disposed()?;
        let request: runmat_fea_source::FeaChangeSetRequest =
            serde_wasm_bindgen::from_value(request)
                .map_err(|error| js_error(&error.to_string()))?;
        let mut session = {
            let mut slot = self.session.borrow_mut();
            std::mem::take(&mut *slot)
        };
        let result = session.stage_fea_changes(request).await;
        *self.session.borrow_mut() = session;
        let result = result.map_err(|error| js_error(&error))?;
        crate::wire::json_contract_to_js(&result).map_err(|error| js_error(&error.to_string()))
    }

    #[wasm_bindgen(js_name = publishFeaChanges)]
    pub async fn publish_fea_changes_js(&self, id: JsValue) -> Result<JsValue, JsValue> {
        self.ensure_not_disposed()?;
        let id: runmat_fea_source::FeaChangeSetId =
            serde_wasm_bindgen::from_value(id).map_err(|error| js_error(&error.to_string()))?;
        let mut session = {
            let mut slot = self.session.borrow_mut();
            std::mem::take(&mut *slot)
        };
        let result = session.publish_fea_changes(id).await;
        *self.session.borrow_mut() = session;
        let result = result.map_err(|error| js_error(&error))?;
        crate::wire::json_contract_to_js(&result).map_err(|error| js_error(&error.to_string()))
    }

    #[wasm_bindgen(js_name = discardFeaChanges)]
    pub async fn discard_fea_changes_js(&self, id: JsValue) -> Result<JsValue, JsValue> {
        self.ensure_not_disposed()?;
        let id: runmat_fea_source::FeaChangeSetId =
            serde_wasm_bindgen::from_value(id).map_err(|error| js_error(&error.to_string()))?;
        let session = {
            let mut slot = self.session.borrow_mut();
            std::mem::take(&mut *slot)
        };
        let result = session.discard_fea_changes(id).await;
        *self.session.borrow_mut() = session;
        let result = result.map_err(|error| js_error(&error))?;
        crate::wire::json_contract_to_js(&result).map_err(|error| js_error(&error.to_string()))
    }

    #[wasm_bindgen(js_name = browseFeaSource)]
    pub fn browse_fea_source_js(&self, request: JsValue) -> Result<JsValue, JsValue> {
        self.ensure_not_disposed()?;
        let request: runmat_runtime::analysis::FeaBrowseRequest =
            serde_wasm_bindgen::from_value(request)
                .map_err(|error| js_error(&format!("Invalid FEA browse request: {error}")))?;
        let result = runmat_runtime::analysis::FeaBrowseOutcome::from(
            self.session.borrow_mut().browse_fea_source(&request),
        );
        crate::wire::json_contract_to_js(&result).map_err(|error| js_error(&error.to_string()))
    }

    #[wasm_bindgen(js_name = feaField)]
    pub async fn fea_field_js(
        &self,
        run_id: String,
        field_id: String,
        options_value: JsValue,
    ) -> Result<JsValue, JsValue> {
        self.ensure_not_disposed()?;
        let options = if options_value.is_null() || options_value.is_undefined() {
            FeaFieldRequestPayload::default()
        } else {
            serde_wasm_bindgen::from_value::<FeaFieldRequestPayload>(options_value)
                .map_err(|err| js_error(&format!("FEA field options parse failed: {err}")))?
        };
        let payload = runmat_runtime::analysis::storage::with_artifact_access(false, |_| {
            load_fea_field(run_id, field_id, options)
        })
        .await
        .map_err(|error| js_error(&error.to_string()))?
        .map_err(|err| js_error(&err))?;
        self.ensure_not_disposed()?;
        serde_wasm_bindgen::to_value(&payload)
            .map_err(|err| js_error(&format!("Failed to serialize FEA field: {err}")))
    }

    #[wasm_bindgen(js_name = resetSession)]
    pub fn reset_session(&self) -> Result<(), JsValue> {
        self.ensure_not_disposed()?;
        if self.in_flight_session.get() {
            return Err(js_error(
                "Cannot reset a RunMat session during an in-flight request",
            ));
        }
        wasm_registry::register_all();
        log::debug!("RunMat wasm: resetting execution session");
        let installed_handoff = self.session.borrow().project_handoff().cloned();
        let config = self.config.borrow();
        let consent = config.telemetry_consent;
        let mut session = RunMatSession::with_options(config.enable_jit, config.verbose)
            .map_err(|err| js_error(&format!("Failed to reset session: {err}")))?;
        session
            .apply_runtime_config(&config.runtime)
            .map_err(|error| js_error(&error.to_string()))?;
        if let Some(settings) = &self.application_settings {
            session
                .install_application_settings(settings.clone())
                .map_err(|err| {
                    js_error(&format!("Failed to restore application settings: {err}"))
                })?;
        }
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
        session.install_execution_services(self.execution_service.clone());
        session.install_analysis_service(std::rc::Rc::new(
            super::fea_analysis_host::BrowserAnalysisService {
                host: self.fea_admission.clone(),
                active: self.active_fea_lease.clone(),
                config: self.config.clone(),
            },
        ));
        session.install_fea_run_admission_service(std::rc::Rc::new(
            super::fea_async_host::BrowserFeaAdmission {
                host: self.fea_admission.clone(),
                active: self.active_fea_lease.clone(),
                pending: self.pending_fea_admission.clone(),
            },
        ));
        if self.execution_service.has_owned_cpu_worker() {
            session.install_owned_builtin_work_service(self.execution_service.clone());
        }
        if let Some(host) = self.dialog_host.borrow().clone() {
            session.install_host_service(host);
        }
        if let Some(handoff) = installed_handoff {
            session
                .install_project_handoff(handoff)
                .map_err(|err| js_error(&format!("Failed to restore project handoff: {err}")))?;
        }
        if self.telemetry_sink.is_some() {
            session.set_telemetry_platform_info(TelemetryPlatformInfo {
                os: Some("web".to_string()),
                arch: Some("wasm32".to_string()),
            });
            session.set_telemetry_sink(self.telemetry_sink.clone());
        }
        #[cfg(target_arch = "wasm32")]
        {
            *self.local_tasks.borrow_mut() = session.browser_local_task_supervisor();
        }
        let mut slot = self.session.borrow_mut();
        *slot = session;
        #[cfg(target_arch = "wasm32")]
        if let Some(handler) = self.input_handler.borrow().as_ref() {
            handler.install_on(&mut slot);
        }
        runtime_reset_plot_state();
        runtime_invalidate_surface_revisions();
        Ok(())
    }

    #[wasm_bindgen(js_name = installProjectHandoff)]
    pub fn install_project_handoff_js(&self, value: JsValue) -> Result<JsValue, JsValue> {
        self.ensure_not_disposed()?;
        let handoff: runmat_package::FrozenProjectHandoff =
            serde_wasm_bindgen::from_value(value)
                .map_err(|err| js_error(&format!("Project handoff parse failed: {err}")))?;
        let revision = self
            .session
            .borrow_mut()
            .install_project_handoff(handoff)
            .map_err(|err| js_error(&format!("Project handoff validation failed: {err}")))?;
        serde_wasm_bindgen::to_value(&revision)
            .map_err(|err| js_error(&format!("Project revision serialization failed: {err}")))
    }

    #[wasm_bindgen(js_name = clearProjectHandoff)]
    pub fn clear_project_handoff_js(&self) -> Result<(), JsValue> {
        self.ensure_not_disposed()?;
        let mut session = self.session.borrow_mut();
        session.clear_captured_project();
        session
            .clear_project_handoff()
            .map_err(|err| js_error(&format!("Project handoff clear failed: {err}")))?;
        Ok(())
    }

    #[wasm_bindgen(js_name = projectRevision)]
    pub fn project_revision_js(&self) -> Result<JsValue, JsValue> {
        self.ensure_not_disposed()?;
        let Some(revision) = self.session.borrow().project_revision() else {
            return Ok(JsValue::NULL);
        };
        serde_wasm_bindgen::to_value(&revision)
            .map_err(|err| js_error(&format!("Project revision serialization failed: {err}")))
    }

    /// Return semantic discovery for the exact caller-frozen run snapshot.
    /// The Core session verifies it against the installed project handoff
    /// before static analysis; no browser filesystem rediscovery occurs.
    #[wasm_bindgen(js_name = discoverTests)]
    pub fn discover_tests_js(&self, value: JsValue) -> Result<JsValue, JsValue> {
        self.ensure_not_disposed()?;
        let snapshot: runmat_test::discovery::FrozenTestRunSnapshot =
            serde_wasm_bindgen::from_value(value)
                .map_err(|err| js_error(&format!("Test snapshot parse failed: {err}")))?;
        let discovery = self
            .session
            .borrow()
            .discover_tests(&snapshot)
            .map_err(|err| js_error(&format!("Test discovery failed: {err}")))?;
        serde_wasm_bindgen::to_value(&discovery)
            .map_err(|err| js_error(&format!("Test discovery serialization failed: {err}")))
    }

    /// Discover and construct the selected immutable plan through the same
    /// Core authority used by native hosts.
    #[wasm_bindgen(js_name = prepareTests)]
    pub fn prepare_tests_js(
        &self,
        snapshot: JsValue,
        selector: JsValue,
    ) -> Result<JsValue, JsValue> {
        self.ensure_not_disposed()?;
        let snapshot: runmat_test::discovery::FrozenTestRunSnapshot =
            serde_wasm_bindgen::from_value(snapshot)
                .map_err(|err| js_error(&format!("Test snapshot parse failed: {err}")))?;
        let selector: runmat_test::descriptor::TestSelector =
            serde_wasm_bindgen::from_value(selector)
                .map_err(|err| js_error(&format!("Test selector parse failed: {err}")))?;
        let prepared = self
            .session
            .borrow()
            .prepare_tests(&snapshot, &selector)
            .map_err(|err| js_error(&format!("Test preparation failed: {err}")))?;
        serde_wasm_bindgen::to_value(&prepared)
            .map_err(|err| js_error(&format!("Test preparation serialization failed: {err}")))
    }

    #[wasm_bindgen(js_name = cancelExecution)]
    pub fn cancel_execution(&self) {
        let active = self.active_interrupt.borrow().clone();
        if let Some(flag) = active {
            flag.cancel(runmat_execution::CancellationReason::User);
            // An executing session is temporarily taken out of its RefCell.
            // Its installed task service remains the owner of child attempts.
            self.execution_service
                .drain_scope(runmat_execution::CancellationReason::User);
            return;
        }
        if !self.in_flight_session.get() {
            self.session.borrow().cancel_execution();
        }
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
            if !self.in_flight_session.get() {
                self.session.borrow_mut().set_compat_mode(parsed);
            }
        } else {
            warn!("RunMat wasm: ignoring unknown language compat mode '{mode}'");
        }
    }

    #[wasm_bindgen(js_name = "setErrorNamespace")]
    pub fn set_error_namespace(&self, namespace: String) {
        if self.disposed.get() {
            return;
        }
        self.config.borrow_mut().error_namespace = namespace.clone();
        if !self.in_flight_session.get() {
            self.session.borrow_mut().set_error_namespace(namespace);
        }
    }

    #[wasm_bindgen(js_name = setInputHandler)]
    pub fn set_input_handler(&self, handler: JsValue) -> Result<(), JsValue> {
        self.ensure_not_disposed()?;
        #[cfg(target_arch = "wasm32")]
        {
            if handler.is_null() || handler.is_undefined() {
                self.session.borrow_mut().clear_async_input_handler();
                self.input_handler.borrow_mut().take();
            } else {
                let func = handler
                    .dyn_into::<js_sys::Function>()
                    .map_err(|_| js_error("setInputHandler expects a Function or null"))?;
                let registration = JsInputHandler::new(func);
                registration.install_on(&mut self.session.borrow_mut());
                self.input_handler.replace(Some(registration));
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
        if self.disposed.get() {
            return;
        }
        if !self.in_flight_session.get() {
            self.session.borrow_mut().set_emit_fusion_plan(enabled);
        }
        if let Ok(mut cfg) = self.config.try_borrow_mut() {
            cfg.emit_fusion_plan = enabled;
        }
    }

    #[wasm_bindgen(js_name = fusionPlanForSource)]
    pub fn fusion_plan_for_source(&self, source: String) -> Result<JsValue, JsValue> {
        self.ensure_not_disposed()?;
        let mut session = self.session.borrow_mut();
        let snapshot = session
            .compile_fusion_plan(&source)
            .map_err(|err| run_error_to_js(&err, Some("<fusion_plan>"), Some(&source)))?;
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
            let mut session = self.take_session()?;

            let materialize_result = session.materialize_variable(target, opts).await;
            drop(session);

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
        self.ensure_not_disposed()?;
        let payload = {
            let session = self.session.borrow();
            StatsPayload::from(session.stats())
        };
        serde_wasm_bindgen::to_value(&payload)
            .map_err(|err| js_error(&format!("Failed to serialize stats: {err}")))
    }

    #[wasm_bindgen(js_name = clearWorkspace)]
    pub fn clear_workspace(&self) -> Result<(), JsValue> {
        if self.disposed.get() {
            return Ok(());
        }
        if self.in_flight_session.get() {
            self.clear_workspace_after_flight.set(true);
        } else {
            self.session
                .borrow_mut()
                .clear_variables()
                .map_err(|err| runtime_error_to_js(&err))?;
        }
        Ok(())
    }

    #[wasm_bindgen(js_name = exportWorkspaceState)]
    pub async fn export_workspace_state(
        &self,
        include_variables: Option<String>,
    ) -> Result<Option<Vec<u8>>, JsValue> {
        self.ensure_not_disposed()?;
        let mode = parse_workspace_export_mode(include_variables.as_deref());

        let mut session = self.take_session()?;
        let result = session.export_workspace_state(mode).await;
        drop(session);

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
            WorkspacePayload::from(
                session
                    .workspace_snapshot()
                    .map_err(|err| runtime_error_to_js(&err))?,
            )
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
        let values = data_array_preview_json(&payload.values, limit);
        let value_text = values
            .iter()
            .map(json_value_text)
            .collect::<Vec<_>>()
            .join(", ");
        let response = DataMaterializedVariablePayload {
            name: array.into(),
            class_name: infer_dataset_class_name(meta.shape.len()).to_string(),
            dtype: Some(meta.dtype.clone()),
            shape: meta.shape.clone(),
            is_gpu: false,
            residency: WorkspaceResidency::Cpu.as_str(),
            size_bytes: Some(estimate_data_array_bytes(&meta.shape, &meta.dtype)),
            preview: Some(DataArrayPreviewPayload {
                values: values.clone(),
                truncated: values.len() < total_elements,
            }),
            value_text,
            value_json: JsonValue::Array(values),
        };
        serde_wasm_bindgen::to_value(&response).map_err(|err| {
            js_error(&format!(
                "Failed to serialize data materialized value: {err}"
            ))
        })
    }

    #[wasm_bindgen(js_name = exportFigureScene)]
    pub async fn export_figure_scene(
        &self,
        handle: u32,
        scene_budget_bytes: Option<u32>,
    ) -> Result<Option<Vec<u8>>, JsValue> {
        self.ensure_not_disposed()?;
        log::debug!("RunMat wasm: exportFigureScene start handle={handle}");
        let export_scene = |figure_handle: FigureHandle| async move {
            match scene_budget_bytes {
                Some(bytes) => {
                    let policy =
                        runmat_plot::event::resolve_scene_export_policy(Some(bytes as usize));
                    runmat_runtime::builtins::plotting::export_figure_scene_with_policy(
                        figure_handle,
                        policy,
                    )
                    .await
                }
                None => {
                    runmat_runtime::builtins::plotting::export_figure_scene(figure_handle).await
                }
            }
        };
        match export_scene(FigureHandle::from(handle)).await {
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
                Ok(None)
            }
            Err(err) => {
                warn!("RunMat wasm: figure scene export rejected: {err}");
                Err(runtime_error_to_js(&err))
            }
        }
    }

    #[wasm_bindgen(js_name = exportGeometryScene)]
    pub fn export_geometry_scene(&self, handle: u32) -> Result<Option<Vec<u8>>, JsValue> {
        self.ensure_not_disposed()?;
        match runtime_export_geometry_scene(handle) {
            Ok(payload) => Ok(payload),
            Err(err) => {
                warn!("RunMat wasm: geometry scene export rejected: {err}");
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

    #[wasm_bindgen(js_name = importGeometryScene)]
    pub fn import_geometry_scene(&self, scene: &[u8]) -> Result<Option<u32>, JsValue> {
        self.ensure_not_disposed()?;
        log::debug!(
            "RunMat wasm: importGeometryScene start bytes={}",
            scene.len()
        );
        match runtime_import_geometry_scene_payload(scene) {
            Ok(handle) => Ok(handle),
            Err(err) => {
                if err.identifier() == Some(ReplayErrorKind::DecodeFailed.identifier()) {
                    warn!("RunMat wasm: geometry scene decode failed: {err}");
                } else {
                    warn!("RunMat wasm: geometry scene import rejected: {err}");
                }
                Err(runtime_error_to_js(&err))
            }
        }
    }

    #[wasm_bindgen(js_name = createFeaFieldScene)]
    pub async fn create_fea_field_scene(
        &self,
        run_id: &str,
        field_id: Option<String>,
        shape_mode: &str,
        exaggeration: f64,
        show_undeformed_outline: bool,
        view: JsValue,
        replace_handle: Option<u32>,
    ) -> Result<JsValue, JsValue> {
        self.ensure_not_disposed()?;
        let view = if view.is_null() || view.is_undefined() {
            None
        } else {
            Some(
                serde_wasm_bindgen::from_value(view)
                    .map_err(|error| JsValue::from_str(&format!("invalid FEA view: {error}")))?,
            )
        };
        let (mut scene, deformation_scale, exact) =
            runmat_runtime::analysis::storage::with_artifact_access(false, |_| {
                runmat_runtime::analysis::field_scene::build_scene_view(
                    run_id,
                    field_id.as_deref(),
                    shape_mode,
                    exaggeration,
                    show_undeformed_outline,
                    view,
                )
            })
            .await
            .map_err(|error| js_error(&error.to_string()))?
            .map_err(|error| JsValue::from_str(&error))?;
        self.ensure_not_disposed()?;
        let handle = if let Some(handle) = replace_handle {
            let previous = runmat_runtime::builtins::plotting::clone_geometry_scene(handle)
                .ok_or_else(|| {
                    JsValue::from_str(&format!("geometry scene handle {handle} does not exist"))
                })?;
            scene.revision = previous.revision.saturating_add(1);
            runmat_runtime::builtins::plotting::replace_geometry_scene(handle, scene)
                .map_err(|error| JsValue::from_str(&error.to_string()))?;
            handle
        } else {
            runtime_import_geometry_scene(scene)
                .map_err(|error| JsValue::from_str(&error.to_string()))?
        };
        serde_wasm_bindgen::to_value(&serde_json::json!({
            "handle": handle,
            "deformationScale": deformation_scale,
            "exact": exact,
        }))
        .map_err(|error| JsValue::from_str(&error.to_string()))
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
        self.config.borrow().telemetry_consent
    }

    #[wasm_bindgen(js_name = gpuStatus)]
    pub fn gpu_status(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.gpu_status)
            .map_err(|err| js_error(&format!("Failed to serialize GPU status: {err}")))
    }

    #[wasm_bindgen(js_name = telemetryClientId)]
    pub fn telemetry_client_id(&self) -> Option<String> {
        self.config.borrow().telemetry_client_id.clone()
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
        self.ensure_not_disposed()?;
        let host = install_fs_provider_value(bindings).map_err(|err| {
            js_error(&format!(
                "Failed to install filesystem provider: {}",
                js_value_to_string(err)
            ))
        })?;
        self.session.borrow_mut().install_host_service(host.clone());
        self.dialog_host.replace(Some(host));
        Ok(())
    }

    #[wasm_bindgen(js_name = dispose)]
    pub fn dispose(&self) {
        if self.disposed.replace(true) {
            return;
        }
        self.execution_service
            .drain_scope(runmat_execution::CancellationReason::Shutdown);
        if let Some(active) = self.active_interrupt.borrow().as_ref() {
            active.cancel(runmat_execution::CancellationReason::Shutdown);
        }
        runmat_runtime::execution::RuntimeExecutionServices::drain_scope(
            self.execution_service.as_ref(),
            runmat_execution::CancellationReason::Shutdown,
        );
        #[cfg(target_arch = "wasm32")]
        self.local_tasks.borrow().shutdown();
        if !self.in_flight_session.get() {
            let mut session = self.session.borrow_mut();
            session.cancel_execution();
            session.shutdown_local_tasks();
            if let Err(err) = session.clear_variables() {
                warn!("RunMat wasm: disposed workspace cleanup failed: {err}");
            }
        } else {
            // `take_session` leaves a fresh placeholder in this slot. Its
            // unrelated task owner must be terminal too.
            self.session.borrow().shutdown_local_tasks();
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.input_handler.borrow_mut().take();
            clear_figure_event_callback_state();
        }
    }
}

fn data_array_preview_json(values: &DataArrayValues, limit: usize) -> Vec<JsonValue> {
    macro_rules! integer_values {
        ($values:expr, $variant:ident) => {
            $values
                .iter()
                .take(limit)
                .map(|value| integer_json_value(&runmat_value::IntValue::$variant(*value)))
                .collect()
        };
    }

    match values {
        DataArrayValues::F64(values) => values
            .iter()
            .take(limit)
            .copied()
            .map(JsonValue::from)
            .collect(),
        DataArrayValues::F32(values) => values
            .iter()
            .take(limit)
            .copied()
            .map(JsonValue::from)
            .collect(),
        DataArrayValues::I8(values) => integer_values!(values, I8),
        DataArrayValues::I16(values) => integer_values!(values, I16),
        DataArrayValues::I32(values) => integer_values!(values, I32),
        DataArrayValues::I64(values) => integer_values!(values, I64),
        DataArrayValues::U8(values) => integer_values!(values, U8),
        DataArrayValues::U16(values) => integer_values!(values, U16),
        DataArrayValues::U32(values) => integer_values!(values, U32),
        DataArrayValues::U64(values) => integer_values!(values, U64),
    }
}

fn json_value_text(value: &JsonValue) -> String {
    value
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| value.to_string())
}

#[cfg(test)]
mod data_array_preview_tests {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn single_preview_preserves_values_and_limit() {
        let preview = data_array_preview_json(&DataArrayValues::F32(vec![1.25, -3.5, 9.0]), 2);
        assert_eq!(
            preview,
            vec![JsonValue::from(1.25_f32), JsonValue::from(-3.5_f32)]
        );
    }
}
