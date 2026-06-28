use std::cell::{Cell, RefCell};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

use log::warn;
use runmat_core::RunMatSession;
use runmat_core::{
    TelemetryFailureInfo, TelemetryHost, TelemetryPlatformInfo, TelemetryRunConfig,
    TelemetryRunFinish, TelemetrySink, WorkspaceEntry, WorkspaceResidency,
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
    active_interrupt: RefCell<Option<Arc<AtomicBool>>>,
    telemetry_sink: Option<Arc<dyn TelemetrySink>>,
}

struct WasmInterruptGuard<'a> {
    slot: &'a RefCell<Option<Arc<AtomicBool>>>,
    _runtime_guard: runmat_runtime::interrupt::InterruptGuard,
}

impl Drop for WasmInterruptGuard<'_> {
    fn drop(&mut self) {
        self.slot.borrow_mut().take();
    }
}

fn install_wasm_interrupt(
    slot: &RefCell<Option<Arc<AtomicBool>>>,
    handle: Arc<AtomicBool>,
) -> WasmInterruptGuard<'_> {
    handle.store(false, Ordering::Relaxed);
    slot.borrow_mut().replace(Arc::clone(&handle));
    let runtime_guard = runmat_runtime::interrupt::replace_interrupt(Some(handle));
    WasmInterruptGuard {
        slot,
        _runtime_guard: runtime_guard,
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

async fn inspect_geometry_path(
    path: String,
    budget: Option<GeometryPreviewBudgetPayload>,
) -> Result<GeometryInspectPayload, String> {
    inspect_geometry_path_with_asset(path, budget)
        .await
        .map(|result| result.inspect)
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
        max_triangles,
        budget_policy: match budget.and_then(|budget| budget.budget_policy) {
            Some(GeometryPreviewBudgetPolicyPayload::Strict) => {
                runmat_geometry_io::GeometryImportBudgetPolicy::Strict
            }
            _ => runmat_geometry_io::GeometryImportBudgetPolicy::Truncate,
        },
        units: runmat_geometry_core::UnitSystem::Meter,
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

async fn check_fea_path(path: String) -> Result<FeaCheckPayload, String> {
    let document = runmat_runtime::analysis::load_fea_document_from_path_async(
        &std::path::PathBuf::from(&path),
    )
    .await?;
    match document {
        runmat_runtime::analysis::FeaResolvedDocument::Study(study) => {
            let validation = runmat_runtime::analysis::analysis_validate_study_op(
                &study,
                runmat_runtime::operations::OperationContext::new(None, None),
            )
            .map_err(|err| err.message)?
            .data;
            let plan = if validation.valid {
                Some(
                    runmat_runtime::analysis::analysis_plan_study_op(
                        &study,
                        runmat_runtime::operations::OperationContext::new(None, None),
                    )
                    .map_err(|err| err.message)?
                    .data,
                )
            } else {
                None
            };
            let mut evidence_paths = vec![validation.evidence_artifact_path.clone()];
            if let Some(plan) = plan.as_ref() {
                evidence_paths.push(plan.evidence_artifact_path.clone());
            }
            Ok(FeaCheckPayload {
                path,
                document_kind: "study".to_string(),
                valid: validation.valid,
                validation: serde_json::to_value(validation).map_err(|err| err.to_string())?,
                plan: plan
                    .map(serde_json::to_value)
                    .transpose()
                    .map_err(|err| err.to_string())?,
                diagnostics: Vec::new(),
                evidence_artifact_paths: evidence_paths,
            })
        }
        runmat_runtime::analysis::FeaResolvedDocument::Sweep(sweep) => {
            let validation = runmat_runtime::analysis::analysis_validate_study_sweep_op(
                &sweep,
                runmat_runtime::operations::OperationContext::new(None, None),
            )
            .map_err(|err| err.message)?
            .data;
            let plan = if validation.valid {
                Some(
                    runmat_runtime::analysis::analysis_plan_study_sweep_op(
                        &sweep,
                        runmat_runtime::operations::OperationContext::new(None, None),
                    )
                    .map_err(|err| err.message)?
                    .data,
                )
            } else {
                None
            };
            let mut evidence_paths = vec![validation.evidence_artifact_path.clone()];
            if let Some(plan) = plan.as_ref() {
                evidence_paths.push(plan.evidence_artifact_path.clone());
            }
            Ok(FeaCheckPayload {
                path,
                document_kind: "sweep".to_string(),
                valid: validation.valid,
                validation: serde_json::to_value(validation).map_err(|err| err.to_string())?,
                plan: plan
                    .map(serde_json::to_value)
                    .transpose()
                    .map_err(|err| err.to_string())?,
                diagnostics: Vec::new(),
                evidence_artifact_paths: evidence_paths,
            })
        }
    }
}

async fn run_fea_path(path: String) -> Result<FeaRunPayload, String> {
    ensure_runtime_not_cancelled("FEA run")?;
    let document = runmat_runtime::analysis::load_fea_document_from_path_async(
        &std::path::PathBuf::from(&path),
    )
    .await?;
    ensure_runtime_not_cancelled("FEA run")?;
    match document {
        runmat_runtime::analysis::FeaResolvedDocument::Study(study) => {
            let run = runmat_runtime::analysis::analysis_run_study_op(
                &study,
                runmat_runtime::operations::OperationContext::new(None, None),
            )
            .map_err(operation_error_message)?
            .data;
            ensure_runtime_not_cancelled("FEA run")?;
            let results = runmat_runtime::analysis::analysis_results_by_run_id_op(
                &run.run_id,
                runmat_runtime::analysis::AnalysisResultsQuery::metadata_only(),
                runmat_runtime::operations::OperationContext::new(None, None),
            )
            .ok()
            .map(|envelope| envelope.data);
            let field_descriptors = results
                .as_ref()
                .map(serialize_field_descriptors)
                .transpose()?
                .unwrap_or_default();
            let result_summary = results
                .as_ref()
                .map(|data| serde_json::to_value(&data.summary))
                .transpose()
                .map_err(|err| err.to_string())?;
            let (figure_handles, diagnostics) = generated_fea_figure_handles(&study, &run.run_id)?;
            Ok(FeaRunPayload {
                path,
                document_kind: "study".to_string(),
                run: serde_json::to_value(run).map_err(|err| err.to_string())?,
                results: results
                    .map(serde_json::to_value)
                    .transpose()
                    .map_err(|err| err.to_string())?,
                field_descriptors,
                result_summary,
                figure_handles,
                artifact_manifest: None,
                diagnostics,
                progress_events: Vec::new(),
            })
        }
        runmat_runtime::analysis::FeaResolvedDocument::Sweep(sweep) => {
            let run = runmat_runtime::analysis::analysis_run_study_sweep_op(
                &sweep,
                runmat_runtime::operations::OperationContext::new(None, None),
            )
            .map_err(operation_error_message)?
            .data;
            Ok(FeaRunPayload {
                path,
                document_kind: "sweep".to_string(),
                run: serde_json::to_value(run).map_err(|err| err.to_string())?,
                results: None,
                field_descriptors: Vec::new(),
                result_summary: None,
                figure_handles: Vec::new(),
                artifact_manifest: None,
                diagnostics: Vec::new(),
                progress_events: Vec::new(),
            })
        }
    }
}

fn generated_fea_figure_handles(
    study: &runmat_runtime::analysis::AnalysisStudySpec,
    run_id: &str,
) -> Result<(Vec<u32>, Vec<JsonValue>), String> {
    let generated = runmat_runtime::analysis::analysis_generate_study_run_figures(
        study,
        run_id,
        runmat_runtime::analysis::AnalysisFigureGenerationOptions::default(),
    )?;
    let mut handles = Vec::with_capacity(generated.len());
    let mut diagnostics = Vec::new();
    for figure in generated {
        handles.push(runtime_import_figure(figure.figure).as_u32());
        diagnostics.extend(figure.warnings.into_iter().map(|message| {
            serde_json::json!({
                "code": "FEA_VISUALIZATION_WARNING",
                "severity": "warning",
                "message": message,
                "source": "runmat.fea.visualization",
            })
        }));
    }
    Ok((handles, diagnostics))
}

fn load_fea_results(run_id: String) -> Result<FeaResultsPayload, String> {
    let results = runmat_runtime::analysis::analysis_results_by_run_id_op(
        &run_id,
        runmat_runtime::analysis::AnalysisResultsQuery::metadata_only(),
        runmat_runtime::operations::OperationContext::new(None, None),
    )
    .map_err(|err| err.message)?
    .data;
    let field_descriptors = serialize_field_descriptors(&results)?;
    let result_summary =
        Some(serde_json::to_value(&results.summary).map_err(|err| err.to_string())?);
    Ok(FeaResultsPayload {
        run_id,
        results: serde_json::to_value(results).map_err(|err| err.to_string())?,
        field_descriptors,
        result_summary,
    })
}

fn load_fea_field(
    run_id: String,
    field_id: String,
    options: FeaFieldRequestPayload,
) -> Result<FeaFieldPayload, String> {
    let results = runmat_runtime::analysis::analysis_results_by_run_id_op(
        &run_id,
        runmat_runtime::analysis::AnalysisResultsQuery::field_values(field_id.clone()),
        runmat_runtime::operations::OperationContext::new(None, None),
    )
    .map_err(|err| err.message)?
    .data;
    let descriptor = results
        .field_descriptors
        .iter()
        .find(|descriptor| descriptor.field_id == field_id)
        .ok_or_else(|| format!("FEA field descriptor not found for '{field_id}'"))?;
    let field = results
        .fields
        .iter()
        .find(|field| field.field_id == field_id)
        .ok_or_else(|| format!("FEA field values not found for '{field_id}'"))?;
    let total_count = descriptor.element_count;
    let offset = options.offset.unwrap_or(0).min(total_count);
    let limit = options.limit.unwrap_or(total_count.saturating_sub(offset));
    let values = field.as_host_f64().unwrap_or(&[]);
    let end = offset.saturating_add(limit).min(values.len());
    let sliced_values = if offset < end {
        values[offset..end].to_vec()
    } else {
        Vec::new()
    };
    let count = sliced_values.len();
    Ok(FeaFieldPayload {
        run_id,
        field_id,
        descriptor: serde_json::to_value(descriptor).map_err(|err| err.to_string())?,
        values: sliced_values,
        offset,
        count,
        total_count,
        truncated: offset > 0 || count < total_count,
    })
}

fn serialize_field_descriptors(
    results: &runmat_runtime::analysis::AnalysisResultsData,
) -> Result<Vec<serde_json::Value>, String> {
    results
        .field_descriptors
        .iter()
        .map(|descriptor| serde_json::to_value(descriptor).map_err(|err| err.to_string()))
        .collect()
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
    source: ExecuteRequestSourcePayload,
    compatibility: Option<String>,
    host_policy: Option<ExecuteHostPolicyPayload>,
    requested_outputs: Option<u32>,
    retain_figures: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct GeometryPreviewBudgetPayload {
    max_bytes: Option<u64>,
    max_triangles: Option<u64>,
    max_vertices: Option<u64>,
    timeout_ms: Option<u64>,
    budget_policy: Option<GeometryPreviewBudgetPolicyPayload>,
    tessellation_profile: Option<GeometryPreviewTessellationProfilePayload>,
    xray: Option<bool>,
    allow_create_fea_study: Option<bool>,
}

#[derive(Debug, Clone, Copy, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
enum GeometryPreviewBudgetPolicyPayload {
    #[default]
    Truncate,
    Strict,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct GeometryPreviewTessellationProfilePayload {
    profile_id: Option<String>,
    chord_tolerance: Option<f64>,
    angle_tolerance_deg: Option<f64>,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct GeometryStatsPayload {
    mesh_count: usize,
    total_vertices: u64,
    total_elements: u64,
    region_count: usize,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct GeometryInspectPayload {
    path: String,
    format: String,
    byte_count: u64,
    supported: bool,
    stats: Option<GeometryStatsPayload>,
    geometry_summary: Option<JsonValue>,
    regions: Vec<JsonValue>,
    diagnostics: Vec<JsonValue>,
    degraded_reason: Option<String>,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct GeometryPreviewPayload {
    #[serde(flatten)]
    inspect: GeometryInspectPayload,
    scene_kind: String,
    figure_handle: Option<u32>,
    geometry_scene_handle: Option<u32>,
    truncated: bool,
    preview_message: Option<String>,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct FeaCapabilitiesPayload {
    supported_document_extensions: Vec<&'static str>,
    supported_geometry_extensions: Vec<&'static str>,
    supports_check: bool,
    supports_run: bool,
    supports_results: bool,
    supports_live_progress: bool,
    visualization_backend: &'static str,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct FeaCheckPayload {
    path: String,
    document_kind: String,
    valid: bool,
    validation: JsonValue,
    #[serde(skip_serializing_if = "Option::is_none")]
    plan: Option<JsonValue>,
    diagnostics: Vec<JsonValue>,
    evidence_artifact_paths: Vec<String>,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct FeaRunPayload {
    path: String,
    document_kind: String,
    run: JsonValue,
    #[serde(skip_serializing_if = "Option::is_none")]
    results: Option<JsonValue>,
    field_descriptors: Vec<JsonValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result_summary: Option<JsonValue>,
    figure_handles: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    artifact_manifest: Option<JsonValue>,
    diagnostics: Vec<JsonValue>,
    progress_events: Vec<JsonValue>,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct FeaResultsPayload {
    run_id: String,
    results: JsonValue,
    field_descriptors: Vec<JsonValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result_summary: Option<JsonValue>,
}

#[derive(Debug, serde::Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct FeaFieldRequestPayload {
    offset: Option<usize>,
    limit: Option<usize>,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct FeaFieldPayload {
    run_id: String,
    field_id: String,
    descriptor: JsonValue,
    values: Vec<f64>,
    offset: usize,
    count: usize,
    total_count: usize,
    truncated: bool,
}

#[wasm_bindgen]
impl RunMatWasm {
    #[wasm_bindgen(js_name = executeRequest)]
    pub async fn execute_request_js(&self, request_value: JsValue) -> Result<JsValue, JsValue> {
        let request_payload: ExecuteRequestPayload = serde_wasm_bindgen::from_value(request_value)
            .map_err(|err| js_error(&format!("executeRequest payload parse failed: {err}")))?;
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
        if let Some(compatibility) = request_payload.compatibility.as_deref() {
            if let Some(parsed) = parse_language_compat_from_str(compatibility) {
                request.compatibility = parsed;
            } else {
                return Err(js_error(&format!(
                    "executeRequest compatibility is invalid: {compatibility}"
                )));
            }
        }
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
        let exec_response = session.execute_request(request).await;
        *self.session.borrow_mut() = session;
        let payload = match exec_response.result {
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
                ExecutionPayload::from_outcome(outcome, &exec_response.source_context)
            }
            Err(err) => ExecutionPayload {
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
                stdout: Vec::new(),
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

    #[wasm_bindgen(js_name = inspectGeometry)]
    pub async fn inspect_geometry_js(
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
                        js_error(&format!("inspectGeometry budget parse failed: {err}"))
                    })?,
            )
        };
        let payload = inspect_geometry_path(path, budget)
            .await
            .map_err(|err| js_error(&err))?;
        serde_wasm_bindgen::to_value(&payload)
            .map_err(|err| js_error(&format!("Failed to serialize geometry inspection: {err}")))
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
        let payload = FeaCapabilitiesPayload {
            supported_document_extensions: vec![".fea"],
            supported_geometry_extensions: vec![
                ".stl", ".step", ".stp", ".obj", ".ply", ".gltf", ".glb",
            ],
            supports_check: true,
            supports_run: true,
            supports_results: true,
            supports_live_progress: true,
            visualization_backend: "runmat-plot",
        };
        serde_wasm_bindgen::to_value(&payload)
            .map_err(|err| js_error(&format!("Failed to serialize FEA capabilities: {err}")))
    }

    #[wasm_bindgen(js_name = checkFeaStudy)]
    pub async fn check_fea_study_js(&self, path: String) -> Result<JsValue, JsValue> {
        self.ensure_not_disposed()?;
        let payload = check_fea_path(path).await.map_err(|err| js_error(&err))?;
        serde_wasm_bindgen::to_value(&payload)
            .map_err(|err| js_error(&format!("Failed to serialize FEA check result: {err}")))
    }

    #[wasm_bindgen(js_name = runFeaStudy)]
    pub async fn run_fea_study_js(
        &self,
        path: String,
        artifact_root: Option<String>,
    ) -> Result<JsValue, JsValue> {
        self.ensure_not_disposed()?;
        init_logging_once();
        let interrupt_handle = self.session.borrow().interrupt_handle();
        let _interrupt_guard = install_wasm_interrupt(&self.active_interrupt, interrupt_handle);
        let progress_events = Arc::new(Mutex::new(Vec::new()));
        let progress_events_for_handler = Arc::clone(&progress_events);
        let _progress_guard =
            runmat_runtime::analysis::replace_fea_progress_handler(Some(Arc::new(move |event| {
                if let Ok(mut slot) = progress_events_for_handler.lock() {
                    slot.push(event);
                }
            })));
        if let Some(root) = artifact_root
            .as_deref()
            .filter(|value| !value.trim().is_empty())
        {
            let root = std::path::PathBuf::from(root);
            let _ = runmat_runtime::analysis::configure_fea_runtime(
                runmat_runtime::analysis::FeaRuntimeConfig {
                    artifact_root: Some(root.clone()),
                    study_artifact_root: None,
                    thermo_field_artifact_root: None,
                },
            );
            let _ = runmat_runtime::geometry::configure_prep_artifacts(
                runmat_runtime::geometry::GeometryPrepArtifactConfig {
                    artifact_root: Some(root),
                    ..Default::default()
                },
            );
        }
        let mut payload = run_fea_path(path).await.map_err(|err| js_error(&err))?;
        payload.progress_events = progress_events
            .lock()
            .ok()
            .map(|events| {
                events
                    .iter()
                    .filter_map(|event| serde_json::to_value(event).ok())
                    .collect()
            })
            .unwrap_or_default();
        serde_wasm_bindgen::to_value(&payload)
            .map_err(|err| js_error(&format!("Failed to serialize FEA run result: {err}")))
    }

    #[wasm_bindgen(js_name = feaResults)]
    pub fn fea_results_js(&self, run_id: String) -> Result<JsValue, JsValue> {
        self.ensure_not_disposed()?;
        let payload = load_fea_results(run_id).map_err(|err| js_error(&err))?;
        serde_wasm_bindgen::to_value(&payload)
            .map_err(|err| js_error(&format!("Failed to serialize FEA results: {err}")))
    }

    #[wasm_bindgen(js_name = feaField)]
    pub fn fea_field_js(
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
        let payload = load_fea_field(run_id, field_id, options).map_err(|err| js_error(&err))?;
        serde_wasm_bindgen::to_value(&payload)
            .map_err(|err| js_error(&format!("Failed to serialize FEA field: {err}")))
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
            flag.store(true, Ordering::Relaxed);
            return;
        }
        self.session.borrow().cancel_execution();
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
