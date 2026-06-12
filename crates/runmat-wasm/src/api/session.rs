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
    figure_handles as runtime_figure_handles, import_figure as runtime_import_figure,
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

async fn inspect_geometry_path(
    path: String,
    budget: Option<GeometryPreviewBudgetPayload>,
) -> Result<GeometryInspectPayload, String> {
    let _preview_timeout_ms = budget.as_ref().and_then(|item| item.timeout_ms);
    let metadata = runmat_filesystem::metadata_async(std::path::PathBuf::from(&path))
        .await
        .map_err(|err| format!("failed to inspect geometry metadata {path}: {err}"))?;
    let byte_count = metadata.len();
    if let Some(max_bytes) = budget.as_ref().and_then(|item| item.max_bytes) {
        if byte_count > max_bytes {
            return Ok(GeometryInspectPayload {
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
            });
        }
    }

    let bytes = runmat_filesystem::read_async(std::path::PathBuf::from(&path))
        .await
        .map_err(|err| format!("failed to read geometry file {path}: {err}"))?;
    let inspect = runmat_runtime::geometry::geometry_inspect_op(
        &path,
        &bytes,
        runmat_runtime::operations::OperationContext::new(None, None),
    )
    .map_err(|err| err.message)?;
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
        return Ok(payload);
    }

    let max_triangles = budget
        .as_ref()
        .and_then(|item| item.max_triangles)
        .or(Some(16_000_000));
    let asset = match runmat_runtime::geometry::geometry_load_with_options_op(
        &path,
        &bytes,
        runmat_geometry_io::GeometryImportOptions {
            max_triangles,
            units: runmat_geometry_core::UnitSystem::Meter,
        },
        runmat_runtime::operations::OperationContext::new(None, None),
    ) {
        Ok(envelope) => envelope.data,
        Err(err) => {
            payload.degraded_reason = Some(err.message);
            return Ok(payload);
        }
    };
    let stats = runmat_runtime::geometry::geometry_compute_stats_op(
        &asset,
        runmat_runtime::operations::OperationContext::new(None, None),
    )
    .map_err(|err| err.message)?
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
    Ok(payload)
}

async fn preview_geometry_path(
    path: String,
    budget: Option<GeometryPreviewBudgetPayload>,
) -> Result<GeometryPreviewPayload, String> {
    let inspect = inspect_geometry_path(path, budget.clone()).await?;
    if inspect.degraded_reason.is_some() || inspect.stats.is_none() {
        return Ok(GeometryPreviewPayload {
            inspect,
            scene_kind: "unavailable".to_string(),
            figure_handle: None,
            truncated: false,
            preview_message: Some("Geometry preview is unavailable for this file.".to_string()),
        });
    }
    let asset = load_geometry_asset_for_preview(&inspect.path, budget).await?;
    let figure = runmat_runtime::geometry::geometry_preview_figure(
        &asset,
        format!("Geometry Preview: {}", filename_for_display(&inspect.path)),
        runmat_runtime::geometry::GeometryPreviewFigureOptions::default(),
    )?;
    let figure_handle = runtime_import_figure(figure).as_u32();
    Ok(GeometryPreviewPayload {
        inspect,
        scene_kind: "mesh".to_string(),
        figure_handle: Some(figure_handle),
        truncated: false,
        preview_message: None,
    })
}

async fn load_geometry_asset_for_preview(
    path: &str,
    budget: Option<GeometryPreviewBudgetPayload>,
) -> Result<runmat_geometry_core::GeometryAsset, String> {
    let bytes = runmat_filesystem::read_async(std::path::PathBuf::from(path))
        .await
        .map_err(|err| format!("failed to read geometry file {path}: {err}"))?;
    let max_triangles = budget
        .as_ref()
        .and_then(|item| item.max_triangles)
        .or(Some(16_000_000));
    runmat_runtime::geometry::geometry_load_with_options_op(
        path,
        &bytes,
        runmat_geometry_io::GeometryImportOptions {
            max_triangles,
            units: runmat_geometry_core::UnitSystem::Meter,
        },
        runmat_runtime::operations::OperationContext::new(None, None),
    )
    .map(|envelope| envelope.data)
    .map_err(|err| err.message)
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
    let document = runmat_runtime::analysis::load_fea_document_from_path_async(
        &std::path::PathBuf::from(&path),
    )
    .await?;
    match document {
        runmat_runtime::analysis::FeaResolvedDocument::Study(study) => {
            let run = runmat_runtime::analysis::analysis_run_study_op(
                &study,
                runmat_runtime::operations::OperationContext::new(None, None),
            )
            .map_err(|err| err.message)?
            .data;
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
                figure_handles: Vec::new(),
                artifact_manifest: None,
                diagnostics: Vec::new(),
            })
        }
        runmat_runtime::analysis::FeaResolvedDocument::Sweep(sweep) => {
            let run = runmat_runtime::analysis::analysis_run_study_sweep_op(
                &sweep,
                runmat_runtime::operations::OperationContext::new(None, None),
            )
            .map_err(|err| err.message)?
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
            })
        }
    }
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
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct GeometryPreviewBudgetPayload {
    max_bytes: Option<u64>,
    max_triangles: Option<u64>,
    max_vertices: Option<u64>,
    timeout_ms: Option<u64>,
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
                if !outcome.diagnostics.iter().any(|diagnostic| {
                    diagnostic.severity == runmat_core::abi::DiagnosticSeverity::Error
                }) {
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
        let payload = run_fea_path(path).await.map_err(|err| js_error(&err))?;
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
            flag.store(true, std::sync::atomic::Ordering::Relaxed);
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
