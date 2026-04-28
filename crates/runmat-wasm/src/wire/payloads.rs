use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use uuid::Uuid;
use wasm_bindgen::prelude::JsValue;

use runmat_core::{
    ExecutionProfiling, ExecutionResult, ExecutionStreamEntry, ExecutionStreamKind,
    FusionPlanDecision, FusionPlanEdge, FusionPlanNode, FusionPlanShader, FusionPlanSnapshot,
    MaterializedVariable, StdinEvent, StdinEventKind, WorkspaceEntry, WorkspaceMaterializeOptions,
    WorkspaceMaterializeTarget, WorkspacePreview, WorkspaceSliceOptions, WorkspaceSnapshot,
};
use runmat_runtime::warning_store::RuntimeWarning;

use crate::wire::errors::{js_error, runtime_error_payload, RunMatErrorPayload};
use crate::wire::value::{value_to_json, MAX_DATA_PREVIEW};

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
pub(crate) struct ExecutionPayload {
    pub(crate) value_text: Option<String>,
    pub(crate) value_json: Option<JsonValue>,
    pub(crate) type_info: Option<String>,
    pub(crate) execution_time_ms: u64,
    pub(crate) used_jit: bool,
    pub(crate) error: Option<RunMatErrorPayload>,
    pub(crate) stdout: Vec<ConsoleStreamPayload>,
    pub(crate) workspace: WorkspacePayload,
    pub(crate) figures_touched: Vec<u32>,
    pub(crate) warnings: Vec<WarningPayload>,
    pub(crate) stdin_events: Vec<StdinEventPayload>,
    pub(crate) profiling: Option<ProfilingPayload>,
    pub(crate) fusion_plan: Option<FusionPlanPayload>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct MemoryUsagePayload {
    pub(crate) bytes: u64,
    pub(crate) pages: u32,
}

impl ExecutionPayload {
    pub(crate) fn from_result(result: ExecutionResult, source: &str) -> Self {
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
pub(crate) struct ConsoleStreamPayload {
    pub(crate) stream: &'static str,
    pub(crate) text: String,
    pub(crate) timestamp_ms: u64,
}

impl From<ExecutionStreamEntry> for ConsoleStreamPayload {
    fn from(entry: ExecutionStreamEntry) -> Self {
        let stream = match entry.stream {
            ExecutionStreamKind::Stdout => "stdout",
            ExecutionStreamKind::Stderr => "stderr",
            ExecutionStreamKind::ClearScreen => "clear",
        };
        Self {
            stream,
            text: entry.text,
            timestamp_ms: entry.timestamp_ms,
        }
    }
}

impl ConsoleStreamPayload {
    pub(crate) fn from_console_entry(entry: &runmat_runtime::console::ConsoleEntry) -> Self {
        let stream = match entry.stream {
            runmat_runtime::console::ConsoleStream::Stdout => "stdout",
            runmat_runtime::console::ConsoleStream::Stderr => "stderr",
            runmat_runtime::console::ConsoleStream::ClearScreen => "clear",
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
pub(crate) struct WarningPayload {
    pub(crate) identifier: String,
    pub(crate) message: String,
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
pub(crate) struct StdinEventPayload {
    pub(crate) prompt: String,
    pub(crate) kind: &'static str,
    pub(crate) echo: bool,
    pub(crate) value: Option<String>,
    pub(crate) error: Option<String>,
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
pub(crate) struct WorkspacePayload {
    pub(crate) full: bool,
    pub(crate) version: u64,
    pub(crate) values: Vec<WorkspaceEntryPayload>,
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
pub(crate) struct WorkspaceEntryPayload {
    pub(crate) name: String,
    pub(crate) class_name: String,
    pub(crate) dtype: Option<String>,
    pub(crate) shape: Vec<usize>,
    pub(crate) is_gpu: bool,
    pub(crate) size_bytes: Option<u64>,
    pub(crate) preview: Option<WorkspacePreviewPayload>,
    pub(crate) residency: &'static str,
    pub(crate) preview_token: Option<String>,
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
pub(crate) struct WorkspacePreviewPayload {
    pub(crate) values: Vec<f64>,
    pub(crate) truncated: bool,
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
pub(crate) struct DataMaterializedVariablePayload {
    pub(crate) name: String,
    pub(crate) class_name: String,
    pub(crate) dtype: Option<String>,
    pub(crate) shape: Vec<usize>,
    pub(crate) is_gpu: bool,
    pub(crate) residency: &'static str,
    pub(crate) size_bytes: Option<u64>,
    pub(crate) preview: Option<WorkspacePreviewPayload>,
    pub(crate) value_text: String,
    pub(crate) value_json: JsonValue,
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DataMaterializeOptionsWire {
    pub(crate) limit: Option<usize>,
    pub(crate) slice: Option<DataSliceOptionsWire>,
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DataSliceOptionsWire {
    pub(crate) start: Vec<usize>,
    pub(crate) shape: Vec<usize>,
}

pub(crate) fn parse_data_materialize_options_wire(
    value: JsValue,
) -> Result<DataMaterializeOptionsWire, JsValue> {
    if value.is_undefined() || value.is_null() {
        return Ok(DataMaterializeOptionsWire::default());
    }
    serde_wasm_bindgen::from_value::<DataMaterializeOptionsWire>(value)
        .map_err(|err| js_error(&format!("Invalid data materialize options: {err}")))
}

pub(crate) fn compute_data_preview_slice(
    full_shape: &[usize],
    explicit_slice: Option<&DataSliceOptionsWire>,
    limit: usize,
) -> (Vec<usize>, Vec<usize>) {
    if full_shape.is_empty() {
        return (Vec::new(), Vec::new());
    }

    if let Some(slice) = explicit_slice {
        let mut start = Vec::with_capacity(full_shape.len());
        let mut shape = Vec::with_capacity(full_shape.len());
        for (axis, axis_len) in full_shape.iter().copied().enumerate() {
            let axis_len = axis_len.max(1);
            let axis_start = slice
                .start
                .get(axis)
                .copied()
                .unwrap_or(0)
                .min(axis_len - 1);
            let axis_shape = slice
                .shape
                .get(axis)
                .copied()
                .unwrap_or(axis_len)
                .max(1)
                .min(axis_len - axis_start);
            start.push(axis_start);
            shape.push(axis_shape);
        }
        return (start, shape);
    }

    if full_shape.len() == 1 {
        let span = full_shape[0].max(1).min(limit.max(1));
        return (vec![0], vec![span]);
    }

    let rows = full_shape[0].max(1);
    let cols = full_shape[1].max(1);
    let preview_rows = rows.min((limit as f64).sqrt().floor() as usize).max(1);
    let preview_cols = cols.min((limit / preview_rows).max(1));
    let mut shape = vec![preview_rows, preview_cols];
    shape.extend(full_shape.iter().skip(2).map(|_| 1usize));
    let start = vec![0usize; full_shape.len()];
    (start, shape)
}

pub(crate) fn infer_dataset_class_name(rank: usize) -> &'static str {
    match rank {
        0 => "Scalar",
        1 => "Vector",
        _ => "Tensor",
    }
}

pub(crate) fn estimate_data_array_bytes(shape: &[usize], dtype: &str) -> u64 {
    let elements = shape
        .iter()
        .copied()
        .map(|dim| dim.max(1) as u64)
        .fold(1u64, |acc, dim| acc.saturating_mul(dim));
    elements.saturating_mul(bytes_per_data_element(dtype) as u64)
}

fn bytes_per_data_element(dtype: &str) -> usize {
    let normalized = dtype.to_ascii_lowercase();
    if normalized.contains("64") {
        8
    } else if normalized.contains("32") {
        4
    } else if normalized.contains("16") {
        2
    } else {
        8
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct MaterializedVariablePayload {
    pub(crate) name: String,
    pub(crate) class_name: String,
    pub(crate) dtype: Option<String>,
    pub(crate) shape: Vec<usize>,
    pub(crate) is_gpu: bool,
    pub(crate) residency: &'static str,
    pub(crate) size_bytes: Option<u64>,
    pub(crate) preview: Option<WorkspacePreviewPayload>,
    pub(crate) value_text: String,
    pub(crate) value_json: JsonValue,
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
pub(crate) struct ProfilingPayload {
    pub(crate) total_ms: u64,
    pub(crate) cpu_ms: Option<u64>,
    pub(crate) gpu_ms: Option<u64>,
    pub(crate) kernel_count: Option<u32>,
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
pub(crate) struct FusionPlanPayload {
    pub(crate) nodes: Vec<FusionPlanNodePayload>,
    pub(crate) edges: Vec<FusionPlanEdgePayload>,
    pub(crate) shaders: Vec<FusionPlanShaderPayload>,
    pub(crate) decisions: Vec<FusionPlanDecisionPayload>,
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
pub(crate) struct FusionPlanNodePayload {
    pub(crate) id: String,
    pub(crate) kind: String,
    pub(crate) label: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) residency: Option<String>,
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
pub(crate) struct FusionPlanEdgePayload {
    pub(crate) from: String,
    pub(crate) to: String,
    pub(crate) reason: Option<String>,
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
pub(crate) struct FusionPlanShaderPayload {
    pub(crate) name: String,
    pub(crate) stage: String,
    pub(crate) workgroup_size: Option<[u32; 3]>,
    pub(crate) source_hash: Option<String>,
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
pub(crate) struct FusionPlanDecisionPayload {
    pub(crate) node_id: String,
    pub(crate) fused: bool,
    pub(crate) reason: Option<String>,
    pub(crate) thresholds: Option<String>,
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
pub(crate) struct StatsPayload {
    pub(crate) total_executions: usize,
    pub(crate) jit_compiled: usize,
    pub(crate) interpreter_fallback: usize,
    pub(crate) total_execution_time_ms: u64,
    pub(crate) average_execution_time_ms: f64,
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

pub(crate) fn parse_materialize_target(
    value: JsValue,
) -> Result<WorkspaceMaterializeTarget, JsValue> {
    if value.is_undefined() || value.is_null() {
        return Err(js_error(
            "materializeVariable requires a selector (name or previewToken)",
        ));
    }
    if let Some(token) = value.as_string() {
        return parse_materialize_selector_str(&token);
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

fn parse_materialize_selector_str(selector: &str) -> Result<WorkspaceMaterializeTarget, JsValue> {
    let trimmed = selector.trim();
    if trimmed.is_empty() {
        return Err(js_error(
            "materializeVariable selector string must not be empty",
        ));
    }
    if let Ok(parsed) = Uuid::parse_str(trimmed) {
        return Ok(WorkspaceMaterializeTarget::Token(parsed));
    }
    Ok(WorkspaceMaterializeTarget::Name(trimmed.to_string()))
}

pub(crate) fn parse_materialize_options(
    value: JsValue,
) -> Result<WorkspaceMaterializeOptions, JsValue> {
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
