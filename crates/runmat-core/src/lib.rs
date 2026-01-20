use anyhow::Result;
use runmat_builtins::{self, Tensor, Type, Value};
use runmat_gc::{gc_configure, gc_stats, GcConfig};
use tracing::{debug, info, info_span, warn};

#[cfg(not(target_arch = "wasm32"))]
use runmat_accelerate_api::provider as accel_provider;
use runmat_hir::{LoweringContext, LoweringResult, SemanticError, SourceId};
use runmat_ignition::CompileError;
use runmat_lexer::{tokenize_detailed, Token as LexToken};
pub use runmat_parser::CompatMode;
use runmat_parser::{parse_with_options, ParserOptions, SyntaxError};
use runmat_runtime::builtins::common::gpu_helpers;
use runmat_runtime::warning_store::RuntimeWarning;
use runmat_runtime::{build_runtime_error, RuntimeError};
#[cfg(target_arch = "wasm32")]
use runmat_snapshot::SnapshotBuilder;
use runmat_snapshot::{Snapshot, SnapshotConfig, SnapshotLoader};
use runmat_time::Instant;
#[cfg(feature = "jit")]
use runmat_turbine::TurbineEngine;
use std::collections::{HashMap, HashSet};
use std::future::Future;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::pin::Pin;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use uuid::Uuid;

#[cfg(all(test, target_arch = "wasm32"))]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

mod fusion_snapshot;
mod value_metadata;
use fusion_snapshot::build_fusion_snapshot;

pub use value_metadata::{
    approximate_size_bytes, matlab_class_name, numeric_dtype_label, preview_numeric_values,
    value_shape,
};

/// Host-agnostic RunMat execution session (parser + interpreter + optional JIT).
pub struct RunMatSession {
    /// JIT compiler engine (optional for fallback mode)
    #[cfg(feature = "jit")]
    jit_engine: Option<TurbineEngine>,
    /// Verbose output for debugging
    verbose: bool,
    /// Execution statistics
    stats: ExecutionStats,
    /// Persistent variable context for session state
    variables: HashMap<String, Value>,
    /// Current variable array for bytecode execution
    variable_array: Vec<Value>,
    /// Mapping from variable names to VarId indices
    variable_names: HashMap<String, usize>,
    /// Persistent workspace values keyed by variable name
    workspace_values: HashMap<String, Value>,
    /// User-defined functions context for session state
    function_definitions: HashMap<String, runmat_hir::HirStmt>,
    /// Interned source pool for user-defined functions
    source_pool: SourcePool,
    /// Source IDs for user-defined functions keyed by name
    function_source_ids: HashMap<String, SourceId>,
    /// Loaded snapshot for standard library preloading
    snapshot: Option<Arc<Snapshot>>,
    /// Cooperative cancellation flag shared with the runtime.
    interrupt_flag: Arc<AtomicBool>,
    /// Tracks whether an execution is currently active.
    is_executing: bool,
    /// Optional session-level input handler supplied by the host.
    input_handler: Option<SharedInputHandler>,
    /// Optional async input handler (Phase 2). When set, stdin interactions are awaited
    /// internally by `ExecuteFuture` rather than being surfaced as "pending requests".
    async_input_handler: Option<SharedAsyncInputHandler>,
    /// Maximum number of call stack frames to retain for diagnostics.
    callstack_limit: usize,
    telemetry_consent: bool,
    telemetry_client_id: Option<String>,
    workspace_preview_tokens: HashMap<Uuid, WorkspaceMaterializeTicket>,
    workspace_version: u64,
    emit_fusion_plan: bool,
    compat_mode: CompatMode,
}

#[derive(Debug, Clone)]
struct SourceText {
    name: Arc<str>,
    text: Arc<str>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SourceKey {
    name: Arc<str>,
    text: Arc<str>,
}

#[derive(Default)]
struct SourcePool {
    sources: Vec<SourceText>,
    index: HashMap<SourceKey, SourceId>,
}

impl SourcePool {
    fn intern(&mut self, name: &str, text: &str) -> SourceId {
        let name: Arc<str> = Arc::from(name);
        let text: Arc<str> = Arc::from(text);
        let key = SourceKey {
            name: Arc::clone(&name),
            text: Arc::clone(&text),
        };
        if let Some(id) = self.index.get(&key) {
            return *id;
        }
        let id = SourceId(self.sources.len());
        self.sources.push(SourceText { name, text });
        self.index.insert(key, id);
        id
    }

    fn get(&self, id: SourceId) -> Option<&SourceText> {
        self.sources.get(id.0)
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

#[derive(Debug)]
pub enum RunError {
    Syntax(SyntaxError),
    Semantic(SemanticError),
    Compile(CompileError),
    Runtime(RuntimeError),
}

impl std::fmt::Display for RunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunError::Syntax(err) => write!(f, "{err}"),
            RunError::Semantic(err) => write!(f, "{err}"),
            RunError::Compile(err) => write!(f, "{err}"),
            RunError::Runtime(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for RunError {}

impl From<SyntaxError> for RunError {
    fn from(value: SyntaxError) -> Self {
        RunError::Syntax(value)
    }
}

impl From<SemanticError> for RunError {
    fn from(value: SemanticError) -> Self {
        RunError::Semantic(value)
    }
}

impl From<CompileError> for RunError {
    fn from(value: CompileError) -> Self {
        RunError::Compile(value)
    }
}

impl From<RuntimeError> for RunError {
    fn from(value: RuntimeError) -> Self {
        RunError::Runtime(value)
    }
}

struct PreparedExecution {
    ast: runmat_parser::Program,
    lowering: LoweringResult,
    bytecode: runmat_ignition::Bytecode,
}

#[derive(Debug, Default)]
pub struct ExecutionStats {
    pub total_executions: usize,
    pub jit_compiled: usize,
    pub interpreter_fallback: usize,
    pub total_execution_time_ms: u64,
    pub average_execution_time_ms: f64,
}

#[derive(Debug, Clone)]
pub enum StdinEventKind {
    Line,
    KeyPress,
}

#[derive(Debug, Clone)]
pub struct StdinEvent {
    pub prompt: String,
    pub kind: StdinEventKind,
    pub echo: bool,
    pub value: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub enum InputRequestKind {
    Line { echo: bool },
    KeyPress,
}

#[derive(Debug, Clone)]
pub struct InputRequest {
    pub prompt: String,
    pub kind: InputRequestKind,
}

#[derive(Debug, Clone)]
pub enum InputResponse {
    Line(String),
    KeyPress,
}

#[derive(Debug, Clone)]
pub enum InputHandlerAction {
    Respond(Result<InputResponse, String>),
}

type SharedInputHandler = Arc<dyn Fn(&InputRequest) -> InputHandlerAction + Send + Sync>;

type SharedAsyncInputHandler = Arc<
    dyn Fn(InputRequest) -> Pin<Box<dyn Future<Output = Result<InputResponse, String>> + 'static>>
        + Send
        + Sync,
>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionStreamKind {
    Stdout,
    Stderr,
}

#[derive(Debug, Clone)]
pub struct ExecutionStreamEntry {
    pub stream: ExecutionStreamKind,
    pub text: String,
    pub timestamp_ms: u64,
}

#[derive(Debug, Clone)]
pub struct WorkspacePreview {
    pub values: Vec<f64>,
    pub truncated: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkspaceResidency {
    Cpu,
    Gpu,
    Unknown,
}

impl WorkspaceResidency {
    pub fn as_str(&self) -> &'static str {
        match self {
            WorkspaceResidency::Cpu => "cpu",
            WorkspaceResidency::Gpu => "gpu",
            WorkspaceResidency::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone)]
pub struct WorkspaceEntry {
    pub name: String,
    pub class_name: String,
    pub dtype: Option<String>,
    pub shape: Vec<usize>,
    pub is_gpu: bool,
    pub size_bytes: Option<u64>,
    pub preview: Option<WorkspacePreview>,
    pub residency: WorkspaceResidency,
    pub preview_token: Option<Uuid>,
}

#[derive(Debug, Clone)]
pub struct WorkspaceSnapshot {
    pub full: bool,
    pub version: u64,
    pub values: Vec<WorkspaceEntry>,
}

#[derive(Debug, Clone)]
pub struct MaterializedVariable {
    pub name: String,
    pub class_name: String,
    pub dtype: Option<String>,
    pub shape: Vec<usize>,
    pub is_gpu: bool,
    pub residency: WorkspaceResidency,
    pub size_bytes: Option<u64>,
    pub preview: Option<WorkspacePreview>,
    pub value: Value,
}

#[derive(Debug, Clone)]
pub enum WorkspaceMaterializeTarget {
    Name(String),
    Token(Uuid),
}

#[derive(Debug, Clone)]
pub struct WorkspaceSliceOptions {
    pub start: Vec<usize>,
    pub shape: Vec<usize>,
}

impl WorkspaceSliceOptions {
    fn sanitized(&self, tensor_shape: &[usize]) -> Option<WorkspaceSliceOptions> {
        if tensor_shape.is_empty() {
            return None;
        }
        let mut start = Vec::with_capacity(tensor_shape.len());
        let mut shape = Vec::with_capacity(tensor_shape.len());
        for axis_idx in 0..tensor_shape.len() {
            let axis_len = tensor_shape[axis_idx];
            if axis_len == 0 {
                return None;
            }
            let requested_start = self.start.get(axis_idx).copied().unwrap_or(0);
            let clamped_start = requested_start.min(axis_len.saturating_sub(1));
            let requested_count = self.shape.get(axis_idx).copied().unwrap_or(axis_len);
            let clamped_count = requested_count.max(1).min(axis_len - clamped_start);
            start.push(clamped_start);
            shape.push(clamped_count);
        }
        Some(WorkspaceSliceOptions { start, shape })
    }
}

#[derive(Debug, Clone)]
pub struct WorkspaceMaterializeOptions {
    pub max_elements: usize,
    pub slice: Option<WorkspaceSliceOptions>,
}

impl Default for WorkspaceMaterializeOptions {
    fn default() -> Self {
        Self {
            max_elements: MATERIALIZE_DEFAULT_LIMIT,
            slice: None,
        }
    }
}

fn slice_value_for_preview(value: &Value, slice: &WorkspaceSliceOptions) -> Option<Value> {
    match value {
        Value::Tensor(tensor) => {
            let data = gather_tensor_slice(tensor, slice);
            if data.is_empty() {
                return None;
            }
            let mut shape = slice.shape.clone();
            if shape.is_empty() {
                shape.push(1);
            }
            let rows = shape.get(0).copied().unwrap_or(1);
            let cols = shape.get(1).copied().unwrap_or(1);
            Some(Value::Tensor(Tensor {
                data,
                shape,
                rows,
                cols,
                dtype: tensor.dtype,
            }))
        }
        _ => None,
    }
}

fn gather_tensor_slice(tensor: &Tensor, slice: &WorkspaceSliceOptions) -> Vec<f64> {
    if tensor.shape.is_empty() || slice.shape.iter().any(|count| *count == 0) {
        return Vec::new();
    }
    let total: usize = slice.shape.iter().product();
    let mut result = Vec::with_capacity(total);
    let mut coords = vec![0usize; tensor.shape.len()];
    gather_tensor_slice_recursive(tensor, slice, 0, &mut coords, &mut result);
    result
}

fn gather_tensor_slice_recursive(
    tensor: &Tensor,
    slice: &WorkspaceSliceOptions,
    axis: usize,
    coords: &mut [usize],
    out: &mut Vec<f64>,
) {
    if axis == tensor.shape.len() {
        let idx = column_major_index(&tensor.shape, coords);
        if let Some(value) = tensor.data.get(idx) {
            out.push(*value);
        }
        return;
    }
    let start = slice.start.get(axis).copied().unwrap_or(0);
    let count = slice.shape.get(axis).copied().unwrap_or(1);
    for offset in 0..count {
        coords[axis] = start + offset;
        gather_tensor_slice_recursive(tensor, slice, axis + 1, coords, out);
    }
}

fn column_major_index(shape: &[usize], coords: &[usize]) -> usize {
    let mut idx = 0usize;
    let mut stride = 1usize;
    for (dim_len, coord) in shape.iter().zip(coords.iter()) {
        idx += coord * stride;
        stride *= *dim_len;
    }
    idx
}

#[derive(Debug, Clone, Default)]
pub struct ExecutionProfiling {
    pub total_ms: u64,
    pub cpu_ms: Option<u64>,
    pub gpu_ms: Option<u64>,
    pub kernel_count: Option<u32>,
}

#[derive(Debug, Clone, Default)]
pub struct FusionPlanSnapshot {
    pub nodes: Vec<FusionPlanNode>,
    pub edges: Vec<FusionPlanEdge>,
    pub shaders: Vec<FusionPlanShader>,
    pub decisions: Vec<FusionPlanDecision>,
}

#[derive(Debug, Clone)]
pub struct FusionPlanNode {
    pub id: String,
    pub kind: String,
    pub label: String,
    pub shape: Vec<usize>,
    pub residency: Option<String>,
}

#[derive(Debug, Clone)]
pub struct FusionPlanEdge {
    pub from: String,
    pub to: String,
    pub reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct FusionPlanShader {
    pub name: String,
    pub stage: String,
    pub workgroup_size: Option<[u32; 3]>,
    pub source_hash: Option<String>,
}

#[derive(Debug, Clone)]
pub struct FusionPlanDecision {
    pub node_id: String,
    pub fused: bool,
    pub reason: Option<String>,
    pub thresholds: Option<String>,
}

#[derive(Debug)]
pub struct ExecutionResult {
    pub value: Option<Value>,
    pub execution_time_ms: u64,
    pub used_jit: bool,
    pub error: Option<RuntimeError>,
    /// Type information displayed when output is suppressed by semicolon
    pub type_info: Option<String>,
    /// Ordered console output (stdout/stderr) captured during execution.
    pub streams: Vec<ExecutionStreamEntry>,
    /// Workspace metadata for variables touched during this execution.
    pub workspace: WorkspaceSnapshot,
    /// Figure handles that were mutated during this execution.
    pub figures_touched: Vec<u32>,
    /// Structured MATLAB warnings raised during this execution.
    pub warnings: Vec<RuntimeWarning>,
    /// Optional profiling summary (wall/cpu/gpu).
    pub profiling: Option<ExecutionProfiling>,
    /// Optional fusion plan metadata emitted by Accelerate.
    pub fusion_plan: Option<FusionPlanSnapshot>,
    /// Recorded stdin interactions (prompts, values) during execution.
    pub stdin_events: Vec<StdinEvent>,
}

#[derive(Debug, Clone)]
struct WorkspaceMaterializeTicket {
    name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FinalStmtEmitDisposition {
    Inline,
    #[allow(dead_code)]
    NeedsFallback,
    Suppressed,
}

fn determine_display_label_from_context(
    single_assign_var: Option<usize>,
    id_to_name: &HashMap<usize, String>,
    is_expression_stmt: bool,
    single_stmt_non_assign: bool,
) -> Option<String> {
    if let Some(var_id) = single_assign_var {
        id_to_name.get(&var_id).cloned()
    } else if is_expression_stmt || single_stmt_non_assign {
        Some("ans".to_string())
    } else {
        None
    }
}

/// Format value type information like MATLAB (e.g., "1000x1 vector", "3x3 matrix")
fn format_type_info(value: &Value) -> String {
    match value {
        Value::Int(_) => "scalar".to_string(),
        Value::Num(_) => "scalar".to_string(),
        Value::Bool(_) => "logical scalar".to_string(),
        Value::String(_) => "string".to_string(),
        Value::StringArray(sa) => {
            // MATLAB displays string arrays as m x n string array; for test's purpose, we classify scalar string arrays as "string"
            if sa.shape == vec![1, 1] {
                "string".to_string()
            } else {
                format!("{}x{} string array", sa.rows(), sa.cols())
            }
        }
        Value::CharArray(ca) => {
            if ca.rows == 1 && ca.cols == 1 {
                "char".to_string()
            } else {
                format!("{}x{} char array", ca.rows, ca.cols)
            }
        }
        Value::Tensor(m) => {
            if m.rows() == 1 && m.cols() == 1 {
                "scalar".to_string()
            } else if m.rows() == 1 || m.cols() == 1 {
                format!("{}x{} vector", m.rows(), m.cols())
            } else {
                format!("{}x{} matrix", m.rows(), m.cols())
            }
        }
        Value::Cell(cells) => {
            if cells.data.len() == 1 {
                "1x1 cell".to_string()
            } else {
                format!("{}x1 cell array", cells.data.len())
            }
        }
        Value::GpuTensor(h) => {
            if h.shape.len() == 2 {
                let r = h.shape[0];
                let c = h.shape[1];
                if r == 1 && c == 1 {
                    "scalar (gpu)".to_string()
                } else if r == 1 || c == 1 {
                    format!("{r}x{c} vector (gpu)")
                } else {
                    format!("{r}x{c} matrix (gpu)")
                }
            } else {
                format!("Tensor{:?} (gpu)", h.shape)
            }
        }
        _ => "value".to_string(),
    }
}

impl RunMatSession {
    /// Create a new session
    pub fn new() -> Result<Self> {
        Self::with_options(true, false) // JIT enabled, verbose disabled
    }

    /// Create a new session with specific options
    pub fn with_options(enable_jit: bool, verbose: bool) -> Result<Self> {
        Self::from_snapshot(enable_jit, verbose, None)
    }

    /// Create a new session with snapshot loading
    #[cfg(not(target_arch = "wasm32"))]
    pub fn with_snapshot<P: AsRef<Path>>(
        enable_jit: bool,
        verbose: bool,
        snapshot_path: Option<P>,
    ) -> Result<Self> {
        let snapshot = snapshot_path.and_then(|path| match Self::load_snapshot(path.as_ref()) {
            Ok(snapshot) => {
                info!(
                    "Snapshot loaded successfully from {}",
                    path.as_ref().display()
                );
                Some(Arc::new(snapshot))
            }
            Err(e) => {
                warn!(
                    "Failed to load snapshot from {}: {}, continuing without snapshot",
                    path.as_ref().display(),
                    e
                );
                None
            }
        });
        Self::from_snapshot(enable_jit, verbose, snapshot)
    }

    /// Create a session using snapshot bytes (already fetched from disk or network)
    pub fn with_snapshot_bytes(
        enable_jit: bool,
        verbose: bool,
        snapshot_bytes: Option<&[u8]>,
    ) -> Result<Self> {
        let snapshot =
            snapshot_bytes.and_then(|bytes| match Self::load_snapshot_from_bytes(bytes) {
                Ok(snapshot) => {
                    info!("Snapshot loaded successfully from in-memory bytes");
                    Some(Arc::new(snapshot))
                }
                Err(e) => {
                    warn!("Failed to load snapshot from bytes: {e}, continuing without snapshot");
                    None
                }
            });
        Self::from_snapshot(enable_jit, verbose, snapshot)
    }

    fn from_snapshot(
        enable_jit: bool,
        verbose: bool,
        snapshot: Option<Arc<Snapshot>>,
    ) -> Result<Self> {
        #[cfg(target_arch = "wasm32")]
        let snapshot = {
            match snapshot {
                some @ Some(_) => some,
                None => Self::build_wasm_snapshot(),
            }
        };

        #[cfg(feature = "jit")]
        let jit_engine = if enable_jit {
            match TurbineEngine::new() {
                Ok(engine) => {
                    info!("JIT compiler initialized successfully");
                    Some(engine)
                }
                Err(e) => {
                    warn!("JIT compiler initialization failed: {e}, falling back to interpreter");
                    None
                }
            }
        } else {
            info!("JIT compiler disabled, using interpreter only");
            None
        };

        #[cfg(not(feature = "jit"))]
        if enable_jit {
            info!("JIT support was requested but the 'jit' feature is disabled; running interpreter-only.");
        }

        let session = Self {
            #[cfg(feature = "jit")]
            jit_engine,
            verbose,
            stats: ExecutionStats::default(),
            variables: HashMap::new(),
            variable_array: Vec::new(),
            variable_names: HashMap::new(),
            workspace_values: HashMap::new(),
            function_definitions: HashMap::new(),
            source_pool: SourcePool::default(),
            function_source_ids: HashMap::new(),
            snapshot,
            interrupt_flag: Arc::new(AtomicBool::new(false)),
            is_executing: false,
            input_handler: None,
            async_input_handler: None,
            callstack_limit: runmat_ignition::DEFAULT_CALLSTACK_LIMIT,
            telemetry_consent: true,
            telemetry_client_id: None,
            workspace_preview_tokens: HashMap::new(),
            workspace_version: 0,
            emit_fusion_plan: false,
            compat_mode: CompatMode::Matlab,
        };

        runmat_ignition::set_call_stack_limit(session.callstack_limit);

        // Cache the shared plotting context (if a GPU provider is active) so the
        // runtime can wire zero-copy render paths without instantiating another
        // WebGPU device.
        #[cfg(any(target_arch = "wasm32", not(target_arch = "wasm32")))]
        {
            if let Err(err) =
                runmat_runtime::builtins::plotting::context::ensure_context_from_provider()
            {
                debug!("Plotting context unavailable during session init: {err}");
            }
        }

        Ok(session)
    }

    #[cfg(target_arch = "wasm32")]
    fn build_wasm_snapshot() -> Option<Arc<Snapshot>> {
        use log::{info, warn};

        info!("No snapshot provided; building stdlib snapshot inside wasm runtime");
        let mut config = SnapshotConfig::default();
        config.compression_enabled = false;
        config.validation_enabled = false;
        config.memory_mapping_enabled = false;
        config.parallel_loading = false;
        config.progress_reporting = false;

        match SnapshotBuilder::new(config).build() {
            Ok(snapshot) => {
                info!("WASM snapshot build completed successfully");
                Some(Arc::new(snapshot))
            }
            Err(err) => {
                warn!("Failed to build stdlib snapshot in wasm runtime: {err}");
                None
            }
        }
    }

    /// Load a snapshot from disk
    #[cfg(not(target_arch = "wasm32"))]
    fn load_snapshot(path: &Path) -> Result<Snapshot> {
        let mut loader = SnapshotLoader::new(SnapshotConfig::default());
        let (snapshot, _stats) = loader
            .load(path)
            .map_err(|e| anyhow::anyhow!("Failed to load snapshot: {}", e))?;
        Ok(snapshot)
    }

    /// Load a snapshot from in-memory bytes
    fn load_snapshot_from_bytes(bytes: &[u8]) -> Result<Snapshot> {
        let mut loader = SnapshotLoader::new(SnapshotConfig::default());
        let (snapshot, _stats) = loader
            .load_from_bytes(bytes)
            .map_err(|e| anyhow::anyhow!("Failed to load snapshot: {}", e))?;
        Ok(snapshot)
    }

    /// Install a session-scoped handler for stdin-style interaction prompts.
    pub fn install_input_handler<F>(&mut self, handler: F)
    where
        F: Fn(&InputRequest) -> InputHandlerAction + Send + Sync + 'static,
    {
        self.input_handler = Some(Arc::new(handler));
    }

    /// Remove any previously installed stdin handler.
    pub fn clear_input_handler(&mut self) {
        self.input_handler = None;
    }

    /// Install an async stdin handler (Phase 2). This is the preferred input path for
    /// poll-driven execution (`ExecuteFuture`).
    ///
    /// The handler is invoked when `input()` / `pause()` needs a line or keypress, and the
    /// returned future is awaited by the runtime.
    pub fn install_async_input_handler<F, Fut>(&mut self, handler: F)
    where
        F: Fn(InputRequest) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<InputResponse, String>> + 'static,
    {
        self.async_input_handler = Some(Arc::new(move |req: InputRequest| {
            let fut = handler(req);
            Box::pin(fut)
        }));
    }

    pub fn clear_async_input_handler(&mut self) {
        self.async_input_handler = None;
    }

    pub fn telemetry_consent(&self) -> bool {
        self.telemetry_consent
    }

    pub fn set_telemetry_consent(&mut self, consent: bool) {
        self.telemetry_consent = consent;
    }

    pub fn telemetry_client_id(&self) -> Option<&str> {
        self.telemetry_client_id.as_deref()
    }

    pub fn set_telemetry_client_id(&mut self, cid: Option<String>) {
        self.telemetry_client_id = cid;
    }

    fn build_runtime_input_handler(
        &self,
        stdin_events: Arc<Mutex<Vec<StdinEvent>>>,
    ) -> Arc<
        dyn for<'a> Fn(
                runmat_runtime::interaction::InteractionPrompt<'a>,
            ) -> runmat_runtime::interaction::InteractionDecision
            + Send
            + Sync,
    > {
        let session_handler = self.input_handler.clone();
        Arc::new(move |prompt| {
            if matches!(
                prompt.kind,
                runmat_runtime::interaction::InteractionKind::GpuMapRead
            ) {
                return runmat_runtime::interaction::InteractionDecision::Respond(Err(
                    "gpu map read pending is not supported by input handlers".to_string(),
                ));
            }
            let request_kind = match prompt.kind {
                runmat_runtime::interaction::InteractionKind::Line { echo } => {
                    InputRequestKind::Line { echo }
                }
                runmat_runtime::interaction::InteractionKind::KeyPress => {
                    InputRequestKind::KeyPress
                }
                runmat_runtime::interaction::InteractionKind::GpuMapRead => {
                    InputRequestKind::KeyPress
                }
            };
            let request = InputRequest {
                prompt: prompt.prompt.to_string(),
                kind: request_kind,
            };
            let (event_kind, echo_flag) = match &request.kind {
                InputRequestKind::Line { echo } => (StdinEventKind::Line, *echo),
                InputRequestKind::KeyPress => (StdinEventKind::KeyPress, false),
            };
            let mut event = StdinEvent {
                prompt: request.prompt.clone(),
                kind: event_kind,
                echo: echo_flag,
                value: None,
                error: None,
            };
            let action = if let Some(handler) = &session_handler {
                handler(&request)
            } else {
                match &request.kind {
                    InputRequestKind::Line { echo } => InputHandlerAction::Respond(
                        runmat_runtime::interaction::default_read_line(&request.prompt, *echo)
                            .map(InputResponse::Line),
                    ),
                    InputRequestKind::KeyPress => InputHandlerAction::Respond(
                        runmat_runtime::interaction::default_wait_for_key(&request.prompt)
                            .map(|_| InputResponse::KeyPress),
                    ),
                }
            };
            match action {
                InputHandlerAction::Respond(result) => {
                    let mapped = result
                        .inspect_err(|err| {
                            event.error = Some(err.clone());
                            if let Ok(mut guard) = stdin_events.lock() {
                                guard.push(event.clone());
                            }
                        })
                        .map(|resp| match resp {
                            InputResponse::Line(value) => {
                                event.value = Some(value.clone());
                                if let Ok(mut guard) = stdin_events.lock() {
                                    guard.push(event);
                                }
                                runmat_runtime::interaction::InteractionResponse::Line(value)
                            }
                            InputResponse::KeyPress => {
                                if let Ok(mut guard) = stdin_events.lock() {
                                    guard.push(event);
                                }
                                runmat_runtime::interaction::InteractionResponse::KeyPress
                            }
                        });
                    runmat_runtime::interaction::InteractionDecision::Respond(mapped)
                }
            }
        })
    }

    /// Request cooperative cancellation for the currently running execution.
    pub fn cancel_execution(&self) {
        self.interrupt_flag.store(true, Ordering::Relaxed);
    }

    /// Get snapshot information
    pub fn snapshot_info(&self) -> Option<String> {
        self.snapshot.as_ref().map(|snapshot| {
            format!(
                "Snapshot loaded: {} builtins, {} HIR functions, {} bytecode entries",
                snapshot.builtins.functions.len(),
                snapshot.hir_cache.functions.len(),
                snapshot.bytecode_cache.stdlib_bytecode.len()
            )
        })
    }

    /// Check if a snapshot is loaded
    pub fn has_snapshot(&self) -> bool {
        self.snapshot.is_some()
    }

    fn compile_input(&mut self, input: &str) -> std::result::Result<PreparedExecution, RunError> {
        let source_id = self.source_pool.intern("<repl>", input);
        let ast = {
            let _span = info_span!("runtime.parse").entered();
            parse_with_options(input, ParserOptions::new(self.compat_mode))?
        };
        let lowering = {
            let _span = info_span!("runtime.lower").entered();
            runmat_hir::lower(
                &ast,
                &LoweringContext::new(&self.variable_names, &self.function_definitions),
            )?
        };
        let existing_functions = self.convert_hir_functions_to_user_functions();
        let mut bytecode = {
            let _span = info_span!("runtime.compile.bytecode").entered();
            runmat_ignition::compile(&lowering.hir, &existing_functions)?
        };
        bytecode.source_id = Some(source_id);
        let new_function_names: HashSet<String> = lowering.functions.keys().cloned().collect();
        for (name, func) in bytecode.functions.iter_mut() {
            if new_function_names.contains(name) {
                func.source_id = Some(source_id);
            }
        }
        Ok(PreparedExecution {
            ast,
            lowering,
            bytecode,
        })
    }

    fn populate_callstack(&self, error: &mut RuntimeError) {
        if !error.context.call_stack.is_empty() || error.context.call_frames.is_empty() {
            return;
        }
        let mut rendered = Vec::new();
        if error.context.call_frames_elided > 0 {
            rendered.push(format!(
                "... {} frames elided ...",
                error.context.call_frames_elided
            ));
        }
        for frame in error.context.call_frames.iter().rev() {
            let mut line = frame.function.clone();
            if let (Some(source_id), Some((start, _end))) = (frame.source_id, frame.span) {
                if let Some(source) = self.source_pool.get(SourceId(source_id)) {
                    let (line_num, col) = line_col_from_offset(&source.text, start);
                    line = format!("{} @ {}:{}:{}", frame.function, source.name, line_num, col);
                }
            }
            rendered.push(line);
        }
        error.context.call_stack = rendered;
    }

    /// Compile the input and produce a fusion plan snapshot without executing.
    pub fn compile_fusion_plan(
        &mut self,
        input: &str,
    ) -> std::result::Result<Option<FusionPlanSnapshot>, RunError> {
        let prepared = self.compile_input(input)?;
        Ok(build_fusion_snapshot(
            prepared.bytecode.accel_graph.as_ref(),
            &prepared.bytecode.fusion_groups,
        ))
    }

    /// Execute MATLAB/Octave code
    pub async fn execute(&mut self, input: &str) -> std::result::Result<ExecutionResult, RunError> {
        self.run(input).await
    }

    /// Parse, lower, compile, and execute input.
    pub async fn run(&mut self, input: &str) -> std::result::Result<ExecutionResult, RunError> {
        let _active = ActiveExecutionGuard::new(self)
            .map_err(|err| RunError::Runtime(build_runtime_error(err.to_string()).build()))?;
        runmat_ignition::set_call_stack_limit(self.callstack_limit);
        let exec_span = info_span!(
            "runtime.execute",
            input_len = input.len(),
            verbose = self.verbose
        );
        let _exec_guard = exec_span.enter();
        runmat_runtime::console::reset_thread_buffer();
        runmat_runtime::plotting_hooks::reset_recent_figures();
        runmat_runtime::warning_store::reset();
        reset_provider_telemetry();
        self.interrupt_flag.store(false, Ordering::Relaxed);
        let _interrupt_guard =
            runmat_runtime::interrupt::replace_interrupt(Some(self.interrupt_flag.clone()));
        let start_time = Instant::now();
        self.stats.total_executions += 1;
        let debug_trace = std::env::var("RUNMAT_DEBUG_REPL").is_ok();
        let stdin_events: Arc<Mutex<Vec<StdinEvent>>> = Arc::new(Mutex::new(Vec::new()));
        let runtime_handler = self.build_runtime_input_handler(Arc::clone(&stdin_events));
        let _input_guard = runmat_runtime::interaction::replace_handler(Some(runtime_handler));

        if self.verbose {
            debug!("Executing: {}", input.trim());
        }

        let PreparedExecution {
            ast,
            lowering,
            mut bytecode,
        } = self.compile_input(input)?;
        if self.verbose {
            debug!("AST: {ast:?}");
        }
        let (hir, updated_vars, updated_functions, var_names_map) = (
            lowering.hir,
            lowering.variables,
            lowering.functions,
            lowering.var_names,
        );
        let max_var_id = updated_vars.values().copied().max().unwrap_or(0);
        if debug_trace {
            debug!(?updated_vars, "[repl] updated_vars");
        }
        if debug_trace {
            debug!(workspace_values_before = ?self.workspace_values, "[repl] workspace snapshot before execution");
        }
        let id_to_name: HashMap<usize, String> = var_names_map
            .iter()
            .map(|(var_id, name)| (var_id.0, name.clone()))
            .collect();
        let mut assigned_this_execution: HashSet<String> = HashSet::new();
        let assigned_snapshot: HashSet<String> = updated_vars
            .keys()
            .filter(|name| self.workspace_values.contains_key(name.as_str()))
            .cloned()
            .collect();
        let prev_assigned_snapshot = assigned_snapshot.clone();
        if debug_trace {
            debug!(?assigned_snapshot, "[repl] assigned snapshot");
        }
        let _pending_workspace_guard = runmat_ignition::push_pending_workspace(
            updated_vars.clone(),
            assigned_snapshot.clone(),
        );
        if self.verbose {
            debug!("HIR generated successfully");
        }

        let (single_assign_var, single_stmt_non_assign) = if hir.body.len() == 1 {
            match &hir.body[0] {
                runmat_hir::HirStmt::Assign(var_id, _, _, _) => (Some(var_id.0), false),
                _ => (None, true),
            }
        } else {
            (None, false)
        };

        bytecode.var_names = id_to_name.clone();
        if self.verbose {
            debug!(
                "Bytecode compiled: {} instructions",
                bytecode.instructions.len()
            );
        }

        #[cfg(not(target_arch = "wasm32"))]
        let fusion_snapshot = if self.emit_fusion_plan {
            build_fusion_snapshot(bytecode.accel_graph.as_ref(), &bytecode.fusion_groups)
        } else {
            None
        };
        #[cfg(target_arch = "wasm32")]
        let fusion_snapshot: Option<FusionPlanSnapshot> = None;

        // Prepare variable array with existing values before execution
        self.prepare_variable_array_for_execution(&bytecode, &updated_vars, debug_trace);

        if self.verbose {
            debug!(
                "Variable array after preparation: {:?}",
                self.variable_array
            );
            debug!("Updated variable mapping: {updated_vars:?}");
            debug!("Bytecode instructions: {:?}", bytecode.instructions);
        }

        #[cfg(feature = "jit")]
        let mut used_jit = false;
        #[cfg(not(feature = "jit"))]
        let used_jit = false;
        #[cfg(feature = "jit")]
        let mut execution_completed = false;
        #[cfg(not(feature = "jit"))]
        let execution_completed = false;
        let mut result_value: Option<Value> = None; // Always start fresh for each execution
        let mut suppressed_value: Option<Value> = None; // Track value for type info when suppressed
        let mut error = None;
        let mut workspace_updates: Vec<WorkspaceEntry> = Vec::new();
        let mut ans_update: Option<(usize, Value)> = None;

        // Check if this is an expression statement (ends with Pop)
        let is_expression_stmt = bytecode
            .instructions
            .last()
            .map(|instr| matches!(instr, runmat_ignition::Instr::Pop))
            .unwrap_or(false);

        // Determine whether the final statement ended with a semicolon by inspecting the raw input.
        let is_semicolon_suppressed = {
            let toks = tokenize_detailed(input);
            toks.into_iter()
                .rev()
                .map(|t| t.token)
                .find(|token| {
                    !matches!(
                        token,
                        LexToken::Newline
                            | LexToken::LineComment
                            | LexToken::BlockComment
                            | LexToken::Section
                    )
                })
                .map(|t| matches!(t, LexToken::Semicolon))
                .unwrap_or(false)
        };
        let final_stmt_emit = last_displayable_statement_emit_disposition(&hir.body);

        if self.verbose {
            debug!("HIR body len: {}", hir.body.len());
            if !hir.body.is_empty() {
                debug!("HIR statement: {:?}", &hir.body[0]);
            }
            debug!("is_semicolon_suppressed: {is_semicolon_suppressed}");
        }

        // Use JIT for assignments, interpreter for expressions (to capture results properly)
        #[cfg(feature = "jit")]
        {
            if let Some(ref mut jit_engine) = &mut self.jit_engine {
                if !is_expression_stmt {
                    // Ensure variable array is large enough
                    if self.variable_array.len() < bytecode.var_count {
                        self.variable_array
                            .resize(bytecode.var_count, Value::Num(0.0));
                    }

                    if self.verbose {
                        debug!(
                            "JIT path for assignment: variable_array size: {}, bytecode.var_count: {}",
                            self.variable_array.len(),
                            bytecode.var_count
                        );
                    }

                    // Use JIT for assignments
                    match jit_engine.execute_or_compile(&bytecode, &mut self.variable_array) {
                        Ok((_, actual_used_jit)) => {
                            used_jit = actual_used_jit;
                            execution_completed = true;
                            if actual_used_jit {
                                self.stats.jit_compiled += 1;
                            } else {
                                self.stats.interpreter_fallback += 1;
                            }
                            if let Some(runmat_hir::HirStmt::Assign(var_id, _, _, _)) =
                                hir.body.first()
                            {
                                if let Some(name) = id_to_name.get(&var_id.0) {
                                    assigned_this_execution.insert(name.clone());
                                }
                                if var_id.0 < self.variable_array.len() {
                                    let assignment_value = self.variable_array[var_id.0].clone();
                                    if !is_semicolon_suppressed {
                                        result_value = Some(assignment_value);
                                        if self.verbose {
                                            debug!("JIT assignment result: {result_value:?}");
                                        }
                                    } else {
                                        suppressed_value = Some(assignment_value);
                                        if self.verbose {
                                            debug!("JIT assignment suppressed due to semicolon, captured for type info");
                                        }
                                    }
                                }
                            }

                            if self.verbose {
                                debug!(
                                    "{} assignment successful, variable_array: {:?}",
                                    if actual_used_jit {
                                        "JIT"
                                    } else {
                                        "Interpreter"
                                    },
                                    self.variable_array
                                );
                            }
                        }
                        Err(e) => {
                            if self.verbose {
                                debug!("JIT execution failed: {e}, using interpreter");
                            }
                            // Fall back to interpreter
                        }
                    }
                }
            }
        }

        // Use interpreter if JIT failed or is disabled
        if !execution_completed {
            if self.verbose {
                debug!(
                    "Interpreter path: variable_array size: {}, bytecode.var_count: {}",
                    self.variable_array.len(),
                    bytecode.var_count
                );
            }

            // For expressions, modify bytecode to store result in a temp variable instead of using stack
            let mut execution_bytecode = bytecode.clone();
            if is_expression_stmt && !execution_bytecode.instructions.is_empty() {
                execution_bytecode.instructions.pop(); // Remove the Pop instruction

                // Add StoreVar instruction to store the result in a temporary variable
                let temp_var_id = std::cmp::max(execution_bytecode.var_count, max_var_id + 1);
                execution_bytecode
                    .instructions
                    .push(runmat_ignition::Instr::StoreVar(temp_var_id));
                execution_bytecode.var_count = temp_var_id + 1; // Expand variable count for temp variable

                // Ensure our variable array can hold the temporary variable
                if self.variable_array.len() <= temp_var_id {
                    self.variable_array.resize(temp_var_id + 1, Value::Num(0.0));
                }

                if self.verbose {
                    debug!(
                        "Modified expression bytecode, new instructions: {:?}",
                        execution_bytecode.instructions
                    );
                }
            }

            match self.interpret_with_context(&execution_bytecode).await {
                Ok(runmat_ignition::InterpreterOutcome::Completed(results)) => {
                    // Only increment interpreter_fallback if JIT wasn't attempted
                    if !self.has_jit() || is_expression_stmt {
                        self.stats.interpreter_fallback += 1;
                    }
                    if self.verbose {
                        debug!("Interpreter results: {results:?}");
                    }

                    // Handle assignment statements (x = 42 should show the assigned value unless suppressed)
                    if hir.body.len() == 1 {
                        if let runmat_hir::HirStmt::Assign(var_id, _, _, _) = &hir.body[0] {
                            if let Some(name) = id_to_name.get(&var_id.0) {
                                assigned_this_execution.insert(name.clone());
                            }
                            // For assignments, capture the assigned value for both display and type info
                            if var_id.0 < self.variable_array.len() {
                                let assignment_value = self.variable_array[var_id.0].clone();
                                if !is_semicolon_suppressed {
                                    result_value = Some(assignment_value);
                                    if self.verbose {
                                        debug!("Interpreter assignment result: {result_value:?}");
                                    }
                                } else {
                                    suppressed_value = Some(assignment_value);
                                    if self.verbose {
                                        debug!("Interpreter assignment suppressed due to semicolon, captured for type info");
                                    }
                                }
                            }
                        } else if !is_expression_stmt
                            && !results.is_empty()
                            && !is_semicolon_suppressed
                            && matches!(final_stmt_emit, FinalStmtEmitDisposition::NeedsFallback)
                        {
                            result_value = Some(results[0].clone());
                        }
                    }

                    // For expressions, get the result from the temporary variable (capture for both display and type info)
                    if is_expression_stmt
                        && !execution_bytecode.instructions.is_empty()
                        && result_value.is_none()
                        && suppressed_value.is_none()
                    {
                        let temp_var_id = execution_bytecode.var_count - 1; // The temp variable we added
                        if temp_var_id < self.variable_array.len() {
                            let expression_value = self.variable_array[temp_var_id].clone();
                            if !is_semicolon_suppressed {
                                // Capture for 'ans' update when output is not suppressed
                                ans_update = Some((temp_var_id, expression_value.clone()));
                                result_value = Some(expression_value);
                                if self.verbose {
                                    debug!("Expression result from temp var {temp_var_id}: {result_value:?}");
                                }
                            } else {
                                suppressed_value = Some(expression_value);
                                if self.verbose {
                                    debug!("Expression suppressed, captured for type info from temp var {temp_var_id}: {suppressed_value:?}");
                                }
                            }
                        }
                    } else if !is_semicolon_suppressed
                        && matches!(final_stmt_emit, FinalStmtEmitDisposition::NeedsFallback)
                        && result_value.is_none()
                    {
                        result_value = results.into_iter().last();
                        if self.verbose {
                            debug!("Fallback result from interpreter: {result_value:?}");
                        }
                    }

                    if self.verbose {
                        debug!("Final result_value: {result_value:?}");
                    }
                    debug!("Interpreter execution successful");
                }

                Err(e) => {
                    debug!("Interpreter execution failed: {e}");
                    error = Some(e);
                }
            }
        }

        let execution_time = start_time.elapsed();
        let execution_time_ms = execution_time.as_millis() as u64;

        self.stats.total_execution_time_ms += execution_time_ms;
        self.stats.average_execution_time_ms =
            self.stats.total_execution_time_ms as f64 / self.stats.total_executions as f64;

        // Update variable names mapping and function definitions if execution was successful
        if error.is_none() {
            if let Some((mutated_names, assigned)) = runmat_ignition::take_updated_workspace_state()
            {
                if debug_trace {
                    debug!(
                        ?mutated_names,
                        ?assigned,
                        "[repl] mutated names and assigned return values"
                    );
                }
                self.variable_names = mutated_names.clone();
                let mut new_assigned: HashSet<String> = assigned
                    .difference(&prev_assigned_snapshot)
                    .cloned()
                    .collect();
                new_assigned.extend(assigned_this_execution.iter().cloned());
                for (name, var_id) in &mutated_names {
                    if *var_id >= self.variable_array.len() {
                        continue;
                    }
                    let new_value = &self.variable_array[*var_id];
                    let changed = match self.workspace_values.get(name) {
                        Some(old_value) => old_value != new_value,
                        None => true,
                    };
                    if changed {
                        new_assigned.insert(name.clone());
                    }
                }
                if debug_trace {
                    debug!(?new_assigned, "[repl] new assignments");
                }
                for name in new_assigned {
                    let var_id = mutated_names.get(&name).copied().or_else(|| {
                        id_to_name
                            .iter()
                            .find_map(|(vid, n)| if n == &name { Some(*vid) } else { None })
                    });
                    if let Some(var_id) = var_id {
                        if var_id < self.variable_array.len() {
                            let value_clone = self.variable_array[var_id].clone();
                            self.workspace_values
                                .insert(name.clone(), value_clone.clone());
                            workspace_updates.push(workspace_entry(&name, &value_clone));
                            if debug_trace {
                                debug!(name, ?value_clone, "[repl] workspace update");
                            }
                        }
                    }
                }
            } else {
                for name in &assigned_this_execution {
                    if let Some(var_id) =
                        id_to_name
                            .iter()
                            .find_map(|(vid, n)| if n == name { Some(*vid) } else { None })
                    {
                        if var_id < self.variable_array.len() {
                            let value_clone = self.variable_array[var_id].clone();
                            self.workspace_values
                                .insert(name.clone(), value_clone.clone());
                            workspace_updates.push(workspace_entry(name, &value_clone));
                        }
                    }
                }
            }
            let mut repl_source_id: Option<SourceId> = None;
            for (name, stmt) in &updated_functions {
                if matches!(stmt, runmat_hir::HirStmt::Function { .. }) {
                    let source_id = *repl_source_id
                        .get_or_insert_with(|| self.source_pool.intern("<repl>", input));
                    self.function_source_ids.insert(name.clone(), source_id);
                }
            }
            self.function_definitions = updated_functions;
            // Apply 'ans' update if applicable (persisting expression result)
            if let Some((var_id, value)) = ans_update {
                self.variable_names.insert("ans".to_string(), var_id);
                self.workspace_values.insert("ans".to_string(), value);
                if debug_trace {
                    println!("Updated 'ans' to var_id {}", var_id);
                }
            }
        }

        if self.verbose {
            debug!("Execution completed in {execution_time_ms}ms (JIT: {used_jit})");
        }

        if !is_expression_stmt
            && !is_semicolon_suppressed
            && matches!(final_stmt_emit, FinalStmtEmitDisposition::NeedsFallback)
            && result_value.is_none()
        {
            if let Some(v) = self
                .variable_array
                .iter()
                .rev()
                .find(|v| !matches!(v, Value::Num(0.0)))
                .cloned()
            {
                result_value = Some(v);
            }
        }

        if !is_semicolon_suppressed
            && matches!(final_stmt_emit, FinalStmtEmitDisposition::NeedsFallback)
        {
            if let Some(value) = result_value.as_ref() {
                let label = determine_display_label_from_context(
                    single_assign_var,
                    &id_to_name,
                    is_expression_stmt,
                    single_stmt_non_assign,
                );
                runmat_runtime::console::record_value_output(label.as_deref(), value);
            }
        }

        // Generate type info if we have a suppressed value
        let type_info = suppressed_value.as_ref().map(format_type_info);

        let streams = runmat_runtime::console::take_thread_buffer()
            .into_iter()
            .map(|entry| ExecutionStreamEntry {
                stream: match entry.stream {
                    runmat_runtime::console::ConsoleStream::Stdout => ExecutionStreamKind::Stdout,
                    runmat_runtime::console::ConsoleStream::Stderr => ExecutionStreamKind::Stderr,
                },
                text: entry.text,
                timestamp_ms: entry.timestamp_ms,
            })
            .collect();
        let (workspace_entries, snapshot_full) = if workspace_updates.is_empty() {
            let source_map = if self.workspace_values.is_empty() {
                &self.variables
            } else {
                &self.workspace_values
            };
            if source_map.is_empty() {
                (workspace_updates, false)
            } else {
                let mut entries: Vec<WorkspaceEntry> = source_map
                    .iter()
                    .map(|(name, value)| workspace_entry(name, value))
                    .collect();
                entries.sort_by(|a, b| a.name.cmp(&b.name));
                (entries, true)
            }
        } else {
            (workspace_updates, false)
        };
        let workspace_snapshot = self.build_workspace_snapshot(workspace_entries, snapshot_full);
        let figures_touched = runmat_runtime::plotting_hooks::take_recent_figures();
        let stdin_events = stdin_events
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_default();

        let warnings = runmat_runtime::warning_store::take_all();

        if let Some(runtime_error) = &mut error {
            self.populate_callstack(runtime_error);
        }

        let public_value = if is_semicolon_suppressed {
            None
        } else {
            result_value
        };

        Ok(ExecutionResult {
            value: public_value,
            execution_time_ms,
            used_jit,
            error,
            type_info,
            streams,
            workspace: workspace_snapshot,
            figures_touched,
            warnings,
            profiling: gather_profiling(execution_time_ms),
            fusion_plan: fusion_snapshot,
            stdin_events,
        })
    }

    /// Get execution statistics
    pub fn stats(&self) -> &ExecutionStats {
        &self.stats
    }

    /// Reset execution statistics
    pub fn reset_stats(&mut self) {
        self.stats = ExecutionStats::default();
    }

    /// Clear all variables in the session context
    pub fn clear_variables(&mut self) {
        self.variables.clear();
        self.variable_array.clear();
        self.variable_names.clear();
        self.workspace_values.clear();
        self.workspace_preview_tokens.clear();
    }

    /// Control whether fusion plan snapshots are emitted in [`ExecutionResult`].
    pub fn set_emit_fusion_plan(&mut self, enabled: bool) {
        self.emit_fusion_plan = enabled;
    }

    /// Return the active language compatibility mode.
    pub fn compat_mode(&self) -> CompatMode {
        self.compat_mode
    }

    /// Set the language compatibility mode (`matlab` or `strict`).
    pub fn set_compat_mode(&mut self, mode: CompatMode) {
        self.compat_mode = mode;
    }

    pub fn set_callstack_limit(&mut self, limit: usize) {
        self.callstack_limit = limit;
        runmat_ignition::set_call_stack_limit(limit);
    }

    /// Materialize a workspace variable for inspection (optionally identified by preview token).
    pub fn materialize_variable(
        &mut self,
        target: WorkspaceMaterializeTarget,
        options: WorkspaceMaterializeOptions,
    ) -> Result<MaterializedVariable> {
        let name = match target {
            WorkspaceMaterializeTarget::Name(name) => name,
            WorkspaceMaterializeTarget::Token(id) => self
                .workspace_preview_tokens
                .get(&id)
                .map(|ticket| ticket.name.clone())
                .ok_or_else(|| anyhow::anyhow!("Unknown workspace preview token"))?,
        };
        let value = self
            .workspace_values
            .get(&name)
            .or_else(|| self.variables.get(&name))
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Variable '{name}' not found in workspace"))?;

        let is_gpu = matches!(value, Value::GpuTensor(_));
        let residency = if is_gpu {
            WorkspaceResidency::Gpu
        } else {
            WorkspaceResidency::Cpu
        };
        let host_value = if is_gpu {
            gpu_helpers::gather_value(&value).map_err(|err| {
                anyhow::anyhow!("Failed to gather gpuArray '{name}' for preview: {err}")
            })?
        } else {
            value.clone()
        };

        let value_shape_vec = value_shape(&host_value).unwrap_or_default();
        let mut preview = None;
        if let Some(slice_opts) = options
            .slice
            .as_ref()
            .and_then(|slice| slice.sanitized(&value_shape_vec))
        {
            let slice_elements = slice_opts.shape.iter().product::<usize>();
            let slice_limit = slice_elements.clamp(1, MATERIALIZE_DEFAULT_LIMIT);
            if let Some(slice_value) = slice_value_for_preview(&host_value, &slice_opts) {
                preview = preview_numeric_values(&slice_value, slice_limit)
                    .map(|(values, truncated)| WorkspacePreview { values, truncated });
            }
        }
        if preview.is_none() {
            let max_elements = options.max_elements.clamp(1, MATERIALIZE_DEFAULT_LIMIT);
            preview = preview_numeric_values(&host_value, max_elements)
                .map(|(values, truncated)| WorkspacePreview { values, truncated });
        }
        Ok(MaterializedVariable {
            name,
            class_name: matlab_class_name(&host_value),
            dtype: numeric_dtype_label(&host_value).map(|label| label.to_string()),
            shape: value_shape_vec,
            is_gpu,
            residency,
            size_bytes: approximate_size_bytes(&host_value),
            preview,
            value: host_value,
        })
    }

    /// Get a copy of current variables
    pub fn get_variables(&self) -> &HashMap<String, Value> {
        &self.variables
    }

    /// Interpret bytecode with persistent variable context
    async fn interpret_with_context(
        &mut self,
        bytecode: &runmat_ignition::Bytecode,
    ) -> Result<runmat_ignition::InterpreterOutcome, RuntimeError> {
        runmat_ignition::interpret_with_vars(bytecode, &mut self.variable_array, Some("<repl>"))
            .await
    }

    /// Prepare variable array for execution by populating with existing values
    fn prepare_variable_array_for_execution(
        &mut self,
        bytecode: &runmat_ignition::Bytecode,
        updated_var_mapping: &HashMap<String, usize>,
        debug_trace: bool,
    ) {
        // Create a new variable array of the correct size
        let max_var_id = updated_var_mapping.values().copied().max().unwrap_or(0);
        let required_len = std::cmp::max(bytecode.var_count, max_var_id + 1);
        let mut new_variable_array = vec![Value::Num(0.0); required_len];
        if debug_trace {
            debug!(
                bytecode_var_count = bytecode.var_count,
                required_len, max_var_id, "[repl] prepare variable array"
            );
        }

        // Populate with existing values based on the variable mapping
        for (var_name, &new_var_id) in updated_var_mapping {
            if new_var_id < new_variable_array.len() {
                if let Some(value) = self.workspace_values.get(var_name) {
                    if debug_trace {
                        debug!(
                            var_name,
                            var_id = new_var_id,
                            ?value,
                            "[repl] prepare set var"
                        );
                    }
                    new_variable_array[new_var_id] = value.clone();
                }
            } else if debug_trace {
                debug!(
                    var_name,
                    var_id = new_var_id,
                    len = new_variable_array.len(),
                    "[repl] prepare skipping var"
                );
            }
        }

        // Update our variable array and mapping
        self.variable_array = new_variable_array;
    }

    /// Convert stored HIR function definitions to UserFunction format for compilation
    fn convert_hir_functions_to_user_functions(
        &self,
    ) -> HashMap<String, runmat_ignition::UserFunction> {
        let mut user_functions = HashMap::new();

        for (name, hir_stmt) in &self.function_definitions {
            if let runmat_hir::HirStmt::Function {
                name: func_name,
                params,
                outputs,
                body,
                has_varargin: _,
                has_varargout: _,
                ..
            } = hir_stmt
            {
                // Use the existing HIR utilities to calculate variable count
                let var_map =
                    runmat_hir::remapping::create_complete_function_var_map(params, outputs, body);
                let max_local_var = var_map.len();

                let source_id = self.function_source_ids.get(name).copied();
                if let Some(id) = source_id {
                    if let Some(source) = self.source_pool.get(id) {
                        let _ = (&source.name, &source.text);
                    }
                }
                let user_func = runmat_ignition::UserFunction {
                    name: func_name.clone(),
                    params: params.clone(),
                    outputs: outputs.clone(),
                    body: body.clone(),
                    local_var_count: max_local_var,
                    has_varargin: false,
                    has_varargout: false,
                    var_types: vec![Type::Unknown; max_local_var],
                    source_id,
                };
                user_functions.insert(name.clone(), user_func);
            }
        }

        user_functions
    }

    /// Configure garbage collector
    pub fn configure_gc(&self, config: GcConfig) -> Result<()> {
        gc_configure(config)
            .map_err(|e| anyhow::anyhow!("Failed to configure garbage collector: {}", e))
    }

    /// Get GC statistics
    pub fn gc_stats(&self) -> runmat_gc::GcStats {
        gc_stats()
    }

    /// Show detailed system information
    pub fn show_system_info(&self) {
        let gc_stats = self.gc_stats();
        info!(
            jit = %if self.has_jit() { "available" } else { "disabled/failed" },
            verbose = self.verbose,
            total_executions = self.stats.total_executions,
            jit_compiled = self.stats.jit_compiled,
            interpreter_fallback = self.stats.interpreter_fallback,
            avg_time_ms = self.stats.average_execution_time_ms,
            total_allocations = gc_stats
                .total_allocations
                .load(std::sync::atomic::Ordering::Relaxed),
            minor_collections = gc_stats
                .minor_collections
                .load(std::sync::atomic::Ordering::Relaxed),
            major_collections = gc_stats
                .major_collections
                .load(std::sync::atomic::Ordering::Relaxed),
            current_memory_mb = gc_stats
                .current_memory_usage
                .load(std::sync::atomic::Ordering::Relaxed) as f64
                / 1024.0
                / 1024.0,
            workspace_vars = self.workspace_values.len(),
            "RunMat Session Status"
        );
    }

    #[cfg(feature = "jit")]
    fn has_jit(&self) -> bool {
        self.jit_engine.is_some()
    }

    #[cfg(not(feature = "jit"))]
    fn has_jit(&self) -> bool {
        false
    }

    fn build_workspace_snapshot(
        &mut self,
        entries: Vec<WorkspaceEntry>,
        full: bool,
    ) -> WorkspaceSnapshot {
        self.workspace_version = self.workspace_version.wrapping_add(1);
        let version = self.workspace_version;
        self.workspace_preview_tokens.clear();
        let mut values = Vec::with_capacity(entries.len());
        for mut entry in entries {
            let token = Uuid::new_v4();
            self.workspace_preview_tokens.insert(
                token,
                WorkspaceMaterializeTicket {
                    name: entry.name.clone(),
                },
            );
            entry.preview_token = Some(token);
            values.push(entry);
        }
        WorkspaceSnapshot {
            full,
            version,
            values,
        }
    }
}

fn last_displayable_statement_emit_disposition(
    body: &[runmat_hir::HirStmt],
) -> FinalStmtEmitDisposition {
    use runmat_hir::HirStmt;

    for stmt in body.iter().rev() {
        match stmt {
            HirStmt::ExprStmt(expr, _, _) => return expr_emit_disposition(expr),
            HirStmt::Assign(_, _, _, _) | HirStmt::MultiAssign(_, _, _, _) => {
                return FinalStmtEmitDisposition::Suppressed
            }
            HirStmt::AssignLValue(_, _, _, _) => return FinalStmtEmitDisposition::Suppressed,

            _ => continue,
        }
    }
    FinalStmtEmitDisposition::Suppressed
}

fn expr_emit_disposition(expr: &runmat_hir::HirExpr) -> FinalStmtEmitDisposition {
    use runmat_hir::HirExprKind;
    if let HirExprKind::FuncCall(name, _) = &expr.kind {
        if runmat_builtins::suppresses_auto_output(name) {
            return FinalStmtEmitDisposition::Suppressed;
        }
    }
    FinalStmtEmitDisposition::Inline
}

const WORKSPACE_PREVIEW_LIMIT: usize = 16;
const MATERIALIZE_DEFAULT_LIMIT: usize = 4096;

fn workspace_entry(name: &str, value: &Value) -> WorkspaceEntry {
    let dtype = numeric_dtype_label(value).map(|label| label.to_string());
    let preview = preview_numeric_values(value, WORKSPACE_PREVIEW_LIMIT)
        .map(|(values, truncated)| WorkspacePreview { values, truncated });
    let residency = if matches!(value, Value::GpuTensor(_)) {
        WorkspaceResidency::Gpu
    } else {
        WorkspaceResidency::Cpu
    };
    WorkspaceEntry {
        name: name.to_string(),
        class_name: matlab_class_name(value),
        dtype,
        shape: value_shape(value).unwrap_or_default(),
        is_gpu: matches!(value, Value::GpuTensor(_)),
        size_bytes: approximate_size_bytes(value),
        preview,
        residency,
        preview_token: None,
    }
}

struct ActiveExecutionGuard {
    flag: *mut bool,
}

impl ActiveExecutionGuard {
    fn new(session: &mut RunMatSession) -> Result<Self> {
        if session.is_executing {
            Err(anyhow::anyhow!(
                "RunMatSession is already executing another script"
            ))
        } else {
            session.is_executing = true;
            Ok(Self {
                flag: &mut session.is_executing,
            })
        }
    }
}

impl Drop for ActiveExecutionGuard {
    fn drop(&mut self) {
        unsafe {
            if let Some(flag) = self.flag.as_mut() {
                *flag = false;
            }
        }
    }
}

impl Default for RunMatSession {
    fn default() -> Self {
        Self::new().expect("Failed to create default RunMat session")
    }
}

/// Tokenize the input string and return a space separated string of token names.
/// This is kept for backward compatibility with existing tests.
pub fn format_tokens(input: &str) -> String {
    tokenize_detailed(input)
        .into_iter()
        .map(|t| format!("{:?}", t.token))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Execute MATLAB/Octave code and return the result as a formatted string
pub async fn execute_and_format(input: &str) -> String {
    match RunMatSession::new() {
        Ok(mut engine) => match engine.execute(input).await {
            Ok(result) => {
                if let Some(error) = result.error {
                    format!("Error: {error}")
                } else if let Some(value) = result.value {
                    format!("{value:?}")
                } else {
                    "".to_string()
                }
            }
            Err(e) => format!("Error: {e}"),
        },
        Err(e) => format!("Engine Error: {e}"),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn reset_provider_telemetry() {
    if let Some(provider) = accel_provider() {
        provider.reset_telemetry();
    }
}

#[cfg(target_arch = "wasm32")]
fn reset_provider_telemetry() {}

#[cfg(not(target_arch = "wasm32"))]
fn gather_profiling(execution_time_ms: u64) -> Option<ExecutionProfiling> {
    let provider = accel_provider()?;
    let telemetry = provider.telemetry_snapshot();
    let gpu_ns = telemetry.fused_elementwise.total_wall_time_ns
        + telemetry.fused_reduction.total_wall_time_ns
        + telemetry.matmul.total_wall_time_ns;
    let gpu_ms = gpu_ns.saturating_div(1_000_000);
    Some(ExecutionProfiling {
        total_ms: execution_time_ms,
        cpu_ms: Some(execution_time_ms.saturating_sub(gpu_ms)),
        gpu_ms: Some(gpu_ms),
        kernel_count: Some(
            (telemetry.fused_elementwise.count
                + telemetry.fused_reduction.count
                + telemetry.matmul.count
                + telemetry.kernel_launches.len() as u64)
                .min(u32::MAX as u64) as u32,
        ),
    })
}

#[cfg(target_arch = "wasm32")]
fn gather_profiling(execution_time_ms: u64) -> Option<ExecutionProfiling> {
    Some(ExecutionProfiling {
        total_ms: execution_time_ms,
        ..ExecutionProfiling::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[test]
    fn captures_basic_workspace_assignments() {
        let mut session =
            RunMatSession::with_snapshot_bytes(false, false, None).expect("session init");
        let result = block_on(session.execute("x = 42;")).expect("exec succeeds");
        assert!(
            result
                .workspace
                .values
                .iter()
                .any(|entry| entry.name == "x"),
            "workspace snapshot should include assigned variable"
        );
    }
}
