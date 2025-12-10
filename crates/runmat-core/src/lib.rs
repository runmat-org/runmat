use anyhow::Result;
use log::{debug, info, warn};
use runmat_builtins::{Type, Value};
use runmat_gc::{gc_configure, gc_stats, GcConfig};

#[cfg(not(target_arch = "wasm32"))]
use runmat_accelerate_api::provider as accel_provider;
use runmat_lexer::{tokenize_detailed, Token as LexToken};
use runmat_parser::parse;
use runmat_runtime::builtins::common::gpu_helpers;
use runmat_runtime::warning_store::RuntimeWarning;
use runmat_snapshot::{Snapshot, SnapshotConfig, SnapshotLoader};
#[cfg(feature = "jit")]
use runmat_turbine::TurbineEngine;
use std::collections::{HashMap, HashSet};
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::time::Instant;
use uuid::Uuid;

#[cfg(not(target_arch = "wasm32"))]
mod fusion_snapshot;
mod value_metadata;
#[cfg(not(target_arch = "wasm32"))]
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
    /// Loaded snapshot for standard library preloading
    snapshot: Option<Arc<Snapshot>>,
    /// Cooperative cancellation flag shared with the runtime.
    interrupt_flag: Arc<AtomicBool>,
    /// Tracks whether an execution is currently active.
    is_executing: bool,
    /// Optional session-level input handler supplied by the host.
    input_handler: Option<SharedInputHandler>,
    pending_executions: HashMap<Uuid, PendingFrame>,
    telemetry_consent: bool,
    telemetry_client_id: Option<String>,
    workspace_preview_tokens: HashMap<Uuid, WorkspaceMaterializeTicket>,
    workspace_version: u64,
    emit_fusion_plan: bool,
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
    Pending,
}

type SharedInputHandler = Arc<dyn Fn(&InputRequest) -> InputHandlerAction + Send + Sync>;

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
pub struct WorkspaceMaterializeOptions {
    pub max_elements: usize,
}

impl Default for WorkspaceMaterializeOptions {
    fn default() -> Self {
        Self {
            max_elements: MATERIALIZE_DEFAULT_LIMIT,
        }
    }
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
    pub error: Option<String>,
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
    /// Pending stdin request if the execution is waiting for input.
    pub stdin_requested: Option<PendingInput>,
}

#[derive(Debug, Clone)]
pub struct PendingInput {
    pub id: Uuid,
    pub request: InputRequest,
    pub waiting_ms: u64,
}

struct PendingFrame {
    plan: ExecutionPlan,
    pending: runmat_ignition::PendingExecution,
    streams: Vec<ExecutionStreamEntry>,
    pending_since: Instant,
}

#[derive(Debug, Clone)]
struct WorkspaceMaterializeTicket {
    name: String,
}

struct ExecutionPlan {
    assigned_this_execution: HashSet<String>,
    id_to_name: HashMap<usize, String>,
    prev_assigned_snapshot: HashSet<String>,
    updated_functions: HashMap<String, runmat_hir::HirStmt>,
    execution_bytecode: runmat_ignition::Bytecode,
    single_assign_var: Option<usize>,
    single_stmt_non_assign: bool,
    is_expression_stmt: bool,
    is_semicolon_suppressed: bool,
    result_value: Option<Value>,
    suppressed_value: Option<Value>,
    error: Option<String>,
    workspace_updates: Vec<WorkspaceEntry>,
    fusion_snapshot: Option<FusionPlanSnapshot>,
    start_time: Instant,
    used_jit: bool,
    stdin_events: Arc<Mutex<Vec<StdinEvent>>>,
    workspace_guard: Option<runmat_ignition::PendingWorkspaceGuard>,
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
            snapshot,
            interrupt_flag: Arc::new(AtomicBool::new(false)),
            is_executing: false,
            input_handler: None,
            pending_executions: HashMap::new(),
            telemetry_consent: true,
            telemetry_client_id: None,
            workspace_preview_tokens: HashMap::new(),
            workspace_version: 0,
            emit_fusion_plan: false,
        };

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

    pub fn pending_requests(&self) -> Vec<PendingInput> {
        let now = Instant::now();
        self.pending_executions
            .iter()
            .map(|(id, frame)| PendingInput {
                id: *id,
                request: pending_interaction_to_request(&frame.pending.interaction),
                waiting_ms: now
                    .saturating_duration_since(frame.pending_since)
                    .as_millis() as u64,
            })
            .collect()
    }

    /// Cancel a specific pending stdin request by ID. Returns true if the request existed.
    pub fn cancel_pending_request(&mut self, request_id: Uuid) -> bool {
        self.pending_executions.remove(&request_id).is_some()
    }

    /// Drop all pending stdin requests (useful when the host wants to hard-reset a session).
    pub fn cancel_all_pending_requests(&mut self) {
        self.pending_executions.clear();
    }

    pub fn resume_input(
        &mut self,
        request_id: Uuid,
        response: Result<InputResponse, String>,
    ) -> Result<ExecutionResult> {
        let mut frame = self
            .pending_executions
            .remove(&request_id)
            .ok_or_else(|| anyhow::anyhow!("Unknown pending request {request_id}"))?;
        let _active = ActiveExecutionGuard::new(self)?;
        runmat_runtime::interaction::push_queued_response(response.map(map_input_response));
        let stdin_events = Arc::clone(&frame.plan.stdin_events);
        let runtime_handler = self.build_runtime_input_handler(stdin_events);
        let _input_guard = runmat_runtime::interaction::replace_handler(Some(runtime_handler));
        let _interrupt_guard =
            runmat_runtime::interrupt::replace_interrupt(Some(self.interrupt_flag.clone()));
        match runmat_ignition::resume_with_state(frame.pending.state, &mut self.variable_array) {
            Ok(runmat_ignition::InterpreterOutcome::Completed(values)) => {
                let mut streams = frame.streams;
                streams.extend(drain_console_streams());
                self.finalize_pending_execution(frame.plan, values, streams)
            }
            Ok(runmat_ignition::InterpreterOutcome::Pending(next_pending)) => {
                let chunk = drain_console_streams();
                frame.streams.extend(chunk.clone());
                frame.pending = next_pending;
                let new_id = Uuid::new_v4();
                let elapsed_ms = frame.plan.start_time.elapsed().as_millis() as u64;
                let fusion_plan = frame.plan.fusion_snapshot.clone();
                let stdin_arc = Arc::clone(&frame.plan.stdin_events);
                let used_jit = frame.plan.used_jit;
                let pending_input = PendingInput {
                    id: new_id,
                    request: pending_interaction_to_request(&frame.pending.interaction),
                    waiting_ms: 0,
                };
                let stdin_events = stdin_arc
                    .lock()
                    .map(|guard| guard.clone())
                    .unwrap_or_default();
                frame.pending_since = Instant::now();
                self.pending_executions.insert(new_id, frame);
                Ok(ExecutionResult {
                    value: None,
                    execution_time_ms: elapsed_ms,
                    used_jit,
                    error: None,
                    type_info: None,
                    streams: chunk,
                    workspace: self.build_workspace_snapshot(Vec::new(), false),
                    figures_touched: Vec::new(),
                    warnings: Vec::new(),
                    profiling: None,
                    fusion_plan,
                    stdin_events,
                    stdin_requested: Some(pending_input),
                })
            }
            Err(err) => Err(anyhow::anyhow!(err)),
        }
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
            let request_kind = match prompt.kind {
                runmat_runtime::interaction::InteractionKind::Line { echo } => {
                    InputRequestKind::Line { echo }
                }
                runmat_runtime::interaction::InteractionKind::KeyPress => {
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
                            .map(InputResponse::Line)
                            .map_err(|e| e),
                    ),
                    InputRequestKind::KeyPress => InputHandlerAction::Respond(
                        runmat_runtime::interaction::default_wait_for_key(&request.prompt)
                            .map(|_| InputResponse::KeyPress)
                            .map_err(|e| e),
                    ),
                }
            };
            match action {
                InputHandlerAction::Respond(result) => {
                    let mapped = result
                        .map_err(|err| {
                            event.error = Some(err.clone());
                            if let Ok(mut guard) = stdin_events.lock() {
                                guard.push(event.clone());
                            }
                            err
                        })
                        .and_then(|resp| match resp {
                            InputResponse::Line(value) => {
                                event.value = Some(value.clone());
                                if let Ok(mut guard) = stdin_events.lock() {
                                    guard.push(event);
                                }
                                Ok(runmat_runtime::interaction::InteractionResponse::Line(
                                    value,
                                ))
                            }
                            InputResponse::KeyPress => {
                                if let Ok(mut guard) = stdin_events.lock() {
                                    guard.push(event);
                                }
                                Ok(runmat_runtime::interaction::InteractionResponse::KeyPress)
                            }
                        });
                    runmat_runtime::interaction::InteractionDecision::Respond(mapped)
                }
                InputHandlerAction::Pending => {
                    if let Ok(mut guard) = stdin_events.lock() {
                        guard.push(event);
                    }
                    runmat_runtime::interaction::InteractionDecision::Pending
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

    /// Execute MATLAB/Octave code
    pub fn execute(&mut self, input: &str) -> Result<ExecutionResult> {
        if !self.pending_executions.is_empty() {
            return Err(anyhow::anyhow!(
                "Cannot execute new code while pending stdin requests exist"
            ));
        }
        let _active = ActiveExecutionGuard::new(self)?;
        self.execute_internal(input)
    }

    fn execute_internal(&mut self, input: &str) -> Result<ExecutionResult> {
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

        // Parse the input
        let ast = parse(input)
            .map_err(|e| anyhow::anyhow!("Failed to parse input '{}': {}", input, e))?;
        if self.verbose {
            debug!("AST: {ast:?}");
        }

        // Lower to HIR with existing variable and function context
        let lowering_result = runmat_hir::lower_with_full_context(
            &ast,
            &self.variable_names,
            &self.function_definitions,
        )
        .map_err(|e| anyhow::anyhow!("Failed to lower to HIR: {}", e))?;
        let (hir, updated_vars, updated_functions, var_names_map) = (
            lowering_result.hir,
            lowering_result.variables,
            lowering_result.functions,
            lowering_result.var_names,
        );
        let max_var_id = updated_vars.values().copied().max().unwrap_or(0);
        if debug_trace {
            println!("updated_vars: {:?}", updated_vars);
        }
        if debug_trace {
            println!("workspace_values_before: {:?}", self.workspace_values);
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
            println!("assigned_snapshot: {:?}", assigned_snapshot);
        }
        let mut pending_workspace_guard = Some(runmat_ignition::push_pending_workspace(
            updated_vars.clone(),
            assigned_snapshot.clone(),
        ));
        if self.verbose {
            debug!("HIR generated successfully");
        }

        let (single_assign_var, single_stmt_non_assign) = if hir.body.len() == 1 {
            match &hir.body[0] {
                runmat_hir::HirStmt::Assign(var_id, _, _) => (Some(var_id.0), false),
                _ => (None, true),
            }
        } else {
            (None, false)
        };

        // Compile to bytecode with existing function definitions
        let existing_functions = self.convert_hir_functions_to_user_functions();
        let bytecode = runmat_ignition::compile_with_functions(&hir, &existing_functions)
            .map_err(|e| anyhow::anyhow!("Failed to compile to bytecode: {}", e))?;
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

        #[cfg_attr(not(feature = "jit"), allow(unused_mut))]
        let mut used_jit = false;
        let mut result_value: Option<Value> = None; // Always start fresh for each execution
        let mut suppressed_value: Option<Value> = None; // Track value for type info when suppressed
        let mut error = None;
        let mut workspace_updates: Vec<WorkspaceEntry> = Vec::new();

        // Check if this is an expression statement (ends with Pop)
        let is_expression_stmt = bytecode
            .instructions
            .last()
            .map(|instr| matches!(instr, runmat_ignition::Instr::Pop))
            .unwrap_or(false);

        // Detect whether the user's input ends with a semicolon at the token level
        let ends_with_semicolon = {
            let toks = tokenize_detailed(input);
            toks.into_iter()
                .rev()
                .map(|t| t.token)
                .find(|_| true)
                .map(|t| matches!(t, LexToken::Semicolon))
                .unwrap_or(false)
        };

        // Check if this is a semicolon-suppressed statement (expression or assignment)
        // Control flow statements never return values regardless of semicolons
        let is_semicolon_suppressed = if hir.body.len() == 1 {
            match &hir.body[0] {
                runmat_hir::HirStmt::ExprStmt(_, _) => ends_with_semicolon,
                runmat_hir::HirStmt::Assign(_, _, _) => ends_with_semicolon,
                runmat_hir::HirStmt::If { .. }
                | runmat_hir::HirStmt::While { .. }
                | runmat_hir::HirStmt::For { .. }
                | runmat_hir::HirStmt::Break
                | runmat_hir::HirStmt::Continue
                | runmat_hir::HirStmt::Return
                | runmat_hir::HirStmt::Function { .. }
                | runmat_hir::HirStmt::MultiAssign(_, _, _)
                | runmat_hir::HirStmt::AssignLValue(_, _, _)
                | runmat_hir::HirStmt::Switch { .. }
                | runmat_hir::HirStmt::TryCatch { .. }
                | runmat_hir::HirStmt::Global(_)
                | runmat_hir::HirStmt::Persistent(_)
                | runmat_hir::HirStmt::Import {
                    path: _,
                    wildcard: _,
                }
                | runmat_hir::HirStmt::ClassDef { .. } => true,
            }
        } else {
            false
        };

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
                            if actual_used_jit {
                                self.stats.jit_compiled += 1;
                            } else {
                                self.stats.interpreter_fallback += 1;
                            }
                            // For assignments, capture the assigned value for both display and type info
                            // Prefer the variable slot indicated by HIR if available.
                            let assignment_value =
                                if let Some(runmat_hir::HirStmt::Assign(var_id, _, _)) =
                                    hir.body.first()
                                {
                                    if let Some(name) = id_to_name.get(&var_id.0) {
                                        assigned_this_execution.insert(name.clone());
                                    }
                                    if var_id.0 < self.variable_array.len() {
                                        Some(self.variable_array[var_id.0].clone())
                                    } else {
                                        None
                                    }
                                } else {
                                    self.variable_array
                                        .iter()
                                        .rev()
                                        .find(|v| !matches!(v, Value::Num(0.0)))
                                        .cloned()
                                };

                            if !is_semicolon_suppressed {
                                result_value = assignment_value.clone();
                                if self.verbose {
                                    debug!("JIT assignment result: {result_value:?}");
                                }
                            } else {
                                suppressed_value = assignment_value;
                                if self.verbose {
                                    debug!("JIT assignment suppressed due to semicolon, captured for type info");
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
        if !used_jit {
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

            match self.interpret_with_context(&execution_bytecode) {
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
                        if let runmat_hir::HirStmt::Assign(var_id, _, _) = &hir.body[0] {
                            if self.verbose {
                                debug!(
                                    "Assignment detected, var_id: {}, ends_with_semicolon: {}",
                                    var_id.0, ends_with_semicolon
                                );
                            }
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
                    } else if !is_semicolon_suppressed && result_value.is_none() {
                        result_value = results.into_iter().last();
                        if self.verbose {
                            debug!("Fallback result from interpreter: {result_value:?}");
                        }
                    }

                    if self.verbose {
                        debug!("Final result_value: {result_value:?}");
                    }
                    debug!(
                        "Interpreter execution successful, variable_array: {:?}",
                        self.variable_array
                    );
                }
                Ok(runmat_ignition::InterpreterOutcome::Pending(pending_exec)) => {
                    let plan = self.capture_execution_plan(
                        &assigned_this_execution,
                        &id_to_name,
                        &prev_assigned_snapshot,
                        &updated_functions,
                        &execution_bytecode,
                        single_assign_var,
                        single_stmt_non_assign,
                        is_expression_stmt,
                        is_semicolon_suppressed,
                        &result_value,
                        &suppressed_value,
                        &error,
                        &workspace_updates,
                        &fusion_snapshot,
                        start_time,
                        used_jit,
                        Arc::clone(&stdin_events),
                        debug_trace,
                        pending_workspace_guard.take(),
                    );
                    let console_streams = drain_console_streams();
                    let pending_result =
                        self.defer_pending_execution(plan, pending_exec, console_streams)?;
                    return Ok(pending_result);
                }
                Err(e) => {
                    debug!("Interpreter execution failed: {e}");
                    error = Some(format!("Execution failed: {e}"));
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
                    println!("mutated_names: {:?}", mutated_names);
                    println!("assigned_returned: {:?}", assigned);
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
                    println!("new_assigned: {:?}", new_assigned);
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
                                println!("workspace_update: {} -> {:?}", name, value_clone);
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
            self.function_definitions = updated_functions;
        }

        if self.verbose {
            debug!("Execution completed in {execution_time_ms}ms (JIT: {used_jit})");
        }

        // Generate type info if we have a suppressed value
        let type_info = suppressed_value.as_ref().map(format_type_info);

        // Final fallback: if not suppressed and still no value, try last non-zero variable slot
        if !is_semicolon_suppressed && result_value.is_none() {
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
        let workspace_snapshot = self.build_workspace_snapshot(workspace_updates, false);
        let figures_touched = runmat_runtime::plotting_hooks::take_recent_figures();
        let stdin_events = stdin_events
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_default();

        let warnings = runmat_runtime::warning_store::take_all();

        Ok(ExecutionResult {
            value: result_value,
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
            stdin_requested: None,
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

        let max_elements = options.max_elements.max(1).min(MATERIALIZE_DEFAULT_LIMIT);
        let preview = preview_numeric_values(&host_value, max_elements)
            .map(|(values, truncated)| WorkspacePreview { values, truncated });
        Ok(MaterializedVariable {
            name,
            class_name: matlab_class_name(&host_value),
            dtype: numeric_dtype_label(&host_value).map(|label| label.to_string()),
            shape: value_shape(&host_value).unwrap_or_else(|| vec![]),
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
    fn interpret_with_context(
        &mut self,
        bytecode: &runmat_ignition::Bytecode,
    ) -> Result<runmat_ignition::InterpreterOutcome, String> {
        runmat_ignition::interpret_with_vars(bytecode, &mut self.variable_array, Some("<repl>"))
    }

    fn capture_execution_plan(
        &self,
        assigned_this_execution: &HashSet<String>,
        id_to_name: &HashMap<usize, String>,
        prev_assigned_snapshot: &HashSet<String>,
        updated_functions: &HashMap<String, runmat_hir::HirStmt>,
        execution_bytecode: &runmat_ignition::Bytecode,
        single_assign_var: Option<usize>,
        single_stmt_non_assign: bool,
        is_expression_stmt: bool,
        is_semicolon_suppressed: bool,
        result_value: &Option<Value>,
        suppressed_value: &Option<Value>,
        error: &Option<String>,
        workspace_updates: &[WorkspaceEntry],
        fusion_snapshot: &Option<FusionPlanSnapshot>,
        start_time: Instant,
        used_jit: bool,
        stdin_events: Arc<Mutex<Vec<StdinEvent>>>,
        _debug_trace: bool,
        workspace_guard: Option<runmat_ignition::PendingWorkspaceGuard>,
    ) -> ExecutionPlan {
        ExecutionPlan {
            assigned_this_execution: assigned_this_execution.clone(),
            id_to_name: id_to_name.clone(),
            prev_assigned_snapshot: prev_assigned_snapshot.clone(),
            updated_functions: updated_functions.clone(),
            execution_bytecode: execution_bytecode.clone(),
            single_assign_var,
            single_stmt_non_assign,
            is_expression_stmt,
            is_semicolon_suppressed,
            result_value: result_value.clone(),
            suppressed_value: suppressed_value.clone(),
            error: error.clone(),
            workspace_updates: workspace_updates.to_vec(),
            fusion_snapshot: fusion_snapshot.clone(),
            start_time,
            used_jit,
            stdin_events,
            workspace_guard,
        }
    }

    fn defer_pending_execution(
        &mut self,
        plan: ExecutionPlan,
        pending: runmat_ignition::PendingExecution,
        new_streams: Vec<ExecutionStreamEntry>,
    ) -> Result<ExecutionResult> {
        let request = pending_interaction_to_request(&pending.interaction);
        let id = Uuid::new_v4();
        let elapsed_ms = plan.start_time.elapsed().as_millis() as u64;
        let fusion_plan = plan.fusion_snapshot.clone();
        let stdin_arc = Arc::clone(&plan.stdin_events);
        let used_jit = plan.used_jit;
        let frame = PendingFrame {
            plan,
            pending,
            streams: new_streams.clone(),
            pending_since: Instant::now(),
        };
        self.pending_executions.insert(id, frame);
        let stdin_events = stdin_arc
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_default();
        let pending_input = PendingInput {
            id,
            request,
            waiting_ms: 0,
        };
        Ok(ExecutionResult {
            value: None,
            execution_time_ms: elapsed_ms,
            used_jit,
            error: None,
            type_info: None,
            streams: new_streams,
            workspace: self.build_workspace_snapshot(Vec::new(), false),
            figures_touched: Vec::new(),
            warnings: Vec::new(),
            profiling: None,
            fusion_plan,
            stdin_events,
            stdin_requested: Some(pending_input),
        })
    }

    fn finalize_pending_execution(
        &mut self,
        mut plan: ExecutionPlan,
        interpreter_values: Vec<Value>,
        mut streams: Vec<ExecutionStreamEntry>,
    ) -> Result<ExecutionResult> {
        // Drop the pending workspace guard now that the interpreter has resumed.
        let _ = plan.workspace_guard.take();

        if !self.has_jit() || plan.is_expression_stmt {
            self.stats.interpreter_fallback += 1;
        }

        if let Some(var_id) = plan.single_assign_var {
            if let Some(name) = plan.id_to_name.get(&var_id) {
                plan.assigned_this_execution.insert(name.clone());
            }
            if var_id < self.variable_array.len() {
                let assignment_value = self.variable_array[var_id].clone();
                if !plan.is_semicolon_suppressed {
                    plan.result_value = Some(assignment_value);
                } else {
                    plan.suppressed_value = Some(assignment_value);
                }
            }
        } else if plan.single_stmt_non_assign
            && !plan.is_expression_stmt
            && !interpreter_values.is_empty()
            && !plan.is_semicolon_suppressed
        {
            plan.result_value = Some(interpreter_values[0].clone());
        }

        if plan.is_expression_stmt && plan.result_value.is_none() && plan.suppressed_value.is_none()
        {
            if plan.execution_bytecode.var_count > 0 {
                let temp_var_id = plan.execution_bytecode.var_count - 1;
                if temp_var_id < self.variable_array.len() {
                    let expression_value = self.variable_array[temp_var_id].clone();
                    if !plan.is_semicolon_suppressed {
                        plan.result_value = Some(expression_value);
                    } else {
                        plan.suppressed_value = Some(expression_value);
                    }
                }
            }
        } else if !plan.is_semicolon_suppressed && plan.result_value.is_none() {
            plan.result_value = interpreter_values.into_iter().last();
        }

        let execution_time = plan.start_time.elapsed();
        let execution_time_ms = execution_time.as_millis() as u64;
        self.stats.total_execution_time_ms += execution_time_ms;
        self.stats.average_execution_time_ms =
            self.stats.total_execution_time_ms as f64 / self.stats.total_executions as f64;

        if plan.error.is_none() {
            if let Some((mutated_names, assigned)) = runmat_ignition::take_updated_workspace_state()
            {
                self.variable_names = mutated_names.clone();
                let mut new_assigned: HashSet<String> = assigned
                    .difference(&plan.prev_assigned_snapshot)
                    .cloned()
                    .collect();
                new_assigned.extend(plan.assigned_this_execution.iter().cloned());
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
                for name in new_assigned {
                    let var_id = mutated_names.get(&name).copied().or_else(|| {
                        plan.id_to_name
                            .iter()
                            .find_map(|(vid, n)| if n == &name { Some(*vid) } else { None })
                    });
                    if let Some(var_id) = var_id {
                        if var_id < self.variable_array.len() {
                            let value_clone = self.variable_array[var_id].clone();
                            self.workspace_values
                                .insert(name.clone(), value_clone.clone());
                            plan.workspace_updates
                                .push(workspace_entry(&name, &value_clone));
                        }
                    }
                }
            } else {
                for name in &plan.assigned_this_execution {
                    if let Some(var_id) =
                        plan.id_to_name
                            .iter()
                            .find_map(|(vid, n)| if n == name { Some(*vid) } else { None })
                    {
                        if var_id < self.variable_array.len() {
                            let value_clone = self.variable_array[var_id].clone();
                            self.workspace_values
                                .insert(name.clone(), value_clone.clone());
                            plan.workspace_updates
                                .push(workspace_entry(name, &value_clone));
                        }
                    }
                }
            }
            self.function_definitions = plan.updated_functions.clone();
        }

        if plan.is_semicolon_suppressed && plan.suppressed_value.is_some() {
            // keep
        }
        let type_info = plan.suppressed_value.as_ref().map(format_type_info);
        if !plan.is_semicolon_suppressed && plan.result_value.is_none() {
            if let Some(v) = self
                .variable_array
                .iter()
                .rev()
                .find(|v| !matches!(v, Value::Num(0.0)))
                .cloned()
            {
                plan.result_value = Some(v);
            }
        }

        streams.extend(drain_console_streams());
        let workspace_snapshot = self.build_workspace_snapshot(plan.workspace_updates, false);
        let figures_touched = runmat_runtime::plotting_hooks::take_recent_figures();
        let stdin_events = plan
            .stdin_events
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_default();
        let warnings = runmat_runtime::warning_store::take_all();

        Ok(ExecutionResult {
            value: plan.result_value,
            execution_time_ms,
            used_jit: plan.used_jit,
            error: plan.error,
            type_info,
            streams,
            workspace: workspace_snapshot,
            figures_touched,
            warnings,
            profiling: gather_profiling(execution_time_ms),
            fusion_plan: plan.fusion_snapshot,
            stdin_events,
            stdin_requested: None,
        })
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
            println!(
                "prepare: bytecode.var_count={} required_len={} max_var_id={}",
                bytecode.var_count, required_len, max_var_id
            );
        }

        // Populate with existing values based on the variable mapping
        for (var_name, &new_var_id) in updated_var_mapping {
            if new_var_id < new_variable_array.len() {
                if let Some(value) = self.workspace_values.get(var_name) {
                    if debug_trace {
                        println!(
                            "prepare: setting {} (var_id={}) -> {:?}",
                            var_name, new_var_id, value
                        );
                    }
                    new_variable_array[new_var_id] = value.clone();
                }
            } else if debug_trace {
                println!(
                    "prepare: skipping {} (var_id={}) because len={}",
                    var_name,
                    new_var_id,
                    new_variable_array.len()
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
            } = hir_stmt
            {
                // Use the existing HIR utilities to calculate variable count
                let var_map =
                    runmat_hir::remapping::create_complete_function_var_map(params, outputs, body);
                let max_local_var = var_map.len();

                let user_func = runmat_ignition::UserFunction {
                    name: func_name.clone(),
                    params: params.clone(),
                    outputs: outputs.clone(),
                    body: body.clone(),
                    local_var_count: max_local_var,
                    has_varargin: false,
                    has_varargout: false,
                    var_types: vec![Type::Unknown; max_local_var],
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
        println!("RunMat Session Status");
        println!("==========================");
        println!();

        println!(
            "JIT Compiler: {}",
            if self.has_jit() {
                "Available"
            } else {
                "Disabled/Failed"
            }
        );
        println!("Verbose Mode: {}", self.verbose);
        println!();

        println!("Execution Statistics:");
        println!("  Total Executions: {}", self.stats.total_executions);
        println!("  JIT Compiled: {}", self.stats.jit_compiled);
        println!("  Interpreter Used: {}", self.stats.interpreter_fallback);
        println!(
            "  Average Time: {:.2}ms",
            self.stats.average_execution_time_ms
        );
        println!();

        let gc_stats = self.gc_stats();
        println!("Garbage Collector:");
        println!(
            "  Total Allocations: {}",
            gc_stats
                .total_allocations
                .load(std::sync::atomic::Ordering::Relaxed)
        );
        println!(
            "  Minor Collections: {}",
            gc_stats
                .minor_collections
                .load(std::sync::atomic::Ordering::Relaxed)
        );
        println!(
            "  Major Collections: {}",
            gc_stats
                .major_collections
                .load(std::sync::atomic::Ordering::Relaxed)
        );
        println!(
            "  Current Memory: {:.2} MB",
            gc_stats
                .current_memory_usage
                .load(std::sync::atomic::Ordering::Relaxed) as f64
                / 1024.0
                / 1024.0
        );
        println!();
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
        shape: value_shape(value).unwrap_or_else(|| vec![]),
        is_gpu: matches!(value, Value::GpuTensor(_)),
        size_bytes: approximate_size_bytes(value),
        preview,
        residency,
        preview_token: None,
    }
}

fn drain_console_streams() -> Vec<ExecutionStreamEntry> {
    runmat_runtime::console::take_thread_buffer()
        .into_iter()
        .map(|entry| ExecutionStreamEntry {
            stream: match entry.stream {
                runmat_runtime::console::ConsoleStream::Stdout => ExecutionStreamKind::Stdout,
                runmat_runtime::console::ConsoleStream::Stderr => ExecutionStreamKind::Stderr,
            },
            text: entry.text,
            timestamp_ms: entry.timestamp_ms,
        })
        .collect()
}

fn pending_interaction_to_request(
    interaction: &runmat_runtime::interaction::PendingInteraction,
) -> InputRequest {
    let kind = match interaction.kind {
        runmat_runtime::interaction::InteractionKind::Line { echo } => {
            InputRequestKind::Line { echo }
        }
        runmat_runtime::interaction::InteractionKind::KeyPress => InputRequestKind::KeyPress,
    };
    InputRequest {
        prompt: interaction.prompt.clone(),
        kind,
    }
}

fn map_input_response(response: InputResponse) -> runmat_runtime::interaction::InteractionResponse {
    match response {
        InputResponse::Line(value) => runmat_runtime::interaction::InteractionResponse::Line(value),
        InputResponse::KeyPress => runmat_runtime::interaction::InteractionResponse::KeyPress,
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
pub fn execute_and_format(input: &str) -> String {
    match RunMatSession::new() {
        Ok(mut engine) => match engine.execute(input) {
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
