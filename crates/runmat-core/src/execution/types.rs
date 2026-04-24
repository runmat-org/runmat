use runmat_builtins::Value;
use runmat_runtime::warning_store::RuntimeWarning;
use runmat_runtime::RuntimeError;

use crate::fusion::FusionPlanSnapshot;
use crate::workspace::WorkspaceSnapshot;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionStreamKind {
    Stdout,
    Stderr,
    ClearScreen,
}

#[derive(Debug, Clone)]
pub struct ExecutionStreamEntry {
    pub stream: ExecutionStreamKind,
    pub text: String,
    pub timestamp_ms: u64,
}

#[derive(Debug, Clone, Default)]
pub struct ExecutionProfiling {
    pub total_ms: u64,
    pub cpu_ms: Option<u64>,
    pub gpu_ms: Option<u64>,
    pub kernel_count: Option<u32>,
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
