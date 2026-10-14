use runmat_runtime::warning_store::RuntimeWarning;
use runmat_runtime::RuntimeError;
use runmat_value::Value;

use crate::fusion::FusionPlanSnapshot;
use crate::workspace::WorkspaceSnapshot;

#[derive(Debug, Default)]
pub struct ExecutionStats {
    pub total_executions: usize,
    pub jit_compiled: usize,
    pub interpreter_fallback: usize,
    pub native_osr_transfers: usize,
    pub vectorized_native_regions: u64,
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
    /// Outer command wall time, including admitted cleanup. This can exceed
    /// ExecutionOutcome::execution_time_ms, which measures the inner evaluator.
    pub total_ms: u64,
    /// Attributed CPU duration, when measured. Not derived by subtracting GPU
    /// time from wall duration, since execution and waiting can overlap.
    pub cpu_ms: Option<u64>,
    /// Attributed device execution duration, when measured. Provider-wide host
    /// wall counters are not device execution measurements.
    pub gpu_ms: Option<u64>,
    /// Exact attributed kernel launches, when observed; not submission counts
    /// or the length of a bounded provider diagnostic log.
    pub kernel_count: Option<u32>,
}

#[derive(Debug)]
pub struct SessionExecutionResult {
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

pub type RootedSessionExecutionResult =
    runmat_runtime::execution::RootedResult<SessionExecutionResult>;

impl runmat_gc::Trace for SessionExecutionResult {
    fn trace(&self, tracer: &mut dyn runmat_gc::Tracer) {
        if let Some(value) = &self.value {
            runmat_gc::Trace::trace(value, tracer);
        }
    }
}
