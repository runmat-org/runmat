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

pub type ExecutionStreamKind = runmat_runtime::console::ConsoleStream;
pub type ExecutionStreamEntry = runmat_runtime::console::ConsoleEntry;

#[derive(Debug, Clone, Default)]
pub struct ExecutionProfiling {
    /// Read-only sealed command facts, not a claim of global route coverage.
    pub command_evidence: Option<CommandExecutionEvidence>,
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

/// Core's original admission-to-seal projection. Counts concern only admitted
/// work; even a pending-free command does not establish all-route coverage.
#[derive(Debug, Clone, Copy)]
pub struct CommandExecutionEvidence {
    pub physical_submissions: Option<runmat_execution::profiling::PhysicalSubmissionCounts>,
    pub deferred_work: Option<runmat_execution::profiling::DeferredWorkCounts>,
    pub route_coverage: runmat_execution::profiling::CommandRouteCoverage,
    pub canonical_device_route_admitted: bool,
    pub unproven_device_route: bool,
    pub unproven_device_time: bool,
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
    pub streams: runmat_runtime::console::CapturedConsoleEntries,
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
