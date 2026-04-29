use anyhow::Result;
use runmat_builtins::{self, Type, Value};
use runmat_gc::{gc_configure, gc_stats, GcConfig};
use tracing::{debug, info, info_span, warn};

use runmat_hir::{LoweringContext, LoweringResult, SourceId};
use runmat_lexer::{tokenize_detailed, Token as LexToken};
use runmat_parser::{parse_with_options, ParserOptions};
use runmat_runtime::{build_runtime_error, gather_if_needed_async, RuntimeError};
use runmat_runtime::{
    runtime_export_workspace_state, runtime_import_workspace_state, WorkspaceReplayMode,
};
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
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use uuid::Uuid;

use crate::execution::{
    ExecutionResult, ExecutionStats, ExecutionStreamEntry, ExecutionStreamKind, InputRequest,
    InputRequestKind, InputResponse, SharedAsyncInputHandler, StdinEvent, StdinEventKind,
};
use crate::fusion::{build_fusion_snapshot, FusionPlanSnapshot};
use crate::profiling::{gather_profiling, reset_provider_telemetry};
use crate::source_pool::{line_col_from_offset, SourcePool};
use crate::telemetry::{TelemetryPlatformInfo, TelemetrySink};
use crate::workspace::{
    determine_display_label_from_context, format_type_info, gather_gpu_preview_values,
    gpu_dtype_label, gpu_size_bytes, last_displayable_statement_emit_disposition,
    last_emit_var_index, last_expr_emits_value, last_unsuppressed_assign_var,
    slice_value_for_preview, workspace_entry, FinalStmtEmitDisposition, MaterializedVariable,
    WorkspaceEntry, WorkspaceExportMode, WorkspaceMaterializeOptions, WorkspaceMaterializeTarget,
    WorkspacePreview, WorkspaceResidency, WorkspaceSnapshot, MATERIALIZE_DEFAULT_LIMIT,
};
use crate::{
    approximate_size_bytes, matlab_class_name, numeric_dtype_label, preview_numeric_values,
    value_shape, CompatMode, RunError,
};

mod compile;
mod config;
mod run;
mod snapshot;
mod workspace;

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
    /// Optional async input handler (Phase 2). When set, stdin interactions are awaited
    /// internally by `ExecuteFuture` rather than being surfaced as "pending requests".
    async_input_handler: Option<SharedAsyncInputHandler>,
    /// Maximum number of call stack frames to retain for diagnostics.
    callstack_limit: usize,
    /// Namespace prefix for runtime/semantic error identifiers.
    error_namespace: String,
    /// Default source name used for diagnostics.
    default_source_name: String,
    /// Override source name for the current execution.
    source_name_override: Option<String>,
    pub(crate) telemetry_consent: bool,
    pub(crate) telemetry_client_id: Option<String>,
    pub(crate) telemetry_platform: TelemetryPlatformInfo,
    pub(crate) telemetry_sink: Option<Arc<dyn TelemetrySink>>,
    workspace_preview_tokens: HashMap<Uuid, WorkspaceMaterializeTicket>,
    workspace_version: u64,
    emit_fusion_plan: bool,
    compat_mode: CompatMode,
    /// Persisted numeric display format for this session (survives across executions).
    format_mode: runmat_builtins::FormatMode,
}

pub(crate) struct PreparedExecution {
    ast: runmat_parser::Program,
    lowering: LoweringResult,
    bytecode: runmat_vm::Bytecode,
}

#[derive(Debug, Clone)]
struct WorkspaceMaterializeTicket {
    name: String,
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
