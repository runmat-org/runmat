use anyhow::Result;
use runmat_gc::{gc_configure, gc_stats, GcConfig};
use runmat_value::Value;
use tracing::{debug, info, info_span};

use runmat_hir::{LoweringContext, LoweringResult, SourceId};
use runmat_lexer::{tokenize_detailed, Token as LexToken};
use runmat_parser::{parse_with_options, ParserOptions};
use runmat_runtime::{build_runtime_error, gather_if_needed_async, RuntimeError};
use runmat_runtime::{
    runtime_export_workspace_state, runtime_import_workspace_state, WorkspaceReplayMode,
};
use runmat_time::Instant;
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

use crate::execution::{
    ExecutionStats, ExecutionStreamEntry, ExecutionStreamKind, InputRequest, InputRequestKind,
    InputResponse, SharedAsyncInputHandler, StdinEvent, StdinEventKind,
};
use crate::fusion::{build_fusion_snapshot, FusionPlanSnapshot};
use crate::profiling::{gather_profiling, reset_provider_telemetry};
use crate::source_pool::{line_col_from_offset, SourcePool};
use crate::telemetry::{TelemetryPlatformInfo, TelemetrySink};
use crate::workspace::{
    determine_display_label_from_context, execution_display_context, format_type_info,
    gather_gpu_preview_values, gpu_dtype_label, gpu_size_bytes, last_emit_var_index,
    last_store_var_index, slice_value_for_preview, workspace_entry, FinalStmtEmitDisposition,
    MaterializedVariable, WorkspaceEntry, WorkspaceExportMode, WorkspaceMaterializeOptions,
    WorkspaceMaterializeTarget, WorkspacePreview, WorkspaceResidency, WorkspaceSnapshot,
    MATERIALIZE_DEFAULT_LIMIT,
};
use crate::{
    approximate_size_bytes, matlab_class_name, numeric_dtype_label, preview_numeric_values,
    value_shape, CompatMode, RunError,
};

fn runtime_language_mode(mode: CompatMode) -> runmat_runtime::context::RuntimeLanguageMode {
    match mode {
        CompatMode::Matlab => runmat_runtime::context::RuntimeLanguageMode::Matlab,
        CompatMode::RunMat => runmat_runtime::context::RuntimeLanguageMode::RunMat,
        CompatMode::Strict => runmat_runtime::context::RuntimeLanguageMode::Strict,
    }
}

mod compile;
pub(crate) mod semantic_definition;
mod cancellation;
mod config;
mod dynamic;
mod executable;
mod host_services;
mod input;
mod application_settings;
#[cfg(not(target_arch = "wasm32"))]
mod host_driver;
mod http_host;
#[cfg(not(target_arch = "wasm32"))]
mod process_host;
#[cfg(not(target_arch = "wasm32"))]
mod tcp_host;
mod init;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod local_tasks_tests;
mod project;
mod run;
mod user_path;
mod workspace;

/// Host-agnostic RunMat execution session (parser + interpreter + optional JIT).
#[cfg(not(target_arch = "wasm32"))]
type SessionLocalTaskSupervisor = runmat_runtime::context::RuntimeLocalTaskSupervisor;
#[cfg(target_arch = "wasm32")]
type SessionLocalTaskSupervisor = runmat_runtime::context::BrowserLocalTaskSupervisor;

pub struct RunMatSession {
    /// Verbose output for debugging
    verbose: bool,
    /// Execution statistics
    stats: ExecutionStats,
    /// Current workspace bindings with stable ABI identity and current VM slot.
    workspace_bindings: HashMap<String, SessionWorkspaceBinding>,
    /// Canonical, rooted base-workspace state for this session. Executors use
    /// invocation-local slot projections and publish ordered mutations here.
    base_workspace: runmat_runtime::workspace::BaseWorkspaceOwner,
    /// Stable ABI identity for this interactive workspace.
    abi_workspace_handle: crate::abi::WorkspaceHandle,
    /// Source identity for the active execution request (if source-scoped).
    active_source_identity: Option<crate::abi::SourceIdentity>,
    /// Semantic function registry persisted across interactive inputs.
    function_registry: runmat_vm::FunctionRegistry,
    next_semantic_function_id: usize,
    /// Canonically compiled functions discovered through the runtime search path.
    dynamic_function_cache: Arc<Mutex<HashMap<PathBuf, DynamicFunctionCacheEntry>>>,
    /// Startup path facts are captured before an async application-settings
    /// read; no later ambient provider/CWD can redefine this session owner.
    startup_user_path_authority:
        runmat_runtime::builtins::common::path_state::CapturedUserPathAuthority,
    startup_environment_path: String,
    user_path_hydrated: bool,
    /// Loaded C MEX modules and their persistent state, scoped to this session.
    #[cfg(not(target_arch = "wasm32"))]
    mex_runtime: std::rc::Rc<runmat_runtime::foreign::MexRuntimeSession>,
    #[cfg(not(target_arch = "wasm32"))]
    foreign_runtime: std::rc::Rc<runmat_runtime::foreign::ForeignRuntime>,
    #[cfg(not(target_arch = "wasm32"))]
    native_ffi_adapter: std::rc::Rc<runmat_runtime::foreign::NativeFfiAdapter>,
    #[cfg(not(target_arch = "wasm32"))]
    java_adapter: std::rc::Rc<runmat_runtime::foreign::JavaAdapter>,
    #[cfg(not(target_arch = "wasm32"))]
    python_adapter: std::rc::Rc<runmat_runtime::foreign::PythonAdapter>,
    /// Session-scoped production native entry publication and invalidation.
    #[cfg(not(target_arch = "wasm32"))]
    generic_native_cache: crate::generic_native::GenericNativeCache,
    #[cfg(not(target_arch = "wasm32"))]
    native_tiering_enabled: bool,
    /// Optional host-frozen project snapshot used by every compilation in this session.
    project_handoff: Option<runmat_package::FrozenProjectHandoff>,
    /// Interned source pool for user-defined functions
    source_pool: SourcePool,
    /// Publishes one immutable cancellation generation per admitted execution.
    cancellation: cancellation::SessionCancellation,
    /// Root-scoped execution authority inherited by every invocation.
    runtime_context: runmat_runtime::context::RuntimeContext,
    /// Strong local executor owner. RuntimeContext holds only its weak port.
    local_tasks: std::rc::Rc<SessionLocalTaskSupervisor>,
    /// Typed handle retained alongside the executor-neutral runtime port so a
    /// host may explicitly persist or restore this session's bounded profile.
    placement_session: std::rc::Rc<runmat_accelerate::placement::PlacementSession>,
    /// Tracks whether an execution is currently active.
    is_executing: bool,
    /// Optional async input handler (Phase 2). When set, stdin interactions are awaited
    /// internally by `ExecuteFuture` rather than being surfaced as "pending requests".
    async_input_handler: Option<SharedAsyncInputHandler>,
    /// Maximum number of call stack frames to retain for diagnostics.
    callstack_limit: usize,
    /// Namespace prefix for runtime/semantic error identifiers.
    error_namespace: String,
    /// Active source name for diagnostics (set from execution requests).
    active_source_name: String,
    /// Resolved active source path for file-aware builtins and source discovery.
    active_source_fullpath_name: Option<String>,
    pub(crate) telemetry_consent: bool,
    pub(crate) telemetry_client_id: Option<String>,
    pub(crate) telemetry_platform: TelemetryPlatformInfo,
    pub(crate) telemetry_sink: Option<Arc<dyn TelemetrySink>>,
    workspace_preview_tokens: HashMap<Uuid, WorkspaceMaterializeTicket>,
    emit_fusion_plan: bool,
    compat_mode: CompatMode,
    top_level_await_enabled: bool,
    dynamic_eval_enabled: bool,
    /// Persisted numeric display format for this session (survives across executions).
    format_mode: runmat_value::FormatMode,
    /// Preloaded companion statements discovered asynchronously by the request path.
    pending_companion_source_discovery: Option<compile::CompanionSourceDiscovery>,
}

impl RunMatSession {
    /// Obtain the host-owned local driver before borrowing this session for an
    /// execution. `execute_request` and `invoke_executable` drive callbacks
    /// during foreground work. Library hosts must additionally keep
    /// `drive_until(next_input_or_event)` polled between requests for idle
    /// callbacks to make progress. Dropping the session cancels local jobs.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn local_task_driver(
        &self,
    ) -> std::rc::Rc<runmat_runtime::context::RuntimeLocalTaskSupervisor> {
        self.local_tasks.clone()
    }

    /// Browser host disposal retains this exact owner while the session is
    /// temporarily moved out for an asynchronous JavaScript request.
    #[cfg(target_arch = "wasm32")]
    pub fn browser_local_task_supervisor(
        &self,
    ) -> std::rc::Rc<runmat_runtime::context::BrowserLocalTaskSupervisor> {
        self.local_tasks.clone()
    }

    /// Stop and release local callback jobs before host disposal. Idempotent;
    /// subsequent scheduling reports a stable closed-session error.
    pub fn shutdown_local_tasks(&self) {
        self.local_tasks.shutdown();
    }

    /// Drive one callback from session-owned native work. Interactive hosts
    /// poll this future alongside their next command or input event.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn service_background_once(&self) -> bool {
        self.mex_runtime.service_background_once().await
    }
}

pub(crate) struct PreparedExecution {
    ast: runmat_parser::Program,
    lowering: LoweringResult,
    mir: runmat_mir::MirAssembly,
    analysis: runmat_mir::analysis::AnalysisStore,
    pub(crate) bytecode: runmat_vm::Bytecode,
    function_registry_after_success: runmat_vm::FunctionRegistry,
    current_function_ids: std::collections::HashSet<runmat_hir::FunctionId>,
    next_semantic_function_id_after_success: usize,
}

impl PreparedExecution {
    #[cfg(test)]
    pub(crate) fn lowering(&self) -> &LoweringResult {
        &self.lowering
    }

    #[cfg(test)]
    pub(crate) fn analysis(&self) -> &runmat_mir::analysis::AnalysisStore {
        &self.analysis
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SessionWorkspaceBinding {
    pub(crate) key: crate::abi::WorkspaceBindingKey,
    pub(crate) slot: usize,
}

#[derive(Debug, Clone)]
struct WorkspaceMaterializeTicket {
    name: String,
}

#[derive(Clone)]
struct DynamicFunctionCacheEntry {
    source_text: String,
    project_revision: Option<runmat_package::ProjectRevision>,
    registry: Arc<runmat_vm::FunctionRegistry>,
}

#[cfg(test)]
impl RunMatSession {
    pub(crate) fn cached_source_count(&self) -> usize {
        self.dynamic_function_cache
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .len()
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

impl Drop for RunMatSession {
    fn drop(&mut self) {
        if let Err(error) = self.base_workspace.close() {
            tracing::warn!(%error, "could not revoke base workspace during session shutdown");
        }
        self.local_tasks.shutdown();
        #[cfg(not(target_arch = "wasm32"))]
        if let Err(error) = self.mex_runtime.shutdown(self.runtime_context.clone()) {
            tracing::warn!(%error, "could not finish MEX lifecycle during session shutdown");
        }
        self.runtime_context
            .execution()
            .drain_scope(runmat_execution::CancellationReason::Shutdown);
    }
}

impl Default for RunMatSession {
    fn default() -> Self {
        Self::new().expect("Failed to create default RunMat session")
    }
}
