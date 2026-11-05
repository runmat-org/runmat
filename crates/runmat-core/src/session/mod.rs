use anyhow::Result;
use runmat_gc::{gc_configure, gc_stats, GcConfig};
use runmat_value::Value;
use tracing::{debug, info, info_span};

use runmat_hir::{LoweringContext, LoweringResult};
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
use std::rc::Rc;
use std::sync::Arc;
use uuid::Uuid;

use crate::execution::{
    ExecutionStats, InputRequest, InputRequestKind, InputResponse, SharedAsyncInputHandler,
    StdinEvent, StdinEventKind,
};
use crate::fusion::{build_fusion_snapshot, FusionPlanSnapshot};
use crate::profiling::execution_profiling;
use crate::source_pool::SourcePool;
use crate::telemetry::{TelemetryPlatformInfo, TelemetrySink};
use crate::workspace::{
    execution_display_context, format_type_info, gather_gpu_preview_values, gpu_dtype_label,
    gpu_size_bytes, last_emit_var_index, last_store_var_index, slice_value_for_preview,
    workspace_entry, FinalStmtEmitDisposition, MaterializedVariable, WorkspaceEntry,
    WorkspaceExportMode, WorkspaceMaterializeOptions, WorkspaceMaterializeTarget, WorkspacePreview,
    WorkspaceResidency, WorkspaceSnapshot, MATERIALIZE_DEFAULT_LIMIT,
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

mod application_settings;
mod cancellation;
mod command_profiling;
mod compile;
pub(crate) use compile::program_sources::SourceCompileProjection;
mod config;
mod settings;
mod dynamic;
mod dynamic_cache;
mod error_conversion;
mod executable;
mod fea_source;
mod fea_duration;
pub use fea_duration::{
    FeaExecutionTimingKey, FeaExecutionTimingObservation, FeaExecutionTimingProfile, FeaTimingRoute,
};
mod fea_preflight;
pub use fea_preflight::FeaExecutionPreflight;
mod fea_execution;
pub use fea_execution::{
    FeaGeometryExecutionHost, FeaStudyPreparationError, PreparedSerialFeaStudy,
    PreparedSerialFeaExecutionError,
};
pub use fea_source::CheckedFeaSessionSource;
mod fea_changes;
mod fea_publication;
mod fea_workspace;
pub use fea_publication::PreparedFeaWorkspacePublication;
mod fea_working;
pub use fea_working::{FeaWorkingCheckRequest, FeaWorkingSource};
mod fea_queries;
pub use fea_workspace::{
    FeaCapturedDependency, FeaWorkspaceCapture, FeaWorkspaceCheckRequest, FeaWorkspaceCheckResult,
};
#[cfg(not(target_arch = "wasm32"))]
mod host_driver;
mod host_services;
pub use host_services::SessionGraphicsHost;
mod http_host;
mod init;
mod pinned;
mod portable;
mod program;
mod input;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod local_tasks_tests;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod mex_close_tests;
mod private_source_owner;
#[cfg(not(target_arch = "wasm32"))]
mod process_host;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod profiling_tests;
mod project;
mod retirement;
mod run;
#[cfg(test)]
mod s1_compiled_call_tests;
pub(crate) mod semantic_definition;
mod source_evaluation;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use source_evaluation::bind_function_resolvers;
pub(crate) use source_evaluation::SelectedStaticTarget;
mod source_scope;
pub use source_evaluation::SelectedNativeSourceEvaluator;
#[cfg(not(target_arch = "wasm32"))]
mod tcp_host;
mod user_path;
mod workspace;

pub use retirement::SessionRetirement;

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
    workspace_bindings: workspace::binding_control::BindingControl,
    /// Canonical, rooted base-workspace state for this session. Executors use
    /// invocation-local slot projections and publish ordered mutations here.
    base_workspace: runmat_runtime::workspace::BaseWorkspaceOwner,
    /// Stable ABI identity for this interactive workspace.
    abi_workspace_handle: crate::abi::WorkspaceHandle,
    /// Source identity for the active execution request (if source-scoped).
    active_source_identity: Option<crate::abi::SourceIdentity>,
    /// One explicitly read, non-indexed project primary for this request only.
    active_primary_source: Option<crate::executable::CapturedPrimarySource>,
    /// Semantic function registry persisted across interactive inputs.
    function_registry: runmat_vm::FunctionRegistry,
    next_semantic_function_id: usize,
    /// Canonically compiled functions discovered through the runtime search path.
    dynamic_function_cache: Rc<dynamic_cache::DynamicFunctionCache>,
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
    checked_fea_sources: fea_queries::CheckedSourceCache,
    captured_project: Option<pinned::CapturedProject>,
    pinned_inputs: Option<(
        runmat_filesystem::pinned::InputManifest,
        std::sync::Arc<runmat_filesystem::pinned::PinnedFsProvider>,
    )>,
    /// Immutable foreign bytes whose rebased paths are published by the project handoff.
    #[cfg(not(target_arch = "wasm32"))]
    private_project_root: Option<crate::foreign_artifacts::PrivateArtifactRoot>,
    #[cfg(not(target_arch = "wasm32"))]
    prepared_project_foreign:
        Option<std::sync::Arc<crate::foreign_artifacts::PreparedProjectForeignArtifacts>>,
    #[cfg(not(target_arch = "wasm32"))]
    prepared_selected_foreign_input:
        Option<std::sync::Arc<crate::foreign_artifacts::PreparedSelectedForeignInput>>,
    #[cfg(not(target_arch = "wasm32"))]
    selected_foreign_config: Option<std::sync::Arc<runmat_config::runtime::RunMatRuntimeConfig>>,
    /// Interned source pool for user-defined functions
    source_pool: SourcePool,
    /// Session-owned compiler/executor entry for asynchronous source callbacks.
    source_evaluator: std::rc::Rc<source_evaluation::SessionSourceEvaluator>,
    /// Publishes one immutable cancellation generation per admitted execution.
    cancellation: cancellation::SessionCancellation,
    /// Root-scoped execution authority inherited by every invocation.
    runtime_context: runmat_runtime::context::RuntimeContext,
    program_service: std::rc::Rc<program::SessionProgramService>,
    /// Exact Core-installed effect ports. Workspace-only invocation overlays
    /// may change without conferring a new call/acceleration/foreign owner.
    canonical_device_route_ports: runmat_runtime::context::RuntimeServicePorts,
    /// Only a test child created from an active caller borrows its original
    /// command. Ordinary session entrypoints create their own seal authority.
    inherited_command_profiling: Option<runmat_execution::profiling::CommandProfilingReceipt>,
    /// Strong callback owner stays with the session; its presentation service
    /// holds only a weak reference, so retained selected contexts cannot cycle.
    graphics_callbacks: Option<runmat_runtime::context::RuntimeGraphicsCallbackQueue>,
    /// Strong local executor owner. RuntimeContext holds only its weak port.
    local_tasks: std::rc::Rc<SessionLocalTaskSupervisor>,
    /// Non-cloneable cleanup owner; RuntimeContext exposes only its weak port.
    lifecycle_host: runmat_runtime::context::RuntimeLifecycleHost,
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
        self.retire_graphics_host();
        #[cfg(not(target_arch = "wasm32"))]
        if self.mex_runtime.has_unretired_work() {
            self.mex_runtime.force_abandon();
            tracing::warn!(
                "abandoned MEX lifecycle before forced local-task shutdown; mexAtExit completion is not guaranteed"
            );
        }
        self.lifecycle_host.force_close();
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
    portable_source_map: Option<crate::ExecutableSourceMap>,
    source_projection: compile::program_sources::SourceCompileProjection,
    function_registry_after_success: runmat_vm::FunctionRegistry,
    published_registry_for_reuse: runmat_vm::FunctionRegistry,
    private_companion_function_names: HashSet<String>,
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

#[derive(Debug)]
pub(crate) struct SessionWorkspaceBinding {
    pub(crate) key: crate::abi::WorkspaceBindingKey,
    pub(crate) slot: usize,
    _key_claim: Option<runmat_value::ExactHostAllocationClaim>,
}

impl SessionWorkspaceBinding {
    #[cfg(test)]
    fn unbounded(key: crate::abi::WorkspaceBindingKey, slot: usize) -> Self {
        Self {
            key,
            slot,
            _key_claim: None,
        }
    }

    fn admitted(key: workspace::binding_key::RetainedBindingKey, slot: usize) -> Self {
        let (key, claim) = key.into_parts();
        Self {
            key,
            slot,
            _key_claim: claim,
        }
    }

    fn replace_key(&mut self, key: workspace::binding_key::RetainedBindingKey) {
        let (key, claim) = key.into_parts();
        let old_key = std::mem::replace(&mut self.key, key);
        let old_claim = std::mem::replace(&mut self._key_claim, claim);
        drop(old_key);
        drop(old_claim);
    }
}

#[derive(Debug, Clone)]
struct WorkspaceMaterializeTicket {
    name: String,
}

#[derive(Clone)]
struct DynamicFunctionCacheEntry {
    selection: DynamicFunctionCacheSelection,
    unit_bytes: Arc<[u8]>,
    admitted: Rc<runmat_vm::AdmittedExecutableUnit>,
    source_records: Arc<[runmat_runtime::source_context::ProgramSourceRecord]>,
    entrypoint_origin: runmat_execution::FunctionBindingOrigin,
    binding: runmat_value::BoundFunctionBinding,
    super_constructors: Vec<runmat_runtime::context::RetainedSuperConstructorBinding>,
    super_dependency_origins: HashSet<DynamicDependencyIdentity>,
    dependencies: Arc<[DynamicFunctionDependency]>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum DynamicDependencyIdentity {
    Callable(runmat_execution::FunctionBindingOrigin),
    DeclarationOnly {
        class: runmat_types::ClassIdentity,
        program: runmat_execution::Digest,
    },
}

#[derive(Clone)]
struct DynamicFunctionDependency {
    access_path: PathBuf,
    content_revision: Option<runmat_execution::Digest>,
    uses_pcode_reader: bool,
}

#[derive(Clone)]
struct DynamicFunctionCacheSelection {
    primary_revision: runmat_execution::Digest,
    inventory_revision: runmat_execution::Digest,
    path_origin: runmat_runtime::builtins::common::path_search::FileCandidateOrigin,
}

impl DynamicFunctionCacheSelection {
    fn matches_selection(
        &self,
        primary_revision: &runmat_execution::Digest,
        inventory_revision: &runmat_execution::Digest,
        origin: &runmat_runtime::builtins::common::path_search::FileCandidateOrigin,
    ) -> bool {
        &self.primary_revision == primary_revision
            && &self.inventory_revision == inventory_revision
            && &self.path_origin == origin
    }
}

#[cfg(test)]
mod dynamic_source_cache_tests {
    use super::*;
    use runmat_runtime::builtins::common::path_search::FileCandidateOrigin;

    #[test]
    fn physical_path_cache_does_not_reuse_another_selected_search_authority() {
        let selected = FileCandidateOrigin::SearchRoot {
            ordinal: 0,
            access_root: PathBuf::from("/workspace"),
            relative: PathBuf::from("same.m"),
        };
        let entry = DynamicFunctionCacheSelection {
            primary_revision: runmat_execution::Digest::sha256("function y = same(); y = 1; end"),
            inventory_revision: runmat_execution::Digest::sha256("inventory"),
            path_origin: selected.clone(),
        };
        assert!(entry.matches_selection(
            &entry.primary_revision,
            &entry.inventory_revision,
            &selected
        ));
        assert!(!entry.matches_selection(
            &entry.primary_revision,
            &entry.inventory_revision,
            &FileCandidateOrigin::SearchRoot {
                ordinal: 1,
                access_root: PathBuf::from("/workspace"),
                relative: PathBuf::from("same.m"),
            },
        ));
        assert!(!entry.matches_selection(
            &entry.primary_revision,
            &entry.inventory_revision,
            &FileCandidateOrigin::Direct {
                spelling: PathBuf::from("/workspace/same.m"),
            },
        ));
        assert!(!entry.matches_selection(
            &entry.primary_revision,
            &runmat_execution::Digest::sha256("changed inventory"),
            &selected,
        ));
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "dynamic_source_product_tests.rs"]
mod dynamic_source_product_tests;

#[cfg(test)]
impl RunMatSession {
    pub(crate) fn cached_source_count(&self) -> usize {
        self.dynamic_function_cache.len()
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
        self.workspace_bindings.close();
        self.retire_graphics_host();
        if let Some(callbacks) = self.graphics_callbacks.take() {
            callbacks.close();
        }
        // Forced disposal has no native completion acknowledgement. Revoke
        // MEX callback authority explicitly before closing its dependent
        // local-task driver; a service/context/session Rc cycle means waiting
        // for NativeMexLane::drop would be too late.
        #[cfg(not(target_arch = "wasm32"))]
        let abandoned_mex = self.mex_runtime.has_unretired_work();
        #[cfg(not(target_arch = "wasm32"))]
        if abandoned_mex {
            if let Err(error) = self.mex_runtime.persist_installed_bytes_unverified(
                "MEX host retirement unverified at Session drop",
            ) {
                tracing::warn!(%error, "could not persist MEX private bytes after forced disposal");
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        self.mex_runtime.force_abandon();
        self.lifecycle_host.force_close();
        #[cfg(not(target_arch = "wasm32"))]
        if abandoned_mex {
            tracing::warn!(
                "abandoned MEX lifecycle during forced session disposal; mexAtExit completion is not guaranteed"
            );
        }
        if let Err(error) = self.base_workspace.close() {
            tracing::warn!(%error, "could not revoke base workspace during session shutdown");
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
