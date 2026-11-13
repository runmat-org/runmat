use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use runmat_types::SourceId;
use runmat_value::Value;

use crate::callsite::CallsiteInfo;
use crate::source_context::SourceInfo;

#[derive(Default)]
pub(crate) struct SessionResourceCleanup(pub(crate) RefCell<Vec<Box<dyn FnOnce()>>>);

impl std::fmt::Debug for SessionResourceCleanup {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SessionResourceCleanup")
            .field("registrations", &self.0.borrow().len())
            .finish()
    }
}

/// Mutable state owned by exactly one runtime session.
///
/// Fields are split by semantic responsibility even though they share one
/// allocation. The legacy bridge is the only module allowed to manipulate the
/// complete state; new code uses the focused accessors on [`RuntimeContext`].
#[derive(Debug)]
pub struct RuntimeContextState {
    pub(crate) local_identity: super::RuntimeContextLocalId,
    pub(crate) gpu_device_selection: Cell<super::gpu_device::GpuDeviceSelection>,
    pub(crate) gpu_device_epoch: Cell<u64>,
    pub(crate) gpu_device_reset: super::gpu_device::GpuDeviceResetState,
    /// Non-serialized liveness authority shared by invocation generations of
    /// this session, but never by a parallel-lab fork or a new context.
    pub(crate) session_lease: Arc<()>,
    /// Session-local generation for staged host-effect calls. A shared host
    /// allocation ledger may be installed in multiple sessions; replacing
    /// this session's service cannot revoke their common global owner.
    pub(crate) host_effect_generation: Arc<AtomicU64>,
    pub(crate) resource_cleanup: SessionResourceCleanup,
    #[cfg(target_arch = "wasm32")]
    pub(crate) worker_invocation_resources: Cell<bool>,
    /// A lifecycle host is the strong owner. Cloneable runtime contexts see
    /// only its weak registry, so callback tasks cannot own their host.
    pub(crate) cleanup_registry: RefCell<
        Option<
            std::rc::Weak<
                crate::builtins::introspection::on_cleanup::lifecycle::registry::CleanupRegistry,
            >,
        >,
    >,
    pub(crate) source: RefCell<SourceState>,
    pub(crate) call: RefCell<CallState>,
    pub(crate) workspace: RefCell<Option<crate::workspace::WorkspaceResolver>>,
    pub(crate) session_variables: RefCell<crate::workspace::session::SessionVariableState>,
    /// Inline reservation for one prepared session-global mutation. It sits
    /// outside the RefCell so cancellation/drop can release it during reads.
    pub(crate) session_global_mutation: crate::workspace::session::SessionMutationGate,
    pub(crate) warnings: RefCell<Vec<crate::warning_store::RuntimeWarning>>,
    pub(crate) warning_policy: RefCell<crate::warnings::WarningPolicy>,
    pub(crate) timing: RefCell<super::timing::TimingSessionState>,
    pub(crate) timers: RefCell<super::timer::TimerSessionState>,
    pub(crate) console: RefCell<crate::console::ConsoleState>,
    pub(crate) display_format: runmat_value::DisplayFormatOwner,
    pub(crate) classes: RefCell<crate::class_registry::RuntimeClassState>,
    pub(crate) interaction: RefCell<crate::interaction::InteractionState>,
    pub(crate) test_services: RefCell<Vec<crate::testing::RuntimeTestServices>>,
    pub(crate) test_contexts: RefCell<Vec<std::rc::Rc<RefCell<crate::testing::ContextState>>>>,
    pub(crate) constructor_receivers: RefCell<Vec<Value>>,
    /// Plot scene records retain opaque tokens; callable values and GC roots
    /// remain in their issuing session and disappear at session teardown.
    #[cfg(feature = "plot-core")]
    pub(crate) plot_callbacks:
        RefCell<HashMap<u64, crate::builtins::plotting::state::callbacks::PlotCallbackRoot>>,
    pub(crate) events: RefCell<crate::EventRegistry>,
    pub(crate) runmat_extensions_enabled: Cell<bool>,
    pub(crate) language_mode: Cell<super::RuntimeLanguageMode>,
    pub(crate) top_level_await_enabled: Cell<bool>,
    pub(crate) dynamic_eval_enabled: Cell<bool>,
    pub(crate) execution_stack: Cell<super::RuntimeExecutionStack>,
    pub(crate) execution_assignment: RefCell<Option<runmat_execution::ProgramExecutionAssignment>>,
    pub(crate) execution_job: RefCell<Option<runmat_execution::JobId>>,
    pub(crate) callstack_limit: Cell<usize>,
    pub(crate) error_namespace: RefCell<String>,
    pub(crate) random: RefCell<crate::builtins::common::random::RuntimeRng>,
    pub(crate) random_turn: Arc<crate::builtins::common::random::RandomTurnGate>,
    /// The explicit GPU random stream is independent of the ordinary CPU
    /// stream even when placement later selects a device for CPU semantics.
    pub(crate) gpu_random: RefCell<crate::builtins::common::random::gpu::GpuRandomStream>,
}

impl RuntimeContextState {
    pub fn new(local_identity: super::RuntimeContextLocalId) -> Self {
        Self {
            local_identity,
            gpu_device_selection: Cell::new(super::gpu_device::GpuDeviceSelection::Inherit),
            gpu_device_epoch: Cell::new(1),
            gpu_device_reset: super::gpu_device::GpuDeviceResetState::default(),
            session_lease: Arc::new(()),
            host_effect_generation: Arc::new(AtomicU64::new(0)),
            resource_cleanup: SessionResourceCleanup::default(),
            #[cfg(target_arch = "wasm32")]
            worker_invocation_resources: Cell::new(false),
            cleanup_registry: RefCell::new(None),
            source: RefCell::new(SourceState::default()),
            call: RefCell::new(CallState::default()),
            workspace: RefCell::new(None),
            session_variables: RefCell::new(
                crate::workspace::session::SessionVariableState::default(),
            ),
            session_global_mutation: crate::workspace::session::SessionMutationGate::default(),
            warnings: RefCell::new(Vec::new()),
            warning_policy: RefCell::new(crate::warnings::WarningPolicy::default()),
            timing: RefCell::new(super::timing::TimingSessionState::default()),
            timers: RefCell::new(super::timer::TimerSessionState::default()),
            console: RefCell::new(crate::console::ConsoleState::default()),
            display_format: runmat_value::DisplayFormatOwner::default(),
            classes: RefCell::new(crate::class_registry::RuntimeClassState::default()),
            interaction: RefCell::new(crate::interaction::InteractionState::default()),
            test_services: RefCell::new(Vec::new()),
            test_contexts: RefCell::new(Vec::new()),
            constructor_receivers: RefCell::new(Vec::new()),
            #[cfg(feature = "plot-core")]
            plot_callbacks: RefCell::new(HashMap::new()),
            events: RefCell::new(crate::EventRegistry::default()),
            runmat_extensions_enabled: Cell::new(false),
            language_mode: Cell::new(super::RuntimeLanguageMode::Matlab),
            top_level_await_enabled: Cell::new(true),
            dynamic_eval_enabled: Cell::new(true),
            execution_stack: Cell::new(super::RuntimeExecutionStack::Process),
            execution_assignment: RefCell::new(None),
            execution_job: RefCell::new(None),
            callstack_limit: Cell::new(super::DEFAULT_CALLSTACK_LIMIT),
            error_namespace: RefCell::new(super::DEFAULT_ERROR_NAMESPACE.to_string()),
            random: RefCell::new(crate::builtins::common::random::RuntimeRng::new()),
            random_turn: Arc::new(crate::builtins::common::random::RandomTurnGate::default()),
            gpu_random: RefCell::new(crate::builtins::common::random::gpu::GpuRandomStream::new()),
        }
    }
}

impl Drop for RuntimeContextState {
    fn drop(&mut self) {
        // Domain owners register cleanup without making context depend on
        // builtin modules. No RefCell borrow survives a callback.
        for cleanup in self.resource_cleanup.0.get_mut().drain(..) {
            cleanup();
        }
    }
}

#[derive(Debug)]
pub(crate) struct SourceState {
    pub current: Option<SourceInfo>,
    /// One immutable catalog generation. Poll-local projections retain this
    /// owner rather than copying the source table at every suspension.
    pub catalog: Rc<HashMap<SourceId, SourceInfo>>,
    pub(crate) active_imports: HashMap<SourceId, usize>,
    pub(crate) imported_identities: HashMap<SourceId, runmat_execution::identity::Digest>,
    pub(crate) leased_catalog: HashMap<SourceId, crate::source_context::LeasedSource>,
    pub(crate) issued_authorities: HashMap<SourceId, crate::source_context::RuntimeSourceAuthority>,
    pub(crate) inherited_authorities: HashSet<SourceId>,
    pub(crate) next_issued_id: Option<usize>,
}

impl Default for SourceState {
    fn default() -> Self {
        Self {
            current: None,
            catalog: Rc::new(HashMap::new()),
            active_imports: HashMap::new(),
            imported_identities: HashMap::new(),
            leased_catalog: HashMap::new(),
            issued_authorities: HashMap::new(),
            inherited_authorities: HashSet::new(),
            next_issued_id: Some(0),
        }
    }
}

#[derive(Default, Clone)]
pub(crate) struct CallState {
    pub callsites: Vec<CallsiteInfo>,
    pub function_input_callsites: Vec<CallsiteInfo>,
    pub semantic_invoker: Option<std::rc::Rc<crate::user_functions::FunctionInvoker>>,
    pub bound_invoker: Option<std::rc::Rc<crate::user_functions::BoundFunctionInvoker>>,
    pub external_invoker: Option<std::rc::Rc<crate::user_functions::ExternalFunctionInvoker>>,
    pub lexical_invoker: Option<std::rc::Rc<crate::user_functions::LexicalFunctionInvoker>>,
    pub semantic_resolver: Option<std::sync::Arc<crate::user_functions::FunctionResolver>>,
    pub scoped_name_target_resolver:
        Option<std::rc::Rc<crate::user_functions::ScopedNameTargetResolver>>,
    pub semantic_name_candidates:
        Option<std::sync::Arc<crate::user_functions::SemanticNameCandidates>>,
    pub dynamic_loader: Option<std::rc::Rc<crate::user_functions::DynamicFunctionLoader>>,
    pub current_class_product_resolver: Option<std::rc::Rc<super::CurrentClassProductResolver>>,
    pub native_name_observer: Option<std::rc::Rc<crate::user_functions::NativeNameObserver>>,
    pub dynamic_clearer: Option<std::rc::Rc<crate::user_functions::DynamicFunctionClearer>>,
    pub dynamic_refresher: Option<std::rc::Rc<crate::user_functions::DynamicFunctionRefresher>>,
    pub source_functions: Option<std::sync::Arc<Vec<crate::user_functions::SourceFunctionInfo>>>,
    pub execution_source_resolver:
        Option<std::rc::Rc<crate::user_functions::ExecutionSourceResolver>>,
    pub active_functions: Vec<usize>,
    pub class_access: Option<runmat_types::ClassIdentity>,
}

impl std::fmt::Debug for CallState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CallState")
            .field("callsites", &self.callsites.len())
            .field(
                "function_input_callsites",
                &self.function_input_callsites.len(),
            )
            .field("semantic_invoker", &self.semantic_invoker.is_some())
            .field("external_invoker", &self.external_invoker.is_some())
            .field("lexical_invoker", &self.lexical_invoker.is_some())
            .field("semantic_resolver", &self.semantic_resolver.is_some())
            .field("dynamic_loader", &self.dynamic_loader.is_some())
            .field(
                "current_class_product_resolver",
                &self.current_class_product_resolver.is_some(),
            )
            .field("dynamic_clearer", &self.dynamic_clearer.is_some())
            .field("dynamic_refresher", &self.dynamic_refresher.is_some())
            .field(
                "source_functions",
                &self.source_functions.as_ref().map(|catalog| catalog.len()),
            )
            .field(
                "execution_source_resolver",
                &self.execution_source_resolver.is_some(),
            )
            .field("active_functions", &self.active_functions)
            .field("class_access", &self.class_access)
            .finish()
    }
}
