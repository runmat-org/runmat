use super::{
    ContextFuture, RuntimeContextGuard, RuntimeContextState, RuntimeInvocationArity,
    RuntimeServicePorts,
};
use crate::execution::RuntimeExecutionServices;
use runmat_execution::cancellation::{CancellationSource, CancellationToken};
use std::cell::{Cell, RefCell};
use std::future::Future;
use std::rc::Rc;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

#[path = "runtime/source_selection.rs"]
mod source_selection;
use source_selection::InvocationSourceSelection;
pub use source_selection::ProgramSourceSelectionError;

static NEXT_RUNTIME_CONTEXT_ID: AtomicU64 = AtomicU64::new(1);

/// Process-local identity for one runtime context state allocation.
///
/// This identity lets compatibility adapters partition ambient state while
/// independently polled async sessions share an OS thread. It is not a
/// durable, serialized, or cross-process execution identity.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RuntimeContextLocalId(u64);

impl RuntimeContextLocalId {
    fn next() -> Self {
        let id = NEXT_RUNTIME_CONTEXT_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .expect("runtime context identity space exhausted");
        Self(id)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeLanguageMode {
    Matlab,
    RunMat,
    Strict,
}

/// Language and dynamic-source policy bound to one invocation.
///
/// Session defaults live in [`RuntimeContextState`]. An invocation projection
/// lets independently pending work retain the policy captured at admission
/// without mutating those shared defaults.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeLanguagePolicy {
    pub language_mode: RuntimeLanguageMode,
    pub extensions_enabled: bool,
    pub top_level_await_enabled: bool,
    pub dynamic_eval_enabled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeExecutionStack {
    Process,
    Segmented,
}

#[must_use]
pub struct RuntimeExecutionStackGuard {
    state: Rc<RuntimeContextState>,
    previous: RuntimeExecutionStack,
}

/// Restores the invocation assignment that was active before a nested program
/// execution entered this runtime context.
#[must_use]
pub struct ProgramExecutionAssignmentGuard {
    state: Rc<RuntimeContextState>,
    previous: Option<runmat_execution::ProgramExecutionAssignment>,
}

#[must_use]
pub struct ProgramExecutionJobGuard {
    state: Rc<RuntimeContextState>,
    previous: Option<runmat_execution::JobId>,
}

impl Drop for ProgramExecutionJobGuard {
    fn drop(&mut self) {
        *self.state.execution_job.borrow_mut() = self.previous.take();
    }
}

impl Drop for ProgramExecutionAssignmentGuard {
    fn drop(&mut self) {
        *self.state.execution_assignment.borrow_mut() = self.previous.take();
    }
}

impl Drop for RuntimeExecutionStackGuard {
    fn drop(&mut self) {
        self.state.execution_stack.set(self.previous);
    }
}

pub const DEFAULT_CALLSTACK_LIMIT: usize = 200;
pub const DEFAULT_ERROR_NAMESPACE: &str = "RunMat";

/// Complete explicit runtime authority for one session/invocation tree.
#[derive(Clone)]
pub struct RuntimeContext {
    execution: Rc<dyn RuntimeExecutionServices>,
    program_revision: Option<runmat_execution::ProgramRevision>,
    program_sources: Option<Rc<crate::source_context::ProgramSourceBinding>>,
    source_current: Option<Rc<RefCell<crate::source_context::SourceCurrentProjection>>>,
    source_selection: InvocationSourceSelection,
    source_catalog_view: Option<Rc<RefCell<crate::source_context::SourceCatalogProjection>>>,
    debug_projection: Rc<RefCell<crate::debug_context::DebugFrameProjection>>,
    output_projection: Rc<RefCell<super::output_projection::InvocationOutputProjection>>,
    search_path: Option<Arc<crate::builtins::common::path_state::SearchPath>>,
    services: RuntimeServicePorts,
    state: Rc<RuntimeContextState>,
    invocation_language_policy: Option<Rc<Cell<RuntimeLanguagePolicy>>>,
    invocation_arity: Option<RuntimeInvocationArity>,
    // Invocation-scoped, unlike shared session language state. Scoped guards
    // affect this invocation's clones, never an earlier/later generation.
    cancellation: Rc<RefCell<CancellationToken>>,
}

impl std::fmt::Debug for RuntimeContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RuntimeContext")
            .field("scope_id", &self.execution.scope_id())
            .field("program_revision", &self.program_revision)
            .field("services", &self.services)
            .finish_non_exhaustive()
    }
}

impl RuntimeContext {
    pub fn new(execution: Rc<dyn RuntimeExecutionServices>) -> Self {
        Self::with_cancellation(execution, CancellationSource::new().token())
    }

    pub fn with_cancellation(
        execution: Rc<dyn RuntimeExecutionServices>,
        cancellation: CancellationToken,
    ) -> Self {
        runmat_accelerate_api::register_execution_provider_authorizer(
            super::scope::execution_provider_authorization,
        );
        let state = Rc::new(RuntimeContextState::new(RuntimeContextLocalId::next()));
        crate::class_registry::register_context_state(&state);
        let context = Self {
            execution,
            program_revision: None,
            program_sources: None,
            source_current: None,
            source_selection: InvocationSourceSelection::Inherited,
            source_catalog_view: None,
            debug_projection: Rc::new(RefCell::new(Default::default())),
            output_projection: Rc::new(RefCell::new(Default::default())),
            search_path: None,
            services: RuntimeServicePorts::default(),
            state,
            invocation_language_policy: None,
            invocation_arity: None,
            cancellation: Rc::new(RefCell::new(cancellation)),
        };
        {
            let _guard = context.enter();
            crate::builtins::deep_learning::model::ensure_network_classes_registered();
        }
        context
    }

    pub fn execution(&self) -> &Rc<dyn RuntimeExecutionServices> {
        &self.execution
    }

    pub fn service_ports(&self) -> &RuntimeServicePorts {
        &self.services
    }

    pub(crate) fn state(&self) -> &Rc<RuntimeContextState> {
        &self.state
    }

    /// Read one value from this exact session's global namespace without
    /// relying on ambient runtime activation.
    pub fn session_global_value(
        &self,
        name: &str,
    ) -> Result<Option<runmat_value::Value>, crate::RuntimeError> {
        self.state
            .session_variables
            .try_borrow()
            .map(|variables| variables.global_value(name))
            .map_err(|error| {
                crate::runtime_error::semantic_error(
                    "RunMat:Workspace:GlobalBorrow",
                    format!("session-global workspace read is reentrant: {error}"),
                )
            })
    }

    /// Enumerate this exact session's global namespace without relying on
    /// ambient runtime activation.
    pub fn session_global_names(&self) -> Result<Vec<String>, crate::RuntimeError> {
        self.state
            .session_variables
            .try_borrow()
            .map(|variables| variables.global_names())
            .map_err(|error| {
                crate::runtime_error::semantic_error(
                    "RunMat:Workspace:GlobalBorrow",
                    format!("session-global workspace read is reentrant: {error}"),
                )
            })
    }

    pub fn session_variable_authority(
        &self,
    ) -> crate::workspace::session::SessionVariableAuthority {
        crate::workspace::session::SessionVariableAuthority::new(&self.state)
    }

    pub(super) fn state_identity(&self) -> *const RuntimeContextState {
        Rc::as_ptr(&self.state)
    }

    pub fn local_identity(&self) -> RuntimeContextLocalId {
        self.state.local_identity
    }

    /// Process-local, non-serialized authority for this session's lifetime.
    /// Values and foreign registries must retain only a weak copy.
    pub(crate) fn session_lease(&self) -> std::sync::Weak<()> {
        Arc::downgrade(&self.state.session_lease)
    }

    pub(crate) fn retain_plot_callback(
        &self,
        slot: u64,
        root: crate::builtins::plotting::state::callbacks::PlotCallbackRoot,
    ) {
        let mut callbacks = self.state.plot_callbacks.borrow_mut();
        callbacks.retain(|_, root| root.live.load(Ordering::Acquire));
        callbacks.insert(slot, root);
    }

    pub(crate) fn plot_callback(&self, slot: u64) -> Option<runmat_value::Value> {
        let mut callbacks = self.state.plot_callbacks.borrow_mut();
        callbacks.retain(|_, root| root.live.load(Ordering::Acquire));
        callbacks.get(&slot).map(|root| root.value.clone())
    }

    pub(crate) fn release_plot_callback(&self, slot: u64) {
        self.state.plot_callbacks.borrow_mut().remove(&slot);
    }

    #[cfg(test)]
    pub(crate) fn plot_callback_root_count(&self) -> usize {
        let mut callbacks = self.state.plot_callbacks.borrow_mut();
        callbacks.retain(|_, root| root.live.load(Ordering::Acquire));
        callbacks.len()
    }

    /// Invalidate only this session's staged host-effect invocations when its
    /// installed resource authority changes. Shared ledger owners remain
    /// usable by other sessions and retain prior physical output charges.
    pub fn invalidate_host_effect_session_authority(&self) {
        self.state
            .host_effect_generation
            .fetch_add(1, Ordering::AcqRel);
    }

    pub(crate) fn host_effect_session_generation(&self) -> (Arc<AtomicU64>, u64) {
        let owner = self.state.host_effect_generation.clone();
        let generation = owner.load(Ordering::Acquire);
        (owner, generation)
    }

    pub(crate) fn register_session_resource_cleanup(&self, cleanup: impl FnOnce() + 'static) {
        self.state
            .resource_cleanup
            .0
            .borrow_mut()
            .push(Box::new(cleanup));
    }

    pub fn cancellation(&self) -> CancellationToken {
        self.cancellation.borrow().clone()
    }

    /// Project the selected call's arity without changing sibling futures or
    /// the parent invocation. The VM/native caller owns the actual counts.
    pub fn with_invocation_arity(mut self, arity: RuntimeInvocationArity) -> Self {
        self.invocation_arity = Some(arity);
        self
    }

    pub fn invocation_arity(&self) -> Option<RuntimeInvocationArity> {
        self.invocation_arity
    }

    /// Create work accounting for this invocation generation. The adapter
    /// owns the quantum and the actual scheduling primitive.
    pub fn cpu_cooperation(&self) -> crate::execution::CpuCooperationBudget {
        crate::execution::CpuCooperationBudget::new(self.execution.clone(), self.cancellation())
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancellation.borrow().is_cancelled()
    }

    /// Bind a fresh invocation generation while retaining the session's
    /// language state and services. Previously cloned contexts stay bound to
    /// their own invocation generation.
    pub fn with_cancellation_token(mut self, cancellation: CancellationToken) -> Self {
        self.cancellation = Rc::new(RefCell::new(cancellation));
        self
    }

    pub(crate) fn replace_cancellation_token(
        &self,
        cancellation: CancellationToken,
    ) -> CancellationToken {
        self.cancellation.replace(cancellation)
    }

    pub fn program_revision(&self) -> Option<&runmat_execution::ProgramRevision> {
        self.program_revision.as_ref()
    }

    /// Issue session-local source leases for a complete immutable program.
    /// No caller-provided numeric source identity is imported into the issuer.
    pub fn with_program_sources(
        mut self,
        records: impl IntoIterator<Item = crate::source_context::ProgramSourceRecord>,
    ) -> Result<Self, crate::source_context::ProgramSourceBindingError> {
        let binding = crate::source_context::bind_program_sources(&self, records)?;
        self.program_sources = Some(Rc::new(binding));
        self.source_current = Some(Rc::new(RefCell::new(
            crate::source_context::SourceCurrentProjection::new(None),
        )));
        self.source_selection = InvocationSourceSelection::Inherited;
        self.source_catalog_view = Some(Rc::new(RefCell::new(
            crate::source_context::SourceCatalogProjection::new(
                self.state.source.borrow().catalog.clone(),
            ),
        )));
        Ok(self)
    }

    pub fn portable_program_source_map(&self) -> Option<&runmat_execution::ExecutableSourceMapV2> {
        self.program_sources
            .as_ref()
            .and_then(|binding| binding.portable_map())
            .map(|map| map.as_ref())
    }

    pub fn portable_ordinal_for_source(
        &self,
        source: crate::source_context::RuntimeSourceRef,
    ) -> Option<runmat_types::ProgramSourceId> {
        source.resolve_in(self)?;
        self.program_sources.as_ref()?.source_ordinal(source)
    }

    pub(crate) fn retain_portable_program_source_map(
        &mut self,
        map: std::sync::Arc<runmat_execution::ExecutableSourceMapV2>,
    ) {
        let binding =
            self.program_sources.as_mut().and_then(Rc::get_mut).expect(
                "newly issued program binding is not shared before portable map attachment",
            );
        binding.retain_portable_map(map);
    }

    pub(crate) fn source_catalog_view(
        &self,
    ) -> Option<&Rc<RefCell<crate::source_context::SourceCatalogProjection>>> {
        self.source_catalog_view.as_ref()
    }

    pub(crate) fn same_source_projection_owner(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.state, &other.state)
            && self.source_selection.same_lineage(&other.source_selection)
            && match (&self.program_sources, &other.program_sources) {
                (Some(left), Some(right)) => Rc::ptr_eq(left, right),
                (None, None) => true,
                _ => false,
            }
    }

    pub(super) fn with_poll_debug_frames(mut self) -> Self {
        let snapshot = self.debug_projection.borrow().snapshot();
        self.debug_projection = Rc::new(RefCell::new(snapshot));
        self
    }

    pub(super) fn with_poll_output_projection(mut self) -> Self {
        let owner = super::scope::active_runtime_context()
            .filter(|active| Rc::ptr_eq(active.state(), self.state()))
            .map(|active| Rc::clone(active.output_projection()))
            .unwrap_or_else(|| Rc::clone(&self.output_projection));
        let snapshot = owner.borrow().snapshot();
        self.output_projection = Rc::new(RefCell::new(snapshot));
        self
    }

    pub(crate) fn output_projection(
        &self,
    ) -> &Rc<RefCell<super::output_projection::InvocationOutputProjection>> {
        &self.output_projection
    }

    pub(crate) fn debug_projection(
        &self,
    ) -> &Rc<RefCell<crate::debug_context::DebugFrameProjection>> {
        &self.debug_projection
    }

    /// Add an immutable VM frame to this invocation only. Independently
    /// pending siblings keep their own linked frame heads.
    pub fn with_debug_frame(mut self, frame: crate::debug_context::DebugFrame) -> Self {
        let projection = self.debug_projection.borrow().with_frame(frame);
        self.debug_projection = Rc::new(RefCell::new(projection));
        self
    }

    pub(crate) fn source_current(
        &self,
    ) -> Option<&Rc<RefCell<crate::source_context::SourceCurrentProjection>>> {
        self.source_current.as_ref()
    }

    pub fn program_source_id(
        &self,
        ordinal: runmat_types::ProgramSourceId,
    ) -> Option<runmat_types::SourceId> {
        self.program_sources.as_ref()?.get(ordinal)
    }

    pub fn program_source_ref(
        &self,
        ordinal: runmat_types::ProgramSourceId,
    ) -> Option<crate::source_context::RuntimeSourceRef> {
        self.program_sources.as_ref()?.source_ref(ordinal)
    }

    pub fn with_program_revision(
        mut self,
        revision: Option<runmat_execution::ProgramRevision>,
    ) -> Self {
        if self.program_revision != revision {
            if let (Some(service), Some(previous)) =
                (self.services.placement(), self.program_revision.clone())
            {
                service.invalidate(runmat_execution::PlacementInvalidation::Program {
                    revision: previous,
                });
            }
        }
        self.program_revision = revision;
        self
    }

    pub fn with_execution(mut self, execution: Rc<dyn RuntimeExecutionServices>) -> Self {
        self.execution = execution;
        self
    }

    pub fn with_search_path(
        mut self,
        search_path: Arc<crate::builtins::common::path_state::SearchPath>,
    ) -> Self {
        self.search_path = Some(search_path);
        self
    }

    pub fn search_path(&self) -> Option<&Arc<crate::builtins::common::path_state::SearchPath>> {
        self.search_path.as_ref()
    }

    pub fn set_dynamic_function_loader(
        &self,
        loader: Option<Rc<crate::user_functions::DynamicFunctionLoader>>,
    ) {
        self.state.call.borrow_mut().dynamic_loader = loader;
    }

    /// Install the selected session's exact-definition invocation route.
    /// This does not affect the ordinary name or numeric-function resolver.
    pub fn set_bound_function_invoker(
        &self,
        invoker: Option<Rc<crate::user_functions::BoundFunctionInvoker>>,
    ) {
        self.state.call.borrow_mut().bound_invoker = invoker;
    }

    pub fn set_dynamic_function_clearer(
        &self,
        clearer: Option<Rc<crate::user_functions::DynamicFunctionClearer>>,
    ) {
        self.state.call.borrow_mut().dynamic_clearer = clearer;
    }

    pub fn set_dynamic_function_refresher(
        &self,
        refresher: Option<Rc<crate::user_functions::DynamicFunctionRefresher>>,
    ) {
        self.state.call.borrow_mut().dynamic_refresher = refresher;
    }

    pub fn runmat_extensions_enabled(&self) -> bool {
        self.invocation_language_policy
            .as_ref()
            .map(|policy| policy.get().extensions_enabled)
            .unwrap_or_else(|| self.state.runmat_extensions_enabled.get())
    }

    pub fn set_runmat_extensions_enabled(&self, enabled: bool) {
        if let Some(policy) = &self.invocation_language_policy {
            let mut updated = policy.get();
            updated.extensions_enabled = enabled;
            policy.set(updated);
        } else {
            self.state.runmat_extensions_enabled.set(enabled);
        }
    }

    pub fn language_mode(&self) -> RuntimeLanguageMode {
        self.invocation_language_policy
            .as_ref()
            .map(|policy| policy.get().language_mode)
            .unwrap_or_else(|| self.state.language_mode.get())
    }

    pub fn set_language_mode(&self, mode: RuntimeLanguageMode) {
        if let Some(policy) = &self.invocation_language_policy {
            let mut updated = policy.get();
            updated.language_mode = mode;
            policy.set(updated);
        } else {
            self.state.language_mode.set(mode);
        }
    }

    pub fn top_level_await_enabled(&self) -> bool {
        self.invocation_language_policy
            .as_ref()
            .map(|policy| policy.get().top_level_await_enabled)
            .unwrap_or_else(|| self.state.top_level_await_enabled.get())
    }

    pub fn set_top_level_await_enabled(&self, enabled: bool) {
        if let Some(policy) = &self.invocation_language_policy {
            let mut updated = policy.get();
            updated.top_level_await_enabled = enabled;
            policy.set(updated);
        } else {
            self.state.top_level_await_enabled.set(enabled);
        }
    }

    pub fn dynamic_eval_enabled(&self) -> bool {
        self.invocation_language_policy
            .as_ref()
            .map(|policy| policy.get().dynamic_eval_enabled)
            .unwrap_or_else(|| self.state.dynamic_eval_enabled.get())
    }

    pub fn execution_stack(&self) -> RuntimeExecutionStack {
        self.state.execution_stack.get()
    }

    pub fn enter_execution_stack(
        &self,
        stack: RuntimeExecutionStack,
    ) -> RuntimeExecutionStackGuard {
        let previous = self.state.execution_stack.replace(stack);
        RuntimeExecutionStackGuard {
            state: Rc::clone(&self.state),
            previous,
        }
    }

    pub fn execution_assignment(&self) -> Option<runmat_execution::ProgramExecutionAssignment> {
        self.state.execution_assignment.borrow().clone()
    }

    /// Installs scheduler-owned invocation identity for one execution extent.
    /// Dropping the returned guard restores the previous identity, which keeps
    /// nested and cancelled executions from leaking worker state into their
    /// caller's runtime context.
    pub fn enter_execution_assignment(
        &self,
        assignment: Option<runmat_execution::ProgramExecutionAssignment>,
    ) -> ProgramExecutionAssignmentGuard {
        let previous = self.state.execution_assignment.replace(assignment);
        ProgramExecutionAssignmentGuard {
            state: Rc::clone(&self.state),
            previous,
        }
    }

    pub fn execution_job(&self) -> Option<runmat_execution::JobId> {
        *self.state.execution_job.borrow()
    }

    pub fn enter_execution_job(
        &self,
        job_id: Option<runmat_execution::JobId>,
    ) -> ProgramExecutionJobGuard {
        let previous = self.state.execution_job.replace(job_id);
        ProgramExecutionJobGuard {
            state: Rc::clone(&self.state),
            previous,
        }
    }

    pub fn set_dynamic_eval_enabled(&self, enabled: bool) {
        if let Some(policy) = &self.invocation_language_policy {
            let mut updated = policy.get();
            updated.dynamic_eval_enabled = enabled;
            policy.set(updated);
        } else {
            self.state.dynamic_eval_enabled.set(enabled);
        }
    }

    pub fn callstack_limit(&self) -> usize {
        self.state.callstack_limit.get()
    }

    pub fn set_callstack_limit(&self, limit: usize) {
        self.state.callstack_limit.set(limit);
    }

    pub fn error_namespace(&self) -> String {
        self.state.error_namespace.borrow().clone()
    }

    pub fn set_error_namespace(&self, namespace: impl Into<String>) {
        let namespace = namespace.into();
        let namespace = if namespace.trim().is_empty() {
            DEFAULT_ERROR_NAMESPACE.to_string()
        } else {
            namespace
        };
        *self.state.error_namespace.borrow_mut() = namespace;
    }

    pub fn with_service_ports(mut self, services: RuntimeServicePorts) -> Self {
        self.services = services;
        self
    }

    /// Bind language policy to this invocation and its clones. This does not
    /// change the session defaults observed by sibling or later invocations.
    pub fn with_language_policy(mut self, policy: RuntimeLanguagePolicy) -> Self {
        self.invocation_language_policy = Some(Rc::new(Cell::new(policy)));
        self
    }

    /// Create an isolated worker runtime that retains immutable resolution and
    /// host capabilities while owning independent mutable language state.
    pub fn fork_parallel_lab(&self, services: RuntimeServicePorts) -> Self {
        // A lab is an isolated mutable-language-state owner. Caller-provided
        // ports cannot confer either its current frame or session base
        // workspace; a lab workspace must be installed explicitly afterward.
        let services = services
            .without_workspace()
            .without_base_workspace()
            .without_graphics_presentation()
            .without_callable_artifacts();
        let child = Self::with_cancellation(self.execution.clone(), self.cancellation())
            .with_program_revision(self.program_revision.clone())
            .with_service_ports(services);
        let child = if let Some(search_path) = &self.search_path {
            child.with_search_path(Arc::clone(search_path))
        } else {
            child
        };
        *child.state.source.borrow_mut() = self.state.source.borrow().fork_snapshot();
        let mut child = child;
        child.invocation_arity = self.invocation_arity;
        if let Some(sources) = &self.program_sources {
            child.program_sources = Some(Rc::new(sources.fork_snapshot()));
        }
        if let Some(current) = &self.source_current {
            child.source_current = Some(Rc::new(RefCell::new(
                crate::source_context::SourceCurrentProjection::new(current.borrow().snapshot()),
            )));
        }
        if let Some(view) = &self.source_catalog_view {
            child.source_catalog_view = Some(Rc::new(RefCell::new(
                view.borrow()
                    .fork_with_inherited_sources(&child.state.source.borrow()),
            )));
        }
        *child.state.call.borrow_mut() = self.state.call.borrow().clone();
        *child.state.classes.borrow_mut() = self.state.classes.borrow().clone();
        *child.state.session_variables.borrow_mut() = self.state.session_variables.borrow().clone();
        child
            .state
            .runmat_extensions_enabled
            .set(self.runmat_extensions_enabled());
        child.state.language_mode.set(self.language_mode());
        child
            .state
            .top_level_await_enabled
            .set(self.top_level_await_enabled());
        child
            .state
            .dynamic_eval_enabled
            .set(self.dynamic_eval_enabled());
        child
            .state
            .execution_stack
            .set(self.state.execution_stack.get());
        child
            .state
            .callstack_limit
            .set(self.state.callstack_limit.get());
        *child.state.error_namespace.borrow_mut() = self.state.error_namespace.borrow().clone();
        child
    }

    /// Scope every poll of `future` to this context. This is the supported
    /// bridge for async code that still reaches ambient compatibility APIs.
    pub fn scope<F: Future>(&self, future: F) -> ContextFuture<F> {
        ContextFuture::new(self.clone(), future)
    }

    /// Carry this frame's lexical catch exception through every poll, never
    /// across a yield. `None` explicitly masks an enclosing frame's catch.
    pub fn scope_active_catch<F: Future>(
        &self,
        exception: Option<runmat_value::MException>,
        future: F,
    ) -> super::ActiveCatchFuture<F> {
        super::ActiveCatchFuture::new(self, exception, future)
    }

    pub fn scope_active_catch_shared<F: Future>(
        &self,
        exception: Option<Rc<runmat_value::MException>>,
        future: F,
    ) -> super::ActiveCatchFuture<F> {
        super::ActiveCatchFuture::new_shared(self, exception, future)
    }

    /// Scope a pinned execution child without allocating another future box.
    /// The child is also dropped under the catch authority on abandonment.
    pub fn scope_active_catch_unboxed_shared<F: Future>(
        &self,
        exception: Option<Rc<runmat_value::MException>>,
        future: F,
    ) -> super::UnboxedActiveCatchFuture<F> {
        super::UnboxedActiveCatchFuture::new(self, exception, future)
    }

    pub fn enter_active_catch(
        &self,
        exception: Option<runmat_value::MException>,
    ) -> super::ActiveCatchGuard {
        super::ActiveCatchGuard::enter(self.local_identity(), exception.map(Rc::new))
    }

    /// Enter one synchronous poll with an already-owned exception. This is
    /// used by execution engines to avoid cloning exception payloads per poll.
    pub fn enter_active_catch_shared(
        &self,
        exception: Option<Rc<runmat_value::MException>>,
    ) -> super::ActiveCatchGuard {
        super::ActiveCatchGuard::enter(self.local_identity(), exception)
    }

    pub fn active_catch_exception(&self) -> Option<runmat_value::MException> {
        super::catch::active_for(self)
    }

    /// Snapshot a caller's active catch for a nested invocation without
    /// copying its payload. The callee's own lexical catch may override it.
    pub fn active_catch_exception_shared(&self) -> Option<Rc<runmat_value::MException>> {
        super::catch::active_shared_for(self)
    }

    /// Activate this context for one synchronous executor or foreign-host
    /// extent. Async code should use [`Self::scope`] so the context is removed
    /// across yields.
    pub fn enter(&self) -> RuntimeContextGuard {
        RuntimeContextGuard::enter(self.clone())
    }
}
