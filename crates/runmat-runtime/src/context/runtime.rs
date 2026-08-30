use super::{ContextFuture, RuntimeContextGuard, RuntimeContextState, RuntimeServicePorts};
use crate::execution::RuntimeExecutionServices;
use std::future::Future;
use std::rc::Rc;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};

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
    search_path: Option<Arc<crate::builtins::common::path_state::SearchPath>>,
    services: RuntimeServicePorts,
    state: Rc<RuntimeContextState>,
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
        Self::with_cancellation(execution, Arc::new(AtomicBool::new(false)))
    }

    pub fn with_cancellation(
        execution: Rc<dyn RuntimeExecutionServices>,
        cancellation: Arc<AtomicBool>,
    ) -> Self {
        runmat_accelerate_api::register_execution_provider_authorizer(
            super::scope::execution_provider_authorization,
        );
        let state = Rc::new(RuntimeContextState::new(
            RuntimeContextLocalId::next(),
            cancellation,
        ));
        crate::class_registry::register_context_state(&state);
        Self {
            execution,
            program_revision: None,
            search_path: None,
            services: RuntimeServicePorts::default(),
            state,
        }
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

    pub(super) fn state_identity(&self) -> *const RuntimeContextState {
        Rc::as_ptr(&self.state)
    }

    pub fn local_identity(&self) -> RuntimeContextLocalId {
        self.state.local_identity
    }

    pub fn cancellation(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.state.cancellation.borrow())
    }

    pub fn program_revision(&self) -> Option<&runmat_execution::ProgramRevision> {
        self.program_revision.as_ref()
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

    pub fn set_dynamic_function_clearer(
        &self,
        clearer: Option<Rc<crate::user_functions::DynamicFunctionClearer>>,
    ) {
        self.state.call.borrow_mut().dynamic_clearer = clearer;
    }

    pub fn runmat_extensions_enabled(&self) -> bool {
        self.state.runmat_extensions_enabled.get()
    }

    pub fn set_runmat_extensions_enabled(&self, enabled: bool) {
        self.state.runmat_extensions_enabled.set(enabled);
    }

    pub fn language_mode(&self) -> RuntimeLanguageMode {
        self.state.language_mode.get()
    }

    pub fn set_language_mode(&self, mode: RuntimeLanguageMode) {
        self.state.language_mode.set(mode);
    }

    pub fn top_level_await_enabled(&self) -> bool {
        self.state.top_level_await_enabled.get()
    }

    pub fn set_top_level_await_enabled(&self, enabled: bool) {
        self.state.top_level_await_enabled.set(enabled);
    }

    pub fn dynamic_eval_enabled(&self) -> bool {
        self.state.dynamic_eval_enabled.get()
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
        self.state.dynamic_eval_enabled.set(enabled);
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

    /// Create an isolated worker runtime that retains immutable resolution and
    /// host capabilities while owning independent mutable language state.
    pub fn fork_parallel_lab(&self, services: RuntimeServicePorts) -> Self {
        let child = Self::with_cancellation(self.execution.clone(), self.cancellation())
            .with_program_revision(self.program_revision.clone())
            .with_service_ports(services);
        let child = if let Some(search_path) = &self.search_path {
            child.with_search_path(Arc::clone(search_path))
        } else {
            child
        };
        *child.state.source.borrow_mut() = self.state.source.borrow().clone();
        *child.state.call.borrow_mut() = self.state.call.borrow().clone();
        *child.state.classes.borrow_mut() = self.state.classes.borrow().clone();
        *child.state.session_variables.borrow_mut() = self.state.session_variables.borrow().clone();
        child
            .state
            .runmat_extensions_enabled
            .set(self.state.runmat_extensions_enabled.get());
        child
            .state
            .language_mode
            .set(self.state.language_mode.get());
        child
            .state
            .top_level_await_enabled
            .set(self.state.top_level_await_enabled.get());
        child
            .state
            .dynamic_eval_enabled
            .set(self.state.dynamic_eval_enabled.get());
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

    /// Activate this context for one synchronous executor or foreign-host
    /// extent. Async code should use [`Self::scope`] so the context is removed
    /// across yields.
    pub fn enter(&self) -> RuntimeContextGuard {
        RuntimeContextGuard::enter(self.clone())
    }
}
