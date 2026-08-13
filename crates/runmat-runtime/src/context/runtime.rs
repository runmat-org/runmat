use super::{ContextFuture, RuntimeContextState, RuntimeServicePorts};
use crate::execution::RuntimeExecutionServices;
use std::future::Future;
use std::rc::Rc;
use std::sync::{atomic::AtomicBool, Arc};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeLanguageMode {
    Matlab,
    RunMat,
    Strict,
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
        let state = Rc::new(RuntimeContextState::new(cancellation));
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
        loader: Option<Arc<crate::user_functions::DynamicFunctionLoader>>,
    ) {
        self.state.call.borrow_mut().dynamic_loader = loader;
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

    /// Scope every poll of `future` to this context. This is the only supported
    /// bridge for legacy ambient APIs during R09–R29 migration.
    pub fn scope<F: Future>(&self, future: F) -> ContextFuture<F> {
        ContextFuture::new(self.clone(), future)
    }
}
