use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use runmat_types::SourceId;
use runmat_value::Value;

use crate::callsite::CallsiteInfo;
use crate::debug_context::DebugFrame;
use crate::source_context::SourceInfo;

/// Mutable state owned by exactly one runtime session.
///
/// Fields are split by semantic responsibility even though they share one
/// allocation. The legacy bridge is the only module allowed to manipulate the
/// complete state; new code uses the focused accessors on [`RuntimeContext`].
#[derive(Debug)]
pub struct RuntimeContextState {
    pub(crate) source: RefCell<SourceState>,
    pub(crate) call: RefCell<CallState>,
    pub(crate) output: RefCell<OutputState>,
    pub(crate) debug: RefCell<Vec<DebugFrame>>,
    pub(crate) workspace: RefCell<Option<crate::workspace::WorkspaceResolver>>,
    pub(crate) warnings: RefCell<Vec<crate::warning_store::RuntimeWarning>>,
    pub(crate) console: RefCell<crate::console::ConsoleState>,
    pub(crate) classes: RefCell<crate::class_registry::RuntimeClassState>,
    pub(crate) interaction: RefCell<crate::interaction::InteractionState>,
    pub(crate) test_services: RefCell<Vec<crate::testing::RuntimeTestServices>>,
    pub(crate) test_contexts: RefCell<Vec<std::rc::Rc<RefCell<crate::testing::ContextState>>>>,
    pub(crate) constructor_receivers: RefCell<Vec<Value>>,
    pub(crate) events: RefCell<crate::EventRegistry>,
    pub(crate) runmat_extensions_enabled: Cell<bool>,
    pub(crate) language_mode: Cell<super::RuntimeLanguageMode>,
    pub(crate) top_level_await_enabled: Cell<bool>,
    pub(crate) dynamic_eval_enabled: Cell<bool>,
    pub(crate) callstack_limit: Cell<usize>,
    pub(crate) error_namespace: RefCell<String>,
    pub(crate) cancellation: RefCell<Arc<AtomicBool>>,
}

impl RuntimeContextState {
    pub fn new(cancellation: Arc<AtomicBool>) -> Self {
        Self {
            source: RefCell::new(SourceState::default()),
            call: RefCell::new(CallState::default()),
            output: RefCell::new(OutputState::default()),
            debug: RefCell::new(Vec::new()),
            workspace: RefCell::new(None),
            warnings: RefCell::new(Vec::new()),
            console: RefCell::new(crate::console::ConsoleState::default()),
            classes: RefCell::new(crate::class_registry::RuntimeClassState::default()),
            interaction: RefCell::new(crate::interaction::InteractionState::default()),
            test_services: RefCell::new(Vec::new()),
            test_contexts: RefCell::new(Vec::new()),
            constructor_receivers: RefCell::new(Vec::new()),
            events: RefCell::new(crate::EventRegistry::default()),
            runmat_extensions_enabled: Cell::new(false),
            language_mode: Cell::new(super::RuntimeLanguageMode::Matlab),
            top_level_await_enabled: Cell::new(true),
            dynamic_eval_enabled: Cell::new(true),
            callstack_limit: Cell::new(200),
            error_namespace: RefCell::new("RunMat".to_string()),
            cancellation: RefCell::new(cancellation),
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancellation.borrow().load(Ordering::Relaxed)
    }
}

#[derive(Debug, Default)]
pub(crate) struct SourceState {
    pub current: Option<SourceInfo>,
    pub catalog: HashMap<SourceId, SourceInfo>,
}

#[derive(Default)]
pub(crate) struct CallState {
    pub callsites: Vec<CallsiteInfo>,
    pub function_input_callsites: Vec<CallsiteInfo>,
    pub semantic_invoker: Option<std::sync::Arc<crate::user_functions::FunctionInvoker>>,
    pub semantic_resolver: Option<std::sync::Arc<crate::user_functions::FunctionResolver>>,
    pub dynamic_loader: Option<std::sync::Arc<crate::user_functions::DynamicFunctionLoader>>,
    pub source_functions: Option<std::sync::Arc<Vec<crate::user_functions::SourceFunctionInfo>>>,
    pub active_functions: Vec<usize>,
    pub class_access: Option<String>,
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
            .field("semantic_resolver", &self.semantic_resolver.is_some())
            .field("dynamic_loader", &self.dynamic_loader.is_some())
            .field(
                "source_functions",
                &self.source_functions.as_ref().map(|catalog| catalog.len()),
            )
            .field("active_functions", &self.active_functions)
            .field("class_access", &self.class_access)
            .finish()
    }
}

#[derive(Debug, Default)]
pub(crate) struct OutputState {
    pub requested_outputs: Vec<Option<usize>>,
    pub presentation_outputs: Vec<usize>,
}
