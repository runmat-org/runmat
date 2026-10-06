use runmat_thread_local::runmat_thread_local;
use runmat_value::Value;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::{Rc, Weak};

/// Exact, non-owning capability for one Runtime session's variable store.
/// Executor services retain it across suspension instead of re-resolving an
/// ambient RuntimeContext which may belong to another invocation.
#[derive(Clone, Debug)]
pub struct SessionVariableAuthority {
    state: Weak<crate::context::RuntimeContextState>,
}

impl SessionVariableAuthority {
    pub(crate) fn new(state: &Rc<crate::context::RuntimeContextState>) -> Self {
        Self {
            state: Rc::downgrade(state),
        }
    }

    fn with_state<R>(
        &self,
        operation: impl FnOnce(&SessionVariableState) -> R,
    ) -> Result<R, crate::RuntimeError> {
        let state = self.state.upgrade().ok_or_else(session_variables_closed)?;
        state
            .session_variables
            .try_borrow()
            .map(|variables| operation(&variables))
            .map_err(session_variables_read_borrow)
    }

    fn with_state_mut<R>(
        &self,
        operation: impl FnOnce(&mut SessionVariableState) -> R,
    ) -> Result<R, crate::RuntimeError> {
        let state = self.state.upgrade().ok_or_else(session_variables_closed)?;
        state
            .session_variables
            .try_borrow_mut()
            .map(|mut variables| operation(&mut variables))
            .map_err(session_variables_write_borrow)
    }

    pub fn global_value(&self, name: &str) -> Result<Option<Value>, crate::RuntimeError> {
        self.with_state(|state| state.global_value(name))
    }

    pub fn global_names(&self) -> Result<Vec<String>, crate::RuntimeError> {
        self.with_state(SessionVariableState::global_names)
    }

    pub fn global_revision(&self) -> Result<GlobalRevision, crate::RuntimeError> {
        self.with_state(|state| state.global_revision.clone())
    }

    pub fn declare_global(&self, name: &str) -> Result<Value, crate::RuntimeError> {
        self.with_state_mut(|state| state.declare_global(name))
    }

    pub fn store_global(&self, name: &str, value: Value) -> Result<(), crate::RuntimeError> {
        self.with_state_mut(|state| state.store_global(name, value))
    }

    pub fn global_slot(&self, slot: usize) -> Result<Option<Value>, crate::RuntimeError> {
        self.with_state(|state| state.global_slots.get(&slot).cloned())
    }

    pub fn store_global_slot(&self, slot: usize, value: Value) -> Result<(), crate::RuntimeError> {
        self.with_state_mut(|state| {
            state.global_slots.insert(slot, value);
            state.global_revision = GlobalRevision::default();
        })
    }
}

fn session_variables_closed() -> crate::RuntimeError {
    crate::runtime_error::semantic_error(
        "RunMat:Workspace:SessionClosed",
        "session variable authority is closed",
    )
}

fn session_variables_read_borrow(error: std::cell::BorrowError) -> crate::RuntimeError {
    crate::runtime_error::semantic_error(
        "RunMat:Workspace:GlobalBorrow",
        format!("session-global workspace read is reentrant: {error}"),
    )
}

fn session_variables_write_borrow(error: std::cell::BorrowMutError) -> crate::RuntimeError {
    crate::runtime_error::semantic_error(
        "RunMat:Workspace:GlobalBorrow",
        format!("session-global workspace mutation is reentrant: {error}"),
    )
}

/// Opaque identity for one version of the session global store. Retaining the
/// token makes equality immune to counter wraparound and state replacement.
#[derive(Clone, Debug)]
pub struct GlobalRevision(Rc<()>);

impl Default for GlobalRevision {
    fn default() -> Self {
        Self(Rc::new(()))
    }
}

impl PartialEq for GlobalRevision {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for GlobalRevision {}

/// Named session variables and legacy bytecode-keyed slot storage.
///
/// Slot-to-local synchronization belongs to the executor. This state owns only
/// values whose lifetime and identity cross an individual VM frame.
#[derive(Debug, Default, Clone)]
pub struct SessionVariableState {
    globals: HashMap<String, Value>,
    global_slots: HashMap<usize, Value>,
    global_revision: GlobalRevision,
    persistent_slots: HashMap<(String, usize), Value>,
    persistent_names: HashMap<(String, String), Value>,
}

impl SessionVariableState {
    pub(crate) fn global_value(&self, name: &str) -> Option<Value> {
        self.globals.get(name).cloned()
    }

    pub(crate) fn global_names(&self) -> Vec<String> {
        let mut names = self.globals.keys().cloned().collect::<Vec<_>>();
        names.sort();
        names
    }

    fn declare_global(&mut self, name: &str) -> Value {
        if let Some(value) = self.globals.get(name) {
            return value.clone();
        }
        let value = canonical_empty_workspace_value();
        self.globals.insert(name.to_string(), value.clone());
        self.global_revision = GlobalRevision::default();
        value
    }

    fn store_global(&mut self, name: &str, value: Value) {
        self.globals.insert(name.to_string(), value);
        self.global_revision = GlobalRevision::default();
    }
}

runmat_thread_local! {
    static LEGACY_SESSION_VARIABLES: RefCell<SessionVariableState> = RefCell::new(SessionVariableState::default());
}

fn with_state<R>(operation: impl FnOnce(&SessionVariableState) -> R) -> R {
    if let Some(context) = crate::context::legacy::active() {
        return operation(&context.state().session_variables.borrow());
    }
    LEGACY_SESSION_VARIABLES.with(|state| operation(&state.borrow()))
}

fn with_state_mut<R>(operation: impl FnOnce(&mut SessionVariableState) -> R) -> R {
    if let Some(context) = crate::context::legacy::active() {
        return operation(&mut context.state().session_variables.borrow_mut());
    }
    LEGACY_SESSION_VARIABLES.with(|state| operation(&mut state.borrow_mut()))
}

pub fn global_names() -> Vec<String> {
    with_state(SessionVariableState::global_names)
}

pub fn global_value(name: &str) -> Option<Value> {
    with_state(|state| state.global_value(name))
}

/// Establish a named global and return its current value. A first declaration
/// initializes the shared value to the canonical empty matrix, so declaring a
/// global never exposes a stale local value from the declaring frame.
pub fn declare_global_named(name: &str) -> Value {
    with_state_mut(|state| state.declare_global(name))
}

fn canonical_empty_workspace_value() -> Value {
    Value::Tensor(
        runmat_value::Tensor::new(Vec::new(), vec![0, 0])
            .expect("the canonical empty workspace shape is valid"),
    )
}

/// Revision of the session's named global store. Native frames use this to
/// avoid repeatedly cloning and rooting an unchanged global on every read.
pub fn global_revision() -> GlobalRevision {
    with_state(|state| state.global_revision.clone())
}

/// Store a global through its semantic name.
///
/// Executor-local slot numbers are deliberately not part of this API. Native,
/// bytecode, and browser hosts can therefore share one session global without
/// agreeing on a frame layout.
pub fn store_global_named(name: &str, value: Value) {
    with_state_mut(|state| state.store_global(name, value));
}

pub fn roots() -> Vec<Value> {
    with_state(|state| {
        state
            .globals
            .values()
            .chain(state.global_slots.values())
            .chain(state.persistent_slots.values())
            .chain(state.persistent_names.values())
            .cloned()
            .collect()
    })
}

/// Borrow session roots without copying tensor payloads. An active mutation
/// makes the view unavailable rather than proving there are no retained aliases.
pub fn visit_roots(visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
    fn visit(
        state: &SessionVariableState,
        visitor: &mut dyn FnMut(&Value) -> Option<()>,
    ) -> Option<()> {
        for value in state
            .globals
            .values()
            .chain(state.global_slots.values())
            .chain(state.persistent_slots.values())
            .chain(state.persistent_names.values())
        {
            visitor(value)?;
        }
        Some(())
    }
    if let Some(context) = crate::context::legacy::active() {
        let state = context.state().session_variables.try_borrow().ok()?;
        return visit(&state, visitor);
    }
    LEGACY_SESSION_VARIABLES.with(|state| {
        let state = state.try_borrow().ok()?;
        visit(&state, visitor)
    })
}

/// Store a global addressed by a legacy bytecode slot. This namespace is
/// distinct from user-written global names, even when a name is `var_4`.
pub fn store_global_slot(index: usize, value: Value) {
    with_state_mut(|state| {
        state.global_slots.insert(index, value);
        state.global_revision = GlobalRevision::default();
    });
}

pub fn global_slot_value(index: usize) -> Option<Value> {
    with_state(|state| state.global_slots.get(&index).cloned())
}

pub fn persistent_slot_value(function: &str, index: usize) -> Option<Value> {
    with_state(|state| {
        state
            .persistent_slots
            .get(&(function.to_string(), index))
            .cloned()
    })
}

pub fn persistent_named_value(function: &str, name: &str) -> Option<Value> {
    with_state(|state| {
        state
            .persistent_names
            .get(&(function.to_string(), name.to_string()))
            .cloned()
    })
}

pub fn update_persistent_slot(function: &str, index: usize, value: &Value) {
    with_state_mut(|state| {
        let key = (function.to_string(), index);
        if state.persistent_slots.contains_key(&key) {
            state.persistent_slots.insert(key, value.clone());
        }
    });
}

pub fn store_persistent_slot(function: &str, index: usize, value: Value) {
    with_state_mut(|state| {
        state
            .persistent_slots
            .insert((function.to_string(), index), value);
    });
}

pub fn store_persistent_named(function: &str, name: &str, value: Value) {
    with_state_mut(|state| {
        state
            .persistent_names
            .insert((function.to_string(), name.to_string()), value);
    });
}

#[doc(hidden)]
pub fn reset_legacy_state_for_tests() {
    LEGACY_SESSION_VARIABLES.with(|state| *state.borrow_mut() = SessionVariableState::default());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::RuntimeContext;
    use crate::execution::RuntimeExecutionService;
    use futures::executor::block_on;
    use std::rc::Rc;

    #[test]
    fn session_variable_state_is_context_isolated() {
        let first = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let second = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));

        block_on(first.scope(async {
            store_global_named("answer", Value::Num(42.0));
            assert_eq!(global_value("answer"), Some(Value::Num(42.0)));
        }));
        block_on(second.scope(async {
            assert_eq!(global_value("answer"), None);
        }));
    }

    #[test]
    fn semantic_names_bridge_executor_specific_global_and_persistent_layouts() {
        let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        block_on(context.scope(async {
            store_global_named("answer", Value::Num(42.0));
            assert_eq!(global_value("answer"), Some(Value::Num(42.0)));

            store_persistent_named("counter", "calls", Value::Num(3.0));
            assert_eq!(
                persistent_named_value("counter", "calls"),
                Some(Value::Num(3.0))
            );
        }));
    }

    #[test]
    fn global_revision_changes_for_each_canonical_write_without_aba() {
        let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        block_on(context.scope(async {
            let initial = global_revision();
            store_global_named("answer", Value::Num(1.0));
            let first = global_revision();
            assert_ne!(initial, first);
            assert_eq!(first, global_revision());
            store_global_named("answer", Value::Num(2.0));
            let second = global_revision();
            assert_ne!(first, second);
            store_global_slot(4, Value::Num(2.0));
            let third = global_revision();
            assert_ne!(second, third);
            assert_ne!(initial, third);
            assert_eq!(global_value("answer"), Some(Value::Num(2.0)));
        }));
    }

    #[test]
    fn user_global_with_legacy_slot_spelling_stays_named_and_disjoint() {
        let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        block_on(context.scope(async {
            store_global_named("var_4", Value::Num(11.0));
            store_global_named("other", Value::Num(2.0));
            assert_eq!(global_names(), vec!["other", "var_4"]);

            let before_binding = global_revision();
            store_global_slot(4, Value::Num(2.0));
            assert_ne!(global_revision(), before_binding);
            assert_eq!(global_slot_value(4), Some(Value::Num(2.0)));
            assert_eq!(global_value("var_4"), Some(Value::Num(11.0)));

            store_global_slot(4, Value::Num(3.0));
            store_global_named("other", Value::Num(3.0));
            assert_eq!(global_slot_value(4), Some(Value::Num(3.0)));
            assert_eq!(global_value("other"), Some(Value::Num(3.0)));
            assert_eq!(global_value("var_4"), Some(Value::Num(11.0)));
            assert_eq!(global_names(), vec!["other", "var_4"]);

            store_global_slot(4, Value::Num(7.0));
            assert_eq!(global_slot_value(4), Some(Value::Num(7.0)));
            assert_eq!(global_value("other"), Some(Value::Num(3.0)));
            let roots = roots();
            assert!(roots.contains(&Value::Num(11.0)));
            assert!(roots.contains(&Value::Num(7.0)));
            let mut visited = Vec::new();
            assert_eq!(
                visit_roots(&mut |value| {
                    visited.push(value.clone());
                    Some(())
                }),
                Some(())
            );
            assert!(visited.contains(&Value::Num(11.0)));
            assert!(visited.contains(&Value::Num(7.0)));
        }));
    }
}
