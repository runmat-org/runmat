use runmat_thread_local::runmat_thread_local;
use runmat_value::Value;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

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

/// Named session variables and legacy bytecode-keyed persistent storage.
///
/// Slot-to-local synchronization belongs to the executor. This state owns only
/// values whose lifetime and identity cross an individual VM frame.
#[derive(Debug, Default, Clone)]
pub struct SessionVariableState {
    globals: HashMap<String, Value>,
    global_revision: GlobalRevision,
    persistent_slots: HashMap<(String, usize), Value>,
    persistent_names: HashMap<(String, String), Value>,
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
    with_state(|state| {
        let mut names = state
            .globals
            .keys()
            .filter(|name| !name.starts_with("var_"))
            .cloned()
            .collect::<Vec<_>>();
        names.sort();
        names
    })
}

pub fn global_value(name: &str) -> Option<Value> {
    with_state(|state| state.globals.get(name).cloned())
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
    with_state_mut(|state| {
        state.globals.insert(name.to_string(), value);
        state.global_revision = GlobalRevision::default();
    });
}

pub fn roots() -> Vec<Value> {
    with_state(|state| {
        state
            .globals
            .values()
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

pub fn update_global_slot(index: usize, alias: Option<&str>, value: &Value) {
    with_state_mut(|state| {
        let mut changed = false;
        let slot_key = format!("var_{index}");
        if let std::collections::hash_map::Entry::Occupied(mut entry) =
            state.globals.entry(slot_key)
        {
            entry.insert(value.clone());
            changed = true;
        }
        if let Some(alias) = alias {
            state.globals.insert(alias.to_string(), value.clone());
            changed = true;
        }
        if changed {
            state.global_revision = GlobalRevision::default();
        }
    });
}

pub fn global_slot_value(index: usize) -> Option<Value> {
    global_value(&format!("var_{index}"))
}

pub fn bind_global_slot(index: usize, name: &str) {
    with_state_mut(|state| {
        if let Some(value) = state.globals.get(name).cloned() {
            state.globals.insert(format!("var_{index}"), value);
            state.global_revision = GlobalRevision::default();
        }
    });
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
            update_global_slot(0, Some("answer"), &Value::Num(42.0));
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
            update_global_slot(4, Some("answer"), &Value::Num(2.0));
            let second = global_revision();
            assert_ne!(first, second);
            bind_global_slot(4, "answer");
            let third = global_revision();
            assert_ne!(second, third);
            assert_ne!(initial, third);
            assert_eq!(global_value("answer"), Some(Value::Num(2.0)));
        }));
    }
}
