use runmat_thread_local::runmat_thread_local;
use runmat_value::Value;
use std::cell::RefCell;
use std::collections::HashMap;

/// Named session variables and legacy bytecode-keyed persistent storage.
///
/// Slot-to-local synchronization belongs to the executor. This state owns only
/// values whose lifetime and identity cross an individual VM frame.
#[derive(Debug, Default)]
pub struct SessionVariableState {
    globals: HashMap<String, Value>,
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

pub fn update_global_slot(index: usize, alias: Option<&str>, value: &Value) {
    with_state_mut(|state| {
        let slot_key = format!("var_{index}");
        if state.globals.contains_key(&slot_key) {
            state.globals.insert(slot_key, value.clone());
        }
        if let Some(alias) = alias {
            state.globals.insert(alias.to_string(), value.clone());
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
}
