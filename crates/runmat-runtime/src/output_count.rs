use runmat_thread_local::runmat_thread_local;
use runmat_value::Value;
use std::cell::RefCell;

mod validation;
pub(crate) use validation::validate_fixed_outputs;

runmat_thread_local! {
    static OUTPUT_COUNT_STACK: RefCell<Vec<Option<usize>>> = const { RefCell::new(Vec::new()) };
}

pub struct OutputCountGuard {
    did_push: bool,
    state: Option<std::rc::Rc<crate::context::RuntimeContextState>>,
}

/// Output request scoped to the current runtime builtin dispatch. `Implicit`
/// is deliberately distinct from `Unscoped`: a nested call that explicitly
/// requests no count must not inherit an enclosing presentation count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InvocationOutputScope {
    Unscoped,
    Implicit,
    Requested(usize),
}

impl Drop for OutputCountGuard {
    fn drop(&mut self) {
        if !self.did_push {
            return;
        }
        if let Some(state) = &self.state {
            state.output.borrow_mut().requested_outputs.pop();
        } else {
            OUTPUT_COUNT_STACK.with(|stack| {
                let mut stack = stack.borrow_mut();
                let _ = stack.pop();
            });
        }
    }
}

pub fn push_output_count(count: Option<usize>) -> OutputCountGuard {
    if let Some(state) = active_state() {
        state.output.borrow_mut().requested_outputs.push(count);
        OutputCountGuard {
            did_push: true,
            state: Some(state),
        }
    } else {
        OUTPUT_COUNT_STACK.with(|stack| stack.borrow_mut().push(count));
        OutputCountGuard {
            did_push: true,
            state: None,
        }
    }
}

pub fn current_output_count() -> Option<usize> {
    match current_output_scope() {
        InvocationOutputScope::Requested(count) => Some(count),
        InvocationOutputScope::Unscoped | InvocationOutputScope::Implicit => None,
    }
}

pub fn current_output_scope() -> InvocationOutputScope {
    if let Some(state) = active_state() {
        return match state.output.borrow().requested_outputs.last().copied() {
            Some(Some(count)) => InvocationOutputScope::Requested(count),
            Some(None) => InvocationOutputScope::Implicit,
            None => InvocationOutputScope::Unscoped,
        };
    }
    OUTPUT_COUNT_STACK.with(|stack| match stack.borrow().last().copied() {
        Some(Some(count)) => InvocationOutputScope::Requested(count),
        Some(None) => InvocationOutputScope::Implicit,
        None => InvocationOutputScope::Unscoped,
    })
}

fn active_state() -> Option<std::rc::Rc<crate::context::RuntimeContextState>> {
    crate::context::legacy::active().map(|context| std::rc::Rc::clone(context.state()))
}

pub fn output_list_with_padding(out_count: usize, mut outputs: Vec<Value>) -> Value {
    if outputs.len() > out_count {
        outputs.truncate(out_count);
    }
    if outputs.len() < out_count {
        outputs.resize(out_count, Value::Num(0.0));
    }
    Value::OutputList(outputs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn nested_implicit_scope_does_not_inherit_outer_requested_count() {
        assert_eq!(current_output_scope(), InvocationOutputScope::Unscoped);
        let outer = push_output_count(Some(0));
        assert_eq!(current_output_scope(), InvocationOutputScope::Requested(0));
        {
            let _inner = push_output_count(None);
            assert_eq!(current_output_scope(), InvocationOutputScope::Implicit);
            assert_eq!(current_output_count(), None);
        }
        assert_eq!(current_output_scope(), InvocationOutputScope::Requested(0));
        drop(outer);
        assert_eq!(current_output_scope(), InvocationOutputScope::Unscoped);
    }
}
