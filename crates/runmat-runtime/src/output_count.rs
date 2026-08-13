use runmat_thread_local::runmat_thread_local;
use runmat_value::Value;
use std::cell::RefCell;

runmat_thread_local! {
    static OUTPUT_COUNT_STACK: RefCell<Vec<Option<usize>>> = const { RefCell::new(Vec::new()) };
}

pub struct OutputCountGuard {
    did_push: bool,
    state: Option<std::rc::Rc<crate::context::RuntimeContextState>>,
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
    if let Some(state) = active_state() {
        return state
            .output
            .borrow()
            .requested_outputs
            .last()
            .copied()
            .flatten();
    }
    OUTPUT_COUNT_STACK.with(|stack| stack.borrow().last().cloned().flatten())
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
