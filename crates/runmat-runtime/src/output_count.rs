use runmat_builtins::Value;
use runmat_thread_local::runmat_thread_local;
use std::cell::RefCell;

runmat_thread_local! {
    static OUTPUT_COUNT_STACK: RefCell<Vec<Option<usize>>> = const { RefCell::new(Vec::new()) };
}

pub struct OutputCountGuard {
    did_push: bool,
}

impl Drop for OutputCountGuard {
    fn drop(&mut self) {
        if !self.did_push {
            return;
        }
        OUTPUT_COUNT_STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            let _ = stack.pop();
        });
    }
}

pub fn push_output_count(count: Option<usize>) -> OutputCountGuard {
    OUTPUT_COUNT_STACK.with(|stack| {
        stack.borrow_mut().push(count);
    });
    OutputCountGuard { did_push: true }
}

pub fn current_output_count() -> Option<usize> {
    OUTPUT_COUNT_STACK.with(|stack| stack.borrow().last().cloned().flatten())
}

pub fn output_list_with_padding(out_count: usize, mut outputs: Vec<Value>) -> Value {
    if outputs.len() < out_count {
        outputs.resize(out_count, Value::Num(0.0));
    }
    Value::OutputList(outputs)
}
