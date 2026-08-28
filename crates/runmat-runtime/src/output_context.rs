use runmat_thread_local::runmat_thread_local;
use std::cell::RefCell;

runmat_thread_local! {
    static REQUESTED_OUTPUTS: RefCell<Option<usize>> = const { RefCell::new(None) };
}

pub struct OutputCountGuard {
    prev: Option<usize>,
    state: Option<std::rc::Rc<crate::context::RuntimeContextState>>,
}

impl Drop for OutputCountGuard {
    fn drop(&mut self) {
        if let Some(state) = &self.state {
            let mut output = state.output.borrow_mut();
            output.presentation_outputs.pop();
            if let Some(previous) = self.prev {
                output.presentation_outputs.push(previous);
            }
        } else {
            REQUESTED_OUTPUTS.with(|cell| {
                *cell.borrow_mut() = self.prev;
            });
        }
    }
}

pub fn push_output_count(count: usize) -> OutputCountGuard {
    if let Some(state) = active_state() {
        let prev = state.output.borrow_mut().presentation_outputs.pop();
        state.output.borrow_mut().presentation_outputs.push(count);
        return OutputCountGuard {
            prev,
            state: Some(state),
        };
    }
    let prev = REQUESTED_OUTPUTS.with(|cell| {
        let mut guard = cell.borrow_mut();
        let prev = guard.take();
        *guard = Some(count);
        prev
    });
    OutputCountGuard { prev, state: None }
}

pub fn requested_output_count() -> Option<usize> {
    if let Some(state) = active_state() {
        return state.output.borrow().presentation_outputs.last().copied();
    }
    REQUESTED_OUTPUTS.with(|cell| *cell.borrow())
}

fn active_state() -> Option<std::rc::Rc<crate::context::RuntimeContextState>> {
    crate::context::legacy::active().map(|context| std::rc::Rc::clone(context.state()))
}
