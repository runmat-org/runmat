use runmat_thread_local::runmat_thread_local;
use std::cell::RefCell;

runmat_thread_local! {
    static REQUESTED_OUTPUTS: RefCell<Option<usize>> = const { RefCell::new(None) };
}

pub struct OutputCountGuard {
    prev: Option<usize>,
}

impl Drop for OutputCountGuard {
    fn drop(&mut self) {
        REQUESTED_OUTPUTS.with(|cell| {
            *cell.borrow_mut() = self.prev;
        });
    }
}

pub fn push_output_count(count: usize) -> OutputCountGuard {
    let prev = REQUESTED_OUTPUTS.with(|cell| {
        let mut guard = cell.borrow_mut();
        let prev = guard.take();
        *guard = Some(count);
        prev
    });
    OutputCountGuard { prev }
}

pub fn requested_output_count() -> Option<usize> {
    REQUESTED_OUTPUTS.with(|cell| *cell.borrow())
}
