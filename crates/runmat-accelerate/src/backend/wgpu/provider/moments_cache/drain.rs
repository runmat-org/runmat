//! Iterative physical-retirement callback drain. No cache lock is held here.
use std::cell::{Cell, RefCell};

use anyhow::{anyhow, Result};
use runmat_accelerate_api::allocation::GpuTensorHandle;

thread_local! {
    static PENDING_RELEASE: RefCell<Vec<GpuTensorHandle>> = const { RefCell::new(Vec::new()) };
    static DRAINING_RELEASE: Cell<bool> = const { Cell::new(false) };
}

/// A release callback can retire another cached output. Queue those outputs
/// on the current thread rather than nesting one physical retirement per edge.
pub(in super::super) fn drain_detached(handles: Vec<GpuTensorHandle>) -> Result<()> {
    PENDING_RELEASE.with(|pending| pending.borrow_mut().extend(handles));
    if DRAINING_RELEASE.with(|draining| draining.replace(true)) {
        return Ok(());
    }
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            DRAINING_RELEASE.with(|draining| draining.set(false));
        }
    }
    let _reset = Reset;
    let mut first_error = None;
    while let Some(handle) = PENDING_RELEASE.with(|pending| pending.borrow_mut().pop()) {
        if runmat_accelerate_api::allocation::catch_allocation_cleanup_unwind(|| drop(handle))
            .is_err()
        {
            first_error.get_or_insert_with(|| anyhow!("WGPU cached handle retirement panicked"));
        }
    }
    first_error.map_or(Ok(()), Err)
}
