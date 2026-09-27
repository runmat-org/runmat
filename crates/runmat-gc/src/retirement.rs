//! Detached heap payloads awaiting destruction outside collector locks.

use crate::{GcFinalizer, GcHandle};
use runmat_value::Value;
use std::sync::Arc;

pub struct RetiredValue {
    _value: Value,
    finalizer: Option<Arc<dyn GcFinalizer>>,
}

impl RetiredValue {
    /// The allocator must have removed this initialized slot from its live set.
    /// Its memory may be reclaimed once the payload has been moved here.
    pub(crate) unsafe fn take(ptr: *const Value, handle: Option<GcHandle>) -> Self {
        Self {
            // SAFETY: the caller transfers the initialized slot exactly once.
            _value: unsafe { ptr.read() },
            finalizer: handle.and_then(|handle| crate::FINALIZERS.lock().remove(&handle.into())),
        }
    }
}

impl Drop for RetiredValue {
    fn drop(&mut self) {
        if let Some(finalizer) = self.finalizer.take() {
            finalizer.finalize();
        }
        // The owned Value is dropped next, releasing any allocation leases.
    }
}
