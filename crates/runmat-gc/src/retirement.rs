//! Detached heap payloads awaiting destruction outside collector locks.

use crate::{
    allocator::GcAllocationKind, lexical_activation::LexicalActivation, GcFinalizer, GcHandle,
};
use runmat_value::Value;
use std::sync::Arc;

pub struct RetiredValue {
    _payload: RetiredPayload,
    finalizer: Option<Arc<dyn GcFinalizer>>,
}

enum RetiredPayload {
    Value { _value: Value },
    LexicalActivation { _activation: LexicalActivation },
}

impl RetiredValue {
    /// The allocator must have removed this initialized slot from its live set.
    /// Its memory may be reclaimed once the payload has been moved here.
    pub(crate) unsafe fn take(
        ptr: *const u8,
        handle: Option<GcHandle>,
        kind: GcAllocationKind,
    ) -> Self {
        Self {
            // SAFETY: the allocator's live kind identifies the initialized
            // payload at this slot and the caller transfers it exactly once.
            _payload: match kind {
                GcAllocationKind::Value => RetiredPayload::Value {
                    _value: unsafe { ptr.cast::<Value>().read() },
                },
                GcAllocationKind::LexicalActivation => RetiredPayload::LexicalActivation {
                    _activation: unsafe { ptr.cast::<LexicalActivation>().read() },
                },
            },
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
