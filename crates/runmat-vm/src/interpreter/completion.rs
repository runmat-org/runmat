//! Ownership while a complete VM frame is projected to a language result.
use runmat_runtime::context::RuntimeContext;
use runmat_value::Value;

/// Own the dense returned frame until all fallible output/capture projection
/// succeeds. This does not change the full-frame continuation ABI. Successful
/// consumers adopt their exact result aliases with `finish`; error, cancellation,
/// and unwind retire unadopted residency through the same VM root authority.
#[must_use = "finish projection with the values adopted by the result owner"]
pub struct CompletedFrameProjection {
    values: Vec<Value>,
    #[cfg(feature = "native-accel")]
    ownership: Option<super::resident_ownership::ConsumedValues>,
    #[cfg(feature = "native-accel")]
    runtime: RuntimeContext,
}

impl CompletedFrameProjection {
    /// Transfer ownership of a completed dense VM frame, not a borrowed caller
    /// workspace. Capturing ownership copies only resource identities.
    pub fn new(values: Vec<Value>, runtime: RuntimeContext) -> Self {
        #[cfg(feature = "native-accel")]
        let ownership = {
            let _context = runtime.enter();
            Some(super::resident_ownership::ConsumedValues::new(&values))
        };
        #[cfg(not(feature = "native-accel"))]
        let _ = runtime;
        Self {
            values,
            #[cfg(feature = "native-accel")]
            ownership,
            #[cfg(feature = "native-accel")]
            runtime,
        }
    }

    pub fn values(&self) -> &[Value] {
        &self.values
    }

    /// Call after every fallible projection step. The iterator borrows outputs
    /// and captures, avoiding an aggregate copy of their host payloads.
    pub fn finish<'a>(self, adopted: impl IntoIterator<Item = &'a Value>) {
        #[cfg(feature = "native-accel")]
        {
            let mut projection = self;
            for value in adopted {
                if let Some(ownership) = &mut projection.ownership {
                    ownership.protect(std::slice::from_ref(value));
                }
            }
        }
        #[cfg(not(feature = "native-accel"))]
        let _ = adopted;
        // Drop performs retirement while the originating context is active.
    }
}

impl Drop for CompletedFrameProjection {
    fn drop(&mut self) {
        #[cfg(feature = "native-accel")]
        {
            let _context = self.runtime.enter();
            drop(self.ownership.take());
        }
    }
}
