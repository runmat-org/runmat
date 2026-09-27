use crate::interpreter::dispatch::SequenceState;
use runmat_gc::{GcHandle, GcRoot, Trace, Tracer, ValueRoot};
use runmat_value::Value;
use std::ptr::NonNull;

pub(super) struct SequenceRoot(NonNull<SequenceState>);
impl SequenceRoot {
    /// # Safety
    /// The sequence owner must remain pinned until its root is unregistered;
    /// mutation and scans must be exclusive, just like StackRoot.
    pub unsafe fn new(sequence: &SequenceState) -> Self {
        Self(NonNull::from(sequence))
    }
}
impl ValueRoot for SequenceRoot {
    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        // SAFETY: the registration and frame guard enforce new's invariant.
        let mut result = Some(());
        unsafe { self.0.as_ref() }.visit_retained_values(&mut |value| {
            if result.is_some() {
                result = visitor(value);
            }
        });
        result
    }
}
impl GcRoot for SequenceRoot {
    fn scan(&self) -> Vec<GcHandle> {
        struct Collector(Vec<GcHandle>);
        impl Tracer for Collector {
            fn mark(&mut self, handle: GcHandle) {
                self.0.push(handle);
            }
        }
        let mut collector = Collector(Vec::new());
        // Infallible borrowed projection; ordinary GC scans never copy arrays.
        unsafe { self.0.as_ref() }.visit_retained_values(&mut |value| value.trace(&mut collector));
        collector.0
    }
    fn description(&self) -> String {
        "interpreter_sequence".into()
    }
}
