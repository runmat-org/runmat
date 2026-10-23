use crate::GcHandle;

/// Receives GC handles discovered while tracing an owned value graph.
pub trait Tracer {
    fn mark(&mut self, handle: GcHandle);
}

/// Describes how a type exposes nested GC handles.
pub trait Trace {
    fn trace(&self, tracer: &mut dyn Tracer);
}

impl<T: Trace> Trace for Option<T> {
    fn trace(&self, tracer: &mut dyn Tracer) {
        if let Some(value) = self {
            value.trace(tracer);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::{Trace, Tracer};

    #[test]
    fn optional_values_delegate_to_the_present_owner() {
        struct Probe<'a>(&'a Cell<usize>);
        impl Trace for Probe<'_> {
            fn trace(&self, _: &mut dyn Tracer) {
                self.0.set(self.0.get() + 1);
            }
        }
        struct NoHandles;
        impl Tracer for NoHandles {
            fn mark(&mut self, _: crate::GcHandle) {
                panic!("the probe contains no GC handles");
            }
        }
        let visits = Cell::new(0);
        for value in [None, Some(Probe(&visits))] {
            value.trace(&mut NoHandles);
        }
        assert_eq!(visits.get(), 1);
    }
}
