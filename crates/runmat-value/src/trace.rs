use crate::*;
use runmat_gc_api::{Trace, Tracer};

mod value;
pub use value::ValueTraceWorkspace;

impl Trace for CellArray {
    fn trace(&self, tracer: &mut dyn Tracer) {
        for value in &self.data {
            value.trace(tracer);
        }
    }
}

impl Trace for StructValue {
    fn trace(&self, tracer: &mut dyn Tracer) {
        for value in self.fields.values() {
            value.trace(tracer);
        }
    }
}

impl Trace for StructArray {
    fn trace(&self, tracer: &mut dyn Tracer) {
        self.for_each_value(|value| value.trace(tracer));
    }
}

impl Trace for Closure {
    fn trace(&self, tracer: &mut dyn Tracer) {
        for value in &self.captures {
            value.trace(tracer);
        }
        if let Some(environment) = &self.lexical_environment {
            environment.for_each_handle(|handle| tracer.mark(handle));
        }
    }
}

impl Trace for ObjectInstance {
    fn trace(&self, tracer: &mut dyn Tracer) {
        self.trace_native_payload(tracer);
        for value in self.properties().values() {
            value.trace(tracer);
        }
        for property in self.trace_dynamic_property_values() {
            if let Some(metadata_handle) = property.metadata_handle {
                tracer.mark(metadata_handle);
            }
        }
    }
}

impl Trace for HandleRef {
    fn trace(&self, tracer: &mut dyn Tracer) {
        tracer.mark(self.target);
    }
}

impl Trace for Listener {
    fn trace(&self, tracer: &mut dyn Tracer) {
        tracer.mark(self.target);
        tracer.mark(self.callback);
    }
}

impl Trace for ValueSequence {
    fn trace(&self, tracer: &mut dyn Tracer) {
        for value in self.iter() {
            value.trace(tracer);
        }
    }
}

#[cfg(test)]
mod tests;
