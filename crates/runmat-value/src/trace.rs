use crate::*;
use runmat_gc_api::{Trace, Tracer};

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

impl Trace for Closure {
    fn trace(&self, tracer: &mut dyn Tracer) {
        for value in &self.captures {
            value.trace(tracer);
        }
    }
}

impl Trace for ObjectInstance {
    fn trace(&self, tracer: &mut dyn Tracer) {
        for value in self.properties.values() {
            value.trace(tracer);
        }
        if let Some(dynamic_properties) = &self.dynamic_properties {
            for property in dynamic_properties.values() {
                if let Some(metadata_handle) = property.metadata_handle {
                    tracer.mark(metadata_handle);
                }
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

impl Trace for Value {
    fn trace(&self, tracer: &mut dyn Tracer) {
        match self {
            Value::Cell(cells) => cells.trace(tracer),
            Value::Struct(struct_value) => struct_value.trace(tracer),
            Value::HandleObject(handle) => handle.trace(tracer),
            Value::Listener(listener) => listener.trace(tracer),
            Value::Closure(closure) => closure.trace(tracer),
            Value::Object(object) => object.trace(tracer),
            Value::ObjectArray(array) => array.trace(tracer),
            Value::OutputList(values) => {
                for value in values {
                    value.trace(tracer);
                }
            }
            Value::Int(_)
            | Value::Num(_)
            | Value::Complex(_, _)
            | Value::Bool(_)
            | Value::LogicalArray(_)
            | Value::String(_)
            | Value::StringArray(_)
            | Value::CharArray(_)
            | Value::Tensor(_)
            | Value::SparseTensor(_)
            | Value::ComplexTensor(_)
            | Value::Symbolic(_)
            | Value::SymbolicArray(_)
            | Value::GpuTensor(_)
            | Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle { .. }
            | Value::ClassRef(_)
            | Value::MException(_)
            | Value::Future(_)
            | Value::Task(_)
            | Value::Pool(_)
            | Value::Job(_)
            | Value::Foreign(_) => {}
        }
    }
}
