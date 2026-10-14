use crate::*;
use runmat_gc_api::{Trace, Tracer};

impl Trace for Value {
    fn trace(&self, tracer: &mut dyn Tracer) {
        enum Step<'a> {
            Value(&'a Value),
            Mark(runmat_gc_api::GcHandle),
        }

        let mut pending = Vec::new();
        let mut first = Some(Step::Value(self));
        while let Some(step) = first.take().or_else(|| pending.pop()) {
            let value = match step {
                Step::Value(value) => value,
                Step::Mark(handle) => {
                    tracer.mark(handle);
                    continue;
                }
            };
            match value {
                Value::Cell(cells) => {
                    pending.extend(cells.data.iter().rev().map(Step::Value));
                }
                Value::Struct(struct_value) => {
                    pending.extend(struct_value.fields.values().rev().map(Step::Value));
                }
                Value::StructArray(array) => {
                    for name in array.field_names().collect::<Vec<_>>().into_iter().rev() {
                        if let Some(values) = array.field_values(name) {
                            pending.extend(values.iter().rev().map(Step::Value));
                        }
                    }
                }
                Value::HandleObject(handle) => tracer.mark(handle.target),
                Value::Listener(listener) => {
                    tracer.mark(listener.target);
                    tracer.mark(listener.callback);
                }
                Value::Closure(closure) => {
                    pending.extend(closure.captures.iter().rev().map(Step::Value));
                    if let Some(environment) = &closure.lexical_environment {
                        environment.for_each_handle(|handle| pending.push(Step::Mark(handle)));
                    }
                }
                Value::Object(object) => {
                    let metadata = object
                        .dynamic_property_definitions()
                        .filter_map(|(_, property)| property.metadata_handle)
                        .collect::<Vec<_>>();
                    pending.extend(metadata.into_iter().rev().map(Step::Mark));
                    pending.extend(
                        object
                            .properties
                            .values()
                            .collect::<Vec<_>>()
                            .into_iter()
                            .rev()
                            .map(Step::Value),
                    );
                    let mut native = Vec::new();
                    object.for_each_native_value(|value| native.push(value));
                    pending.extend(native.into_iter().rev().map(Step::Value));
                }
                Value::ObjectArray(array) => {
                    pending.extend(array.data().iter().rev().map(Step::Value));
                }
                Value::OutputList(values) => {
                    pending.extend(values.iter().rev().map(Step::Value));
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
                | Value::BoundFunctionHandle(_)
                | Value::ClassRef(_)
                | Value::MException(_)
                | Value::Future(_)
                | Value::Task(_)
                | Value::Pool(_)
                | Value::Job(_)
                | Value::Distributed(_)
                | Value::Composite(_)
                | Value::Foreign(_) => {}
            }
        }
    }
}
