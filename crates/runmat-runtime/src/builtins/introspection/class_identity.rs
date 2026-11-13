//! Typed runtime class identity shared by reflection, validation and metadata
//! consumers. No query materializes resident payloads or changes ownership.

use runmat_types::{standard, ClassIdentity};
use runmat_value::Value;

/// Presentation boundary for a runtime value's canonical class identity.
pub(crate) fn class_name_for_value(value: &Value) -> String {
    class_identity_for_value(value).display_name().to_owned()
}

pub(crate) fn class_identity_for_value(value: &Value) -> ClassIdentity {
    match value {
        Value::Num(_) | Value::Complex(_, _) => standard::DOUBLE.owned(),
        Value::ComplexTensor(tensor) => tensor.numeric_dtype().class_identity(),
        Value::Tensor(tensor) => tensor.numeric_dtype().class_identity(),
        Value::SparseTensor(sparse) => ClassIdentity::from(sparse.class_name()),
        Value::Int(iv) => ClassIdentity::from(iv.class_name()),
        Value::Bool(_) | Value::LogicalArray(_) => standard::LOGICAL.owned(),
        Value::String(_) | Value::StringArray(_) => standard::STRING.owned(),
        Value::CharArray(_) => standard::CHAR.owned(),
        Value::Symbolic(_) | Value::SymbolicArray(_) => standard::SYMBOLIC.owned(),
        Value::Cell(_) => standard::CELL.owned(),
        Value::Struct(_) | Value::StructArray(_) => standard::STRUCT.owned(),
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::handle_is_explicit(handle) {
                standard::GPU_ARRAY.owned()
            } else if runmat_accelerate_api::handle_is_logical(handle) {
                standard::LOGICAL.owned()
            } else {
                // The owner-admitted physical element type is authoritative;
                // an absent preferred precision must not invent double.
                handle.physical().element_type().class_identity()
            }
        }
        Value::ScopedFunctionHandle(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle(_)
        | Value::Closure(_) => standard::FUNCTION_HANDLE.owned(),
        Value::HandleObject(handle) => handle.class_name.as_identity().clone(),
        Value::Listener(_) => standard::EVENT_LISTENER.owned(),
        Value::ObjectArray(array) => array.class_name().clone(),
        Value::Object(object) => object.class_name().clone(),
        Value::ClassRef(_) => standard::META_CLASS.owned(),
        Value::MException(_) => standard::MEXCEPTION.owned(),
        Value::OutputList(_) => ClassIdentity::from("output_list"),
        Value::Future(_) => ClassIdentity::from("parallel.Future"),
        Value::Task(_) => ClassIdentity::from("parallel.Task"),
        Value::Pool(_) => ClassIdentity::from("parallel.Pool"),
        Value::Job(_) => ClassIdentity::from("parallel.Job"),
        Value::Distributed(_) => ClassIdentity::from("distributed"),
        Value::Composite(_) => ClassIdentity::from("Composite"),
        Value::Foreign(reference) => ClassIdentity::from(reference.type_identity.name.clone()),
    }
}
