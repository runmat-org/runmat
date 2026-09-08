use runmat_value::{
    CharArray, ComplexTensor, LogicalArray, StringArray, StructValue, Tensor, Value,
};

use super::error;

pub(super) fn empty_value(prototype: Option<&Value>) -> crate::BuiltinResult<Value> {
    match prototype {
        Some(Value::LogicalArray(_) | Value::Bool(_)) => LogicalArray::new(Vec::new(), vec![0, 0])
            .map(Value::LogicalArray)
            .map_err(error::internal),
        Some(Value::ComplexTensor(_) | Value::Complex(_, _)) => {
            ComplexTensor::new(Vec::new(), vec![0, 0])
                .map(Value::ComplexTensor)
                .map_err(error::internal)
        }
        Some(Value::String(_)) => Ok(Value::String(String::new())),
        Some(Value::StringArray(_)) => StringArray::new(Vec::new(), vec![0, 0])
            .map(Value::StringArray)
            .map_err(error::internal),
        Some(Value::CharArray(_)) => CharArray::new(Vec::new(), 0, 0)
            .map(Value::CharArray)
            .map_err(error::internal),
        Some(Value::Cell(_)) => {
            crate::make_cell_with_shape(Vec::new(), vec![0, 0]).map_err(error::internal)
        }
        Some(Value::Struct(_)) => Ok(Value::Struct(StructValue::new())),
        _ => empty_double(),
    }
}

fn empty_double() -> crate::BuiltinResult<Value> {
    Tensor::new(Vec::new(), vec![0, 0])
        .map(Value::Tensor)
        .map_err(error::internal)
}
