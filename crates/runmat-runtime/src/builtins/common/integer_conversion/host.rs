use runmat_types::IntegerClass;
use runmat_value::{Tensor, Value};

use super::class::IntegerClassExt;
use super::error::{CastError, UnsupportedValueKind};
use super::representations::{
    cast_complex_value, cast_sparse_value, cast_symbolic_array, cast_tensor_value,
};

pub(super) fn cast_host_value(value: Value, target: IntegerClass) -> Result<Value, CastError> {
    match value {
        Value::Num(value) => Ok(Value::Int(target.cast_scalar(value))),
        Value::Int(value) => Ok(Value::Int(target.cast_int(&value))),
        Value::Bool(value) => Ok(Value::Int(target.cast_scalar(f64::from(value)))),
        Value::Tensor(tensor) => cast_tensor_value(target, tensor),
        Value::SparseTensor(sparse) => cast_sparse_value(target, sparse),
        Value::LogicalArray(array) => {
            let tensor = crate::builtins::common::tensor::logical_to_tensor(&array)
                .map_err(CastError::Internal)?;
            cast_tensor_value(target, tensor)
        }
        Value::CharArray(chars) => {
            let tensor = Tensor::new(
                chars
                    .data
                    .iter()
                    .map(|&value| value as u32 as f64)
                    .collect(),
                vec![chars.rows, chars.cols],
            )
            .map_err(CastError::Internal)?;
            cast_tensor_value(target, tensor)
        }
        value @ (Value::Complex(_, _) | Value::ComplexTensor(_)) => {
            cast_complex_value(value, target)
        }
        Value::String(_) | Value::StringArray(_) => {
            Err(CastError::Unsupported(UnsupportedValueKind::Text))
        }
        Value::Symbolic(expression) => expression
            .numeric_constant_value()
            .map(|value| Value::Int(target.cast_scalar(value)))
            .ok_or(CastError::Unsupported(UnsupportedValueKind::Symbolic)),
        Value::SymbolicArray(array) => cast_symbolic_array(target, array),
        Value::Cell(_) => Err(CastError::Unsupported(UnsupportedValueKind::Cell)),
        Value::Struct(_) | Value::StructArray(_) => {
            Err(CastError::Unsupported(UnsupportedValueKind::Struct))
        }
        Value::ObjectArray(array) => Err(CastError::Unsupported(UnsupportedValueKind::Object(
            array.class_name().to_string(),
        ))),
        Value::Object(object) => Err(CastError::Unsupported(UnsupportedValueKind::Object(
            object.class_name.to_string(),
        ))),
        Value::HandleObject(handle) => Err(CastError::Unsupported(UnsupportedValueKind::Object(
            handle.class_name.to_string(),
        ))),
        Value::Listener(_) => Err(CastError::Unsupported(UnsupportedValueKind::Listener)),
        Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_) => Err(CastError::Unsupported(UnsupportedValueKind::FunctionHandle)),
        Value::ClassRef(_) => Err(CastError::Unsupported(UnsupportedValueKind::ClassReference)),
        Value::MException(_)
        | Value::Future(_)
        | Value::Task(_)
        | Value::Pool(_)
        | Value::Job(_)
        | Value::Distributed(_)
        | Value::Composite(_) => Err(CastError::Unsupported(UnsupportedValueKind::RuntimeHandle)),
        Value::Foreign(_) => Err(CastError::Unsupported(UnsupportedValueKind::Foreign)),
        Value::OutputList(_) => Err(CastError::Unsupported(UnsupportedValueKind::OutputList)),
        Value::GpuTensor(_) => Err(CastError::Internal(
            "resident integer conversion bypassed provider dispatch".into(),
        )),
    }
}
