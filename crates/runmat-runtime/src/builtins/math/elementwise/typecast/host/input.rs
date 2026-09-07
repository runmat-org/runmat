use runmat_builtins::TYPECAST_ERROR_INVALID_INPUT;
use runmat_value::Value;

use crate::BuiltinResult;

use super::super::error;

pub(super) fn shape(source: &Value) -> BuiltinResult<Vec<usize>> {
    match source {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => Ok(vec![1, 1]),
        Value::Tensor(tensor) => Ok(tensor.shape.clone()),
        Value::ComplexTensor(tensor) => Ok(tensor.shape.clone()),
        Value::LogicalArray(array) => Ok(array.shape.clone()),
        Value::SparseTensor(_) => Err(error::build(
            &TYPECAST_ERROR_INVALID_INPUT,
            "input must be full, not sparse",
        )),
        _ => Err(error::build(
            &TYPECAST_ERROR_INVALID_INPUT,
            "input must be numeric or logical",
        )),
    }
}

pub(super) fn validate_vector(shape: &[usize]) -> BuiltinResult<()> {
    if shape.iter().filter(|&&dimension| dimension > 1).count() > 1 {
        return Err(error::build(
            &TYPECAST_ERROR_INVALID_INPUT,
            "input must be a scalar or vector",
        ));
    }
    Ok(())
}
