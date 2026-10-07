use runmat_builtins::SQRT_ERROR_INVALID_INPUT;
use runmat_value::Value;

use super::{complex, errors, host, provider};
use crate::BuiltinResult;

pub(super) async fn evaluate_admitted(value: Value) -> BuiltinResult<Value> {
    crate::builtins::common::validation::reject_typed_complex_integer_for(
        &value,
        runmat_builtins::SQRT_CATALOG_ENTRY.identity,
        &SQRT_ERROR_INVALID_INPUT,
    )?;
    match value {
        Value::GpuTensor(handle) => provider::evaluate(handle).await,
        Value::Complex(real, imag) => Ok(complex::evaluate_scalar(real, imag)),
        Value::ComplexTensor(tensor) => complex::evaluate_tensor(tensor),
        Value::CharArray(chars) => host::evaluate_characters(chars),
        Value::String(_) | Value::StringArray(_) => Err(errors::with_detail(
            &SQRT_ERROR_INVALID_INPUT,
            "expected numeric input",
        )),
        other => host::evaluate(other),
    }
}
