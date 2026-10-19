//! Real-square-root representation dispatch after input admission.

use runmat_builtins::REALSQRT_ERROR_INVALID_INPUT;
use runmat_value::Value;

use super::{errors, host, provider};
use crate::BuiltinResult;

pub(super) async fn evaluate_admitted(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => provider::evaluate(handle).await,
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(errors::with_detail(
            &REALSQRT_ERROR_INVALID_INPUT,
            "complex input is not supported",
        )),
        Value::SparseTensor(sparse) => host::evaluate_sparse(sparse),
        Value::Int(_)
        | Value::Bool(_)
        | Value::LogicalArray(_)
        | Value::CharArray(_)
        | Value::String(_)
        | Value::StringArray(_) => Err(errors::with_detail(
            &REALSQRT_ERROR_INVALID_INPUT,
            "expected real single or double input",
        )),
        other => host::evaluate(other),
    }
}
