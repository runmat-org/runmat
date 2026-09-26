use runmat_value::Value;

use super::CastError;

/// Concrete numeric representations have unconditional conversion semantics.
/// Symbolic inputs require value-dependent evaluation; they and unsupported
/// kinds retain the existing host conversion/error path before any service call.
/// This does not execute conversion or materialize data to admit preparation.
pub(crate) fn admits_preparation(value: &Value) -> Result<bool, CastError> {
    match value {
        Value::GpuTensor(handle) => {
            super::provider::admit_gpu_value(handle)?;
            Ok(true)
        }
        Value::SparseTensor(sparse) => {
            super::representations::admit_sparse_value(sparse)?;
            Ok(true)
        }
        Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::Tensor(_)
        | Value::LogicalArray(_)
        | Value::CharArray(_)
        | Value::Complex(_, _)
        | Value::ComplexTensor(_) => Ok(true),
        _ => Ok(false),
    }
}
