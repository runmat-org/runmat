//! Typed numeric decoding, with the invocation's original gather authority.
use super::*;
use crate::builtins::common::tensor;
pub(super) enum CompatibilityRole {
    Data,
    Control,
}
pub(super) fn preflight(
    value: &Value,
    integer: &runmat_builtins::BuiltinExtensionDescriptor,
    runtime: Option<&crate::context::RuntimeContext>,
    role: CompatibilityRole,
) -> BuiltinResult<()> {
    let extension = |descriptor: &runmat_builtins::BuiltinExtensionDescriptor| {
        crate::compatibility::ensure_builtin_extension_enabled_in_optional_context(
            runtime, descriptor, NAME,
        )
    };
    if !runmat_builtins::OptimNumericInput::OptionalSparseMatrix
        .accepts_metadata(&crate::value_fact::value_fact(value))
    {
        return Err(invalid(
            "expected real numeric data, not logical, complex or object data",
        ));
    }
    crate::builtins::math::optim::common::preflight_host_residency(
        NAME,
        value,
        Some(&runmat_builtins::QUADPROG_GPU_INPUT_EXTENSION),
        runtime,
    )?;
    if crate::builtins::common::validation::value_has_native_integer_class(value) {
        extension(integer)?;
    }
    let class=crate::builtins::introspection::underlying_type::underlying_class_identity_for_value_checked(value)?;
    if matches!(role, CompatibilityRole::Data) && class == runmat_types::standard::SINGLE {
        extension(&runmat_builtins::QUADPROG_SINGLE_EXTENSION)?;
    }
    Ok(())
}

pub(super) async fn gather(value: Value, cooperation: &ScalarCooperation) -> BuiltinResult<Value> {
    crate::dispatcher::gather_if_needed_in_context_async(&value, cooperation.runtime()).await
}
pub(super) async fn matrix(
    value: Value,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Matrix>> {
    let value = gather(value, cooperation).await?;
    conic::matrix::parse(
        value,
        conic::matrix::IntegerPolicy::Exact,
        cooperation,
        |detail: &str| invalid(detail),
    )
    .await
}
/// Matrix arguments f/lb/ub are flattened in their original column-major order.
/// RHS and x0 retain their public vector restriction.
pub(super) async fn vector(
    value: Value,
    bounds: bool,
    array: bool,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    let value = gather(value, cooperation).await?;
    if !cooperation.integer_boundary(&value).await?.is_exact() {
        return Err(invalid(
            "integer data must be exactly representable as binary64",
        ));
    }
    let values = match value {
        Value::Num(value) => vec![value],
        Value::Int(value) => vec![value.to_f64()],
        Value::Tensor(value)
            if value.shape().len() <= 2
                && (array || value.rows() == 1 || value.cols() == 1 || value.is_empty()) =>
        {
            tensor::tensor_into_values_f64(value)
        }
        Value::SparseTensor(value) => {
            let matrix = conic::matrix::parse(
                Value::SparseTensor(value),
                conic::matrix::IntegerPolicy::Exact,
                cooperation,
                |detail: &str| invalid(detail),
            )
            .await?;
            let Some(matrix) = matrix else {
                return Ok(Vec::new());
            };
            if !array && matrix.m != 1 && matrix.n != 1 {
                return Err(invalid("expected a numeric vector"));
            }
            let length = matrix
                .m
                .checked_mul(matrix.n)
                .ok_or_else(|| invalid("vector size overflow"))?;
            let mut values = vec![0.0; length];
            for col in 0..matrix.n {
                for index in matrix.colptr[col]..matrix.colptr[col + 1] {
                    values[col * matrix.m + matrix.rowval[index]] = matrix.nzval[index];
                }
                cooperation
                    .completed((matrix.colptr[col + 1] - matrix.colptr[col]) as u64 + 1)
                    .await?;
            }
            values
        }
        _ => {
            return Err(invalid(
                "expected real numeric data (not logical, complex, or object data)",
            ))
        }
    };
    for chunk in values.chunks(1024) {
        if chunk
            .iter()
            .any(|v| if bounds { v.is_nan() } else { !v.is_finite() })
        {
            return Err(invalid(if bounds {
                "bounds cannot contain NaN"
            } else {
                "numeric data must be finite"
            }));
        }
        cooperation.completed(chunk.len() as u64).await?;
    }
    Ok(values)
}
