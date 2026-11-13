//! Exact broadcast comparisons and range validation.

use super::{
    handle_integer_type, handle_is_logical, host_value, invalid_argument_error, tensor,
    try_complex_ordering_comparison, try_real_ordering_comparison, value_is_numeric_or_logical,
    BuiltinResult, IntegerComparisonError, IntegerComparisonOp, RangeInclusivity, SparseTensor,
    Value,
};
use crate::builtins::introspection::class_identity::class_identity_for_value;

pub(super) async fn value_is_ordered_against_zero_async(
    value: &Value,
    operation: IntegerComparisonOp,
) -> BuiltinResult<bool> {
    admit_pair_before_download(value, &Value::Num(0.0))?;
    let value = host_value(value).await?;
    comparison_all_normalized(&value, &Value::Num(0.0), operation)
}

pub async fn value_is_greater_than_values_async(
    value: &Value,
    bound: &Value,
) -> BuiltinResult<bool> {
    compare_values_async(value, bound, IntegerComparisonOp::Gt).await
}

pub async fn value_is_greater_than_or_equal_values_async(
    value: &Value,
    bound: &Value,
) -> BuiltinResult<bool> {
    compare_values_async(value, bound, IntegerComparisonOp::Ge).await
}

pub async fn value_is_less_than_values_async(value: &Value, bound: &Value) -> BuiltinResult<bool> {
    compare_values_async(value, bound, IntegerComparisonOp::Lt).await
}

pub async fn value_is_less_than_or_equal_values_async(
    value: &Value,
    bound: &Value,
) -> BuiltinResult<bool> {
    compare_values_async(value, bound, IntegerComparisonOp::Le).await
}

async fn compare_values_async(
    value: &Value,
    bound: &Value,
    operation: IntegerComparisonOp,
) -> BuiltinResult<bool> {
    admit_pair_before_download(value, bound)?;
    let value = host_value(value).await?;
    let bound = host_value(bound).await?;
    comparison_all_normalized(&value, &bound, operation)
}

pub async fn value_is_in_range_values_async(
    value: &Value,
    lower: &Value,
    upper: &Value,
    inclusivity: RangeInclusivity,
) -> BuiltinResult<bool> {
    admit_pair_before_download(value, lower)?;
    admit_pair_before_download(value, upper)?;
    let value = host_value(value).await?;
    let lower = host_value(lower).await?;
    let upper = host_value(upper).await?;
    let lower_op = if inclusivity.lower {
        IntegerComparisonOp::Ge
    } else {
        IntegerComparisonOp::Gt
    };
    let upper_op = if inclusivity.upper {
        IntegerComparisonOp::Le
    } else {
        IntegerComparisonOp::Lt
    };
    Ok(comparison_all_normalized(&value, &lower, lower_op)?
        && comparison_all_normalized(&value, &upper, upper_op)?)
}

fn admit_pair_before_download(lhs: &Value, rhs: &Value) -> BuiltinResult<()> {
    // The exact comparison domain owns dense numeric admission. Sparse forms
    // retain their existing specialized path; false means that domain did not
    // claim the operands, not that the validator should reject them.
    crate::builtins::logical::rel::integer_comparison::admit_numeric_pair(lhs, rhs).map_err(
        |error| match error {
            IntegerComparisonError::SizeMismatch => invalid_argument_error(
                "argumentValidation",
                "numeric inputs are not compatible for implicit expansion",
            ),
            IntegerComparisonError::Internal => {
                invalid_argument_error("argumentValidation", "numeric comparison failed")
            }
        },
    )?;
    Ok(())
}

pub async fn value_is_in_range_documented_async(
    value: &Value,
    lower: &Value,
    upper: &Value,
    inclusivity: RangeInclusivity,
) -> BuiltinResult<bool> {
    ensure_same_numeric_class("mustBeInRange", value, lower)?;
    ensure_same_numeric_class("mustBeInRange", value, upper)?;
    value_is_in_range_values_async(value, lower, upper, inclusivity).await
}

fn comparison_all_normalized(
    lhs: &Value,
    rhs: &Value,
    operation: IntegerComparisonOp,
) -> BuiltinResult<bool> {
    if let Value::SparseTensor(sparse) = lhs {
        if let Some(rhs) = scalar_numeric_value(rhs) {
            return sparse_scalar_comparison_all(sparse, &rhs, true, operation);
        }
    }
    if let Value::SparseTensor(sparse) = rhs {
        if let Some(lhs) = scalar_numeric_value(lhs) {
            return sparse_scalar_comparison_all(sparse, &lhs, false, operation);
        }
    }
    let lhs_dense;
    let rhs_dense;
    let lhs = if let Value::SparseTensor(sparse) = lhs {
        lhs_dense = Value::Tensor(
            sparse
                .to_dense()
                .map_err(|error| invalid_argument_error("argumentValidation", error))?,
        );
        &lhs_dense
    } else {
        lhs
    };
    let rhs = if let Value::SparseTensor(sparse) = rhs {
        rhs_dense = Value::Tensor(
            sparse
                .to_dense()
                .map_err(|error| invalid_argument_error("argumentValidation", error))?,
        );
        &rhs_dense
    } else {
        rhs
    };
    comparison_all(lhs, rhs, operation)
}

fn comparison_all(lhs: &Value, rhs: &Value, operation: IntegerComparisonOp) -> BuiltinResult<bool> {
    let map_error = |error| match error {
        IntegerComparisonError::SizeMismatch => invalid_argument_error(
            "argumentValidation",
            "numeric inputs are not compatible for implicit expansion",
        ),
        IntegerComparisonError::Internal => {
            invalid_argument_error("argumentValidation", "numeric comparison failed")
        }
    };
    let result = if let Some(result) =
        try_real_ordering_comparison(lhs, rhs, operation).map_err(map_error)?
    {
        result
    } else {
        try_complex_ordering_comparison(lhs, rhs, operation)
            .map_err(map_error)?
            .ok_or_else(|| {
                invalid_argument_error("argumentValidation", "expected comparable numeric inputs")
            })?
    };
    logical_value_all_true(&result)
}

fn scalar_numeric_value(value: &Value) -> Option<Value> {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => Some(value.clone()),
        Value::Complex(real, _) => Some(Value::Num(*real)),
        Value::Tensor(tensor) if tensor::tensor_element_len(tensor) == 1 => tensor
            .numeric_value_at(0)
            .map(|value| match value.into_int_value() {
                Some(value) => Value::Int(value),
                None => Value::Num(floating_numeric_scalar_to_f64(value)),
            }),
        Value::LogicalArray(array) if array.data.len() == 1 => {
            Some(Value::Bool(array.data[0] != 0))
        }
        Value::ComplexTensor(tensor) if tensor::complex_tensor_element_len(tensor) == 1 => {
            tensor.numeric_value_at(0).map(|(real, _)| {
                if let Some(real) = real.into_int_value() {
                    Value::Int(real)
                } else {
                    Value::Num(floating_numeric_scalar_to_f64(real))
                }
            })
        }
        _ => None,
    }
}

fn sparse_scalar_comparison_all(
    sparse: &SparseTensor,
    scalar: &Value,
    sparse_is_left: bool,
    operation: IntegerComparisonOp,
) -> BuiltinResult<bool> {
    for index in 0..sparse.nnz() {
        let value = sparse.numeric_value_at(index).ok_or_else(|| {
            invalid_argument_error("argumentValidation", "invalid sparse storage")
        })?;
        let value = match value.into_int_value() {
            Some(value) => Value::Int(value),
            None => Value::Num(floating_numeric_scalar_to_f64(value)),
        };
        let matches = if sparse_is_left {
            comparison_all(&value, scalar, operation)?
        } else {
            comparison_all(scalar, &value, operation)?
        };
        if !matches {
            return Ok(false);
        }
    }
    if sparse.nnz() < sparse.rows.saturating_mul(sparse.cols) {
        let zero = Value::Num(0.0);
        return if sparse_is_left {
            comparison_all(&zero, scalar, operation)
        } else {
            comparison_all(scalar, &zero, operation)
        };
    }
    Ok(true)
}

fn floating_numeric_scalar_to_f64(value: runmat_value::NumericScalar) -> f64 {
    match value {
        runmat_value::NumericScalar::F64(value) => value,
        runmat_value::NumericScalar::F32(value) => f64::from(value),
        _ => unreachable!("integer numeric scalar was handled before floating conversion"),
    }
}

fn logical_value_all_true(value: &Value) -> BuiltinResult<bool> {
    match value {
        Value::Bool(value) => Ok(*value),
        Value::LogicalArray(array) => Ok(array.data.iter().all(|value| *value != 0)),
        _ => Err(invalid_argument_error(
            "argumentValidation",
            "comparison did not return logical data",
        )
        .into()),
    }
}

fn ensure_same_numeric_class(builtin: &str, lhs: &Value, rhs: &Value) -> BuiltinResult<()> {
    let lhs = numeric_class_identity(lhs)
        .ok_or_else(|| invalid_argument_error(builtin, "expected numeric input"))?;
    let rhs = numeric_class_identity(rhs)
        .ok_or_else(|| invalid_argument_error(builtin, "expected numeric bound"))?;
    if lhs != rhs {
        return Err(invalid_argument_error(
            builtin,
            "bounds must have the same numeric class as the value",
        )
        .into());
    }
    Ok(())
}

pub(super) fn numeric_class_identity(value: &Value) -> Option<runmat_types::ClassIdentity> {
    match value {
        Value::GpuTensor(handle) if handle_is_logical(handle) => {
            Some(runmat_types::standard::LOGICAL.owned())
        }
        Value::GpuTensor(handle) if handle_integer_type(handle).is_some() => {
            crate::builtins::common::gpu_helpers::expected_gpu_class_identity(
                None,
                handle_integer_type(handle),
                false,
            )
        }
        Value::GpuTensor(handle) => {
            crate::builtins::common::gpu_helpers::expected_gpu_class_identity(
                runmat_accelerate_api::handle_precision(handle),
                None,
                false,
            )
        }
        _ if value_is_numeric_or_logical(value) => Some(class_identity_for_value(value)),
        _ => None,
    }
}
