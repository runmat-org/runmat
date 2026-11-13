//! Executor-neutral condition conversion.
//!
//! Bytecode, native, and browser executors must agree on which runtime values
//! are valid conditions and how accelerator-backed values are gathered.

use runmat_value::Value;

use crate::builtins::common::tensor::{is_scalar_tensor, tensor_element_len, tensor_value_f64};
use crate::{gather_if_needed_async, runtime_error::semantic_error, RuntimeError};

/// Branches accept nonempty real numeric/logical arrays whose elements are all
/// nonzero. Short-circuit operators use the separate scalar conversion below.
pub async fn control_flow_truth_from_value(
    runtime: &crate::context::RuntimeContext,
    value: &Value,
    label: &str,
) -> Result<bool, RuntimeError> {
    let gathered;
    let value = if matches!(value, Value::GpuTensor(_)) {
        gathered = runtime.scope(gather_if_needed_async(value)).await?;
        &gathered
    } else {
        value
    };
    match value {
        Value::LogicalArray(array) => {
            all_nonzero(runtime, array.data.len(), |index| array.data[index] != 0).await
        }
        Value::Tensor(tensor) => {
            all_nonzero(runtime, tensor_element_len(tensor), |index| {
                tensor
                    .numeric_value_at(index)
                    .is_some_and(|value| !value.is_zero())
            })
            .await
        }
        Value::SparseTensor(tensor) if !tensor.is_complex() => {
            let elements = tensor.rows.checked_mul(tensor.cols).ok_or_else(|| {
                invalid_condition(label, "sparse dimensions exceed platform limits".into())
            })?;
            if elements == 0 || tensor.nnz() != elements {
                return Ok(false);
            }
            all_nonzero(runtime, elements, |index| {
                tensor
                    .numeric_value_at(index)
                    .is_some_and(|value| !value.is_zero())
            })
            .await
        }
        _ => scalar_truth_from_host_value(value, label),
    }
}

async fn all_nonzero(
    runtime: &crate::context::RuntimeContext,
    len: usize,
    nonzero: impl Fn(usize) -> bool,
) -> Result<bool, RuntimeError> {
    let mut cooperation = runtime.cpu_cooperation();
    cooperation
        .checkpoint(0)
        .await
        .map_err(|error| error.into_runtime_error_in_context(Some(runtime)))?;
    for start in (0..len).step_by(4096) {
        let end = start.saturating_add(4096).min(len);
        let result = (start..end).all(&nonzero);
        cooperation
            .checkpoint((end - start) as u64)
            .await
            .map_err(|error| error.into_runtime_error_in_context(Some(runtime)))?;
        if !result {
            return Ok(false);
        }
    }
    Ok(len != 0)
}

/// Convert a runtime value to the scalar truth value required by logical
/// operators. Branch-array reduction uses `control_flow_truth_from_value`.
pub async fn logical_truth_from_value(value: &Value, label: &str) -> Result<bool, RuntimeError> {
    let gathered;
    let value = if matches!(value, Value::GpuTensor(_)) {
        gathered = gather_if_needed_async(value)
            .await
            .map_err(|error| RuntimeError::new(format!("{label}: {error}")))?;
        &gathered
    } else {
        value
    };
    scalar_truth_from_host_value(value, label)
}

fn scalar_truth_from_host_value(value: &Value, label: &str) -> Result<bool, RuntimeError> {
    match value {
        Value::Bool(flag) => Ok(*flag),
        Value::Int(integer) => Ok(!integer.is_zero()),
        Value::Num(number) => Ok(*number != 0.0),
        Value::LogicalArray(array) if array.data.len() == 1 => Ok(array.data[0] != 0),
        Value::LogicalArray(array) => Err(invalid_condition(
            label,
            format!("logical array with {} elements", array.data.len()),
        )),
        Value::Tensor(tensor) if is_scalar_tensor(tensor) => Ok(tensor_value_f64(tensor, 0) != 0.0),
        Value::Tensor(tensor) => Err(invalid_condition(
            label,
            format!("numeric array with {} elements", tensor_element_len(tensor)),
        )),
        other => Err(invalid_condition(label, format!("{other:?}"))),
    }
}

fn invalid_condition(label: &str, actual: String) -> RuntimeError {
    semantic_error(
        "InvalidConditionType",
        format!("{label}: expected scalar logical or numeric value, got {actual}"),
    )
}

#[cfg(test)]
mod tests {
    use futures::executor::block_on;
    use runmat_value::{IntValue, LogicalArray, Tensor};

    use super::*;

    #[test]
    fn branches_reduce_real_arrays_without_relaxing_short_circuit_operands() {
        let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        for (value, expected) in [
            (Value::Tensor(Tensor::new_2d(vec![], 0, 0).unwrap()), false),
            (
                Value::Tensor(Tensor::new_2d(vec![2.0, -1.0], 1, 2).unwrap()),
                true,
            ),
            (
                Value::Tensor(Tensor::new_2d(vec![2.0, 0.0], 1, 2).unwrap()),
                false,
            ),
            (
                Value::LogicalArray(LogicalArray::new(vec![1, 1], vec![1, 2]).unwrap()),
                true,
            ),
            (
                Value::Tensor(
                    Tensor::new_integer(
                        runmat_value::IntegerStorage::U64(vec![u64::MAX, 1]),
                        vec![1, 2],
                    )
                    .unwrap(),
                ),
                true,
            ),
            (
                Value::SparseTensor(
                    runmat_value::SparseTensor::new(2, 1, vec![0, 1], vec![0], vec![1.0]).unwrap(),
                ),
                false,
            ),
        ] {
            assert_eq!(
                block_on(control_flow_truth_from_value(&runtime, &value, "branch")).unwrap(),
                expected
            );
            assert!(block_on(logical_truth_from_value(&value, "short-circuit")).is_err());
        }
    }

    #[test]
    fn accepts_scalar_logical_and_numeric_values() {
        assert!(!block_on(logical_truth_from_value(&Value::Bool(false), "condition")).unwrap());
        assert!(block_on(logical_truth_from_value(
            &Value::Int(IntValue::I32(-2)),
            "condition"
        ))
        .unwrap());
        assert!(!block_on(logical_truth_from_value(&Value::Num(0.0), "condition")).unwrap());
        assert!(block_on(logical_truth_from_value(
            &Value::LogicalArray(LogicalArray::new(vec![1], vec![1, 1]).unwrap()),
            "condition"
        ))
        .unwrap());
        assert!(block_on(logical_truth_from_value(
            &Value::Tensor(Tensor::new_2d(vec![3.0], 1, 1).unwrap()),
            "condition"
        ))
        .unwrap());
    }

    #[test]
    fn rejects_nonscalar_and_non_numeric_values_with_semantic_identity() {
        let error = block_on(logical_truth_from_value(
            &Value::Tensor(Tensor::new_2d(vec![1.0, 2.0], 1, 2).unwrap()),
            "if condition",
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:InvalidConditionType"));
        assert!(error
            .to_string()
            .contains("if condition: expected scalar logical or numeric value"));
    }
}
