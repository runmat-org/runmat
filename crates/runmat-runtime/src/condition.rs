//! Executor-neutral condition conversion.
//!
//! Bytecode, native, and browser executors must agree on which runtime values
//! are valid scalar conditions and how accelerator-backed values are gathered.

use runmat_value::Value;

use crate::builtins::common::tensor::{is_scalar_tensor, tensor_element_len, tensor_value_f64};
use crate::{gather_if_needed_async, runtime_error::semantic_error, RuntimeError};

/// Convert a runtime value to the scalar truth value required by control flow.
pub async fn logical_truth_from_value(value: &Value, label: &str) -> Result<bool, RuntimeError> {
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
        Value::GpuTensor(_) => {
            let gathered = gather_if_needed_async(value)
                .await
                // Preserve the VM-era diagnostic behavior while centralizing
                // the conversion: gather failures are execution failures, not
                // invalid-condition semantic errors.
                .map_err(|error| RuntimeError::new(format!("{label}: {error}")))?;
            Box::pin(logical_truth_from_value(&gathered, label)).await
        }
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
