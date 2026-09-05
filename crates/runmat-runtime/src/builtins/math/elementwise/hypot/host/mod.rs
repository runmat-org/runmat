mod conversion;
mod kernel;

use runmat_builtins::{HYPOT_ERROR_INTERNAL, HYPOT_ERROR_SIZE_MISMATCH};
use runmat_value::{NumericStorage, Tensor, Value};

use crate::builtins::common::{broadcast::BroadcastPlan, tensor};
use crate::BuiltinResult;

use super::errors;

pub(super) fn evaluate(left: Value, right: Value) -> BuiltinResult<Value> {
    if let (Some(left), Some(right)) = (conversion::scalar(&left), conversion::scalar(&right)) {
        return Ok(Value::Num(kernel::f64(left, right)));
    }
    compute(
        conversion::into_tensor(left)?,
        conversion::into_tensor(right)?,
    )
}

pub(super) fn compute(left: Tensor, right: Tensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&left.shape, &right.shape)
        .map_err(|error| errors::with_detail(&HYPOT_ERROR_SIZE_MISMATCH, error))?;
    let output_shape = plan.output_shape().to_vec();
    let left_storage = left
        .into_numeric_storage()
        .map_err(|error| errors::with_detail(&HYPOT_ERROR_INTERNAL, error))?;
    let right_storage = right
        .into_numeric_storage()
        .map_err(|error| errors::with_detail(&HYPOT_ERROR_INTERNAL, error))?;
    let storage = if matches!(left_storage, NumericStorage::F32(_))
        && matches!(right_storage, NumericStorage::F32(_))
    {
        let left_values = conversion::single_domain(left_storage);
        let right_values = conversion::single_domain(right_storage);
        NumericStorage::F32(
            plan.iter()
                .map(|(_, left, right)| kernel::f32(left_values[left], right_values[right]))
                .collect(),
        )
    } else {
        let left_values = conversion::double_domain(left_storage)?;
        let right_values = conversion::double_domain(right_storage)?;
        NumericStorage::F64(
            plan.iter()
                .map(|(_, left, right)| kernel::f64(left_values[left], right_values[right]))
                .collect(),
        )
    };
    let output = Tensor::from_numeric_storage(storage, output_shape)
        .map_err(|error| errors::with_detail(&HYPOT_ERROR_INTERNAL, error))?;
    Ok(tensor::tensor_into_value(output))
}

#[cfg(test)]
pub(super) use conversion::scalar;
#[cfg(test)]
pub(super) use kernel::complex_magnitude;
