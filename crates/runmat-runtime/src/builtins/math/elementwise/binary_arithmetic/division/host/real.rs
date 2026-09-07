use runmat_value::{NumericStorage, Tensor, Value};

use crate::builtins::common::{broadcast::BroadcastPlan, tensor};
use crate::BuiltinResult;

use super::DivisionContext;

pub(super) fn division_real_real(
    context: DivisionContext,
    lhs: Tensor,
    rhs: Tensor,
) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape)
        .map_err(|err| context.error_with_detail(context.size_mismatch, &err))?;
    let output_shape = plan.output_shape().to_vec();
    let lhs = lhs
        .into_numeric_storage()
        .map_err(|error| context.internal_error(error))?;
    let rhs = rhs
        .into_numeric_storage()
        .map_err(|error| context.internal_error(error))?;
    let output = match (lhs, rhs) {
        (NumericStorage::F32(lhs), NumericStorage::F32(rhs)) => {
            let mut output = vec![0.0f32; plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                output[output_index] = lhs[lhs_index] / rhs[rhs_index];
            }
            NumericStorage::F32(output)
        }
        (NumericStorage::F64(lhs), NumericStorage::F64(rhs)) => {
            let mut output = vec![0.0f64; plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                output[output_index] = lhs[lhs_index] / rhs[rhs_index];
            }
            NumericStorage::F64(output)
        }
        (NumericStorage::F32(lhs), NumericStorage::F64(rhs)) => {
            let mut output = vec![0.0f32; plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                output[output_index] = (f64::from(lhs[lhs_index]) / rhs[rhs_index]) as f32;
            }
            NumericStorage::F32(output)
        }
        (NumericStorage::F64(lhs), NumericStorage::F32(rhs)) => {
            let mut output = vec![0.0f32; plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                output[output_index] = (lhs[lhs_index] / f64::from(rhs[rhs_index])) as f32;
            }
            NumericStorage::F32(output)
        }
        _ => {
            return Err(context
                .internal_error("integer operands did not use the exact integer arithmetic path"))
        }
    };
    let tensor = Tensor::from_numeric_storage(output, output_shape)
        .map_err(|error| context.internal_error(error))?;
    Ok(tensor::tensor_into_value(tensor))
}
