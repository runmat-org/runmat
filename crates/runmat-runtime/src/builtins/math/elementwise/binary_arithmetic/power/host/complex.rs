use runmat_builtins::POWER_ERROR_SIZE_MISMATCH;
use runmat_value::{ComplexStorage, ComplexTensor, NumericStorage, Tensor, Value};

use crate::builtins::common::{broadcast::BroadcastPlan, random_args::complex_tensor_into_value};
use crate::BuiltinResult;

use super::super::{
    builtin_error, complex_pow_scalar, complex_pow_scalar_f32, power_error_with_detail,
};

pub(super) fn power_complex_complex(
    lhs: &ComplexTensor,
    rhs: &ComplexTensor,
) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape)
        .map_err(|err| power_error_with_detail(&POWER_ERROR_SIZE_MISMATCH, &err))?;
    let output = match (lhs.complex_storage(), rhs.complex_storage()) {
        (ComplexStorage::F64(lhs), ComplexStorage::F64(rhs)) => {
            let mut output = vec![(0.0f64, 0.0f64); plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                let (br, bi) = lhs[lhs_index].into();
                let (er, ei) = rhs[rhs_index].into();
                output[output_index] = complex_pow_scalar(br, bi, er, ei);
            }
            ComplexStorage::F64(output.into())
        }
        (ComplexStorage::F32(lhs), ComplexStorage::F32(rhs)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                let (br, bi) = lhs[lhs_index].into();
                let (er, ei) = rhs[rhs_index].into();
                output[output_index] = complex_pow_scalar_f32(br, bi, er, ei);
            }
            ComplexStorage::F32(output.into())
        }
        (ComplexStorage::F32(lhs), ComplexStorage::F64(rhs)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                let (br, bi) = lhs[lhs_index].into();
                let (er, ei) = rhs[rhs_index].into();
                let value = complex_pow_scalar(f64::from(br), f64::from(bi), er, ei);
                output[output_index] = (value.0 as f32, value.1 as f32);
            }
            ComplexStorage::F32(output.into())
        }
        (ComplexStorage::F64(lhs), ComplexStorage::F32(rhs)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                let (br, bi) = lhs[lhs_index].into();
                let (er, ei) = rhs[rhs_index].into();
                let value = complex_pow_scalar(br, bi, f64::from(er), f64::from(ei));
                output[output_index] = (value.0 as f32, value.1 as f32);
            }
            ComplexStorage::F32(output.into())
        }
        _ => {
            return Err(builtin_error(
                "power: complex integer arithmetic is not supported",
            ))
        }
    };
    let tensor = ComplexTensor::from_complex_storage(output, plan.output_shape().to_vec())
        .map_err(|e| builtin_error(format!("power: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

pub(super) fn power_complex_real(lhs: &ComplexTensor, rhs: &Tensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape)
        .map_err(|err| power_error_with_detail(&POWER_ERROR_SIZE_MISMATCH, &err))?;
    let rhs = rhs
        .clone()
        .into_numeric_storage()
        .map_err(|e| builtin_error(format!("power: {e}")))?;
    let output = power_complex_real_storage(lhs.complex_storage(), &rhs, &plan)?;
    let tensor = ComplexTensor::from_complex_storage(output, plan.output_shape().to_vec())
        .map_err(|e| builtin_error(format!("power: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

pub(super) fn power_real_complex(lhs: &Tensor, rhs: &ComplexTensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape)
        .map_err(|err| power_error_with_detail(&POWER_ERROR_SIZE_MISMATCH, &err))?;
    let lhs = lhs
        .clone()
        .into_numeric_storage()
        .map_err(|e| builtin_error(format!("power: {e}")))?;
    let output = power_real_complex_storage(&lhs, rhs.complex_storage(), &plan)?;
    let tensor = ComplexTensor::from_complex_storage(output, plan.output_shape().to_vec())
        .map_err(|e| builtin_error(format!("power: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

fn power_complex_real_storage(
    base: &ComplexStorage,
    exponent: &NumericStorage,
    plan: &BroadcastPlan,
) -> BuiltinResult<ComplexStorage> {
    Ok(match (base, exponent) {
        (ComplexStorage::F64(base), NumericStorage::F64(exponent)) => {
            let mut output = vec![(0.0f64, 0.0f64); plan.len()];
            for (output_index, base_index, exponent_index) in plan.iter() {
                let (br, bi) = base[base_index].into();
                output[output_index] = complex_pow_scalar(br, bi, exponent[exponent_index], 0.0);
            }
            ComplexStorage::F64(output.into())
        }
        (ComplexStorage::F32(base), NumericStorage::F32(exponent)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, base_index, exponent_index) in plan.iter() {
                let (br, bi) = base[base_index].into();
                output[output_index] =
                    complex_pow_scalar_f32(br, bi, exponent[exponent_index], 0.0);
            }
            ComplexStorage::F32(output.into())
        }
        (ComplexStorage::F32(base), NumericStorage::F64(exponent)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, base_index, exponent_index) in plan.iter() {
                let (br, bi) = base[base_index].into();
                let value =
                    complex_pow_scalar(f64::from(br), f64::from(bi), exponent[exponent_index], 0.0);
                output[output_index] = (value.0 as f32, value.1 as f32);
            }
            ComplexStorage::F32(output.into())
        }
        (ComplexStorage::F64(base), NumericStorage::F32(exponent)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, base_index, exponent_index) in plan.iter() {
                let (br, bi) = base[base_index].into();
                let value = complex_pow_scalar(br, bi, f64::from(exponent[exponent_index]), 0.0);
                output[output_index] = (value.0 as f32, value.1 as f32);
            }
            ComplexStorage::F32(output.into())
        }
        _ => {
            return Err(builtin_error(
                "power: integer operands did not use the exact integer arithmetic path",
            ))
        }
    })
}

fn power_real_complex_storage(
    base: &NumericStorage,
    exponent: &ComplexStorage,
    plan: &BroadcastPlan,
) -> BuiltinResult<ComplexStorage> {
    Ok(match (base, exponent) {
        (NumericStorage::F64(base), ComplexStorage::F64(exponent)) => {
            let mut output = vec![(0.0f64, 0.0f64); plan.len()];
            for (output_index, base_index, exponent_index) in plan.iter() {
                let (er, ei) = exponent[exponent_index].into();
                output[output_index] = complex_pow_scalar(base[base_index], 0.0, er, ei);
            }
            ComplexStorage::F64(output.into())
        }
        (NumericStorage::F32(base), ComplexStorage::F32(exponent)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, base_index, exponent_index) in plan.iter() {
                let (er, ei) = exponent[exponent_index].into();
                output[output_index] = complex_pow_scalar_f32(base[base_index], 0.0, er, ei);
            }
            ComplexStorage::F32(output.into())
        }
        (NumericStorage::F32(base), ComplexStorage::F64(exponent)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, base_index, exponent_index) in plan.iter() {
                let (er, ei) = exponent[exponent_index].into();
                let value = complex_pow_scalar(f64::from(base[base_index]), 0.0, er, ei);
                output[output_index] = (value.0 as f32, value.1 as f32);
            }
            ComplexStorage::F32(output.into())
        }
        (NumericStorage::F64(base), ComplexStorage::F32(exponent)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, base_index, exponent_index) in plan.iter() {
                let (er, ei) = exponent[exponent_index].into();
                let value = complex_pow_scalar(base[base_index], 0.0, f64::from(er), f64::from(ei));
                output[output_index] = (value.0 as f32, value.1 as f32);
            }
            ComplexStorage::F32(output.into())
        }
        _ => {
            return Err(builtin_error(
                "power: integer operands did not use the exact integer arithmetic path",
            ))
        }
    })
}
