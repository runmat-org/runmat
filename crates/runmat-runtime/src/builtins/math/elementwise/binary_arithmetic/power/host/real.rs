use runmat_builtins::POWER_ERROR_SIZE_MISMATCH;
use runmat_value::{ComplexStorage, ComplexTensor, NumericStorage, Tensor, Value};

use crate::builtins::common::{
    broadcast::BroadcastPlan, random_args::complex_tensor_into_value, tensor,
};
use crate::BuiltinResult;

use super::super::{
    builtin_error, complex_pow_scalar, complex_pow_scalar_f32, power_error_with_detail,
};

pub(super) fn power_real_real(lhs: Tensor, rhs: Tensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape)
        .map_err(|err| power_error_with_detail(&POWER_ERROR_SIZE_MISMATCH, &err))?;
    let output_shape = plan.output_shape().to_vec();
    let lhs = lhs
        .into_numeric_storage()
        .map_err(|e| builtin_error(format!("power: {e}")))?;
    let rhs = rhs
        .into_numeric_storage()
        .map_err(|e| builtin_error(format!("power: {e}")))?;
    match (lhs, rhs) {
        (NumericStorage::F64(lhs), NumericStorage::F64(rhs)) => {
            let mut output = Vec::with_capacity(plan.len());
            for (_, lhs_index, rhs_index) in plan.iter() {
                output.push(power_real_pair_f64(lhs[lhs_index], rhs[rhs_index]));
            }
            finish_real_power_f64(output, output_shape)
        }
        (NumericStorage::F32(lhs), NumericStorage::F32(rhs)) => {
            let mut output = Vec::with_capacity(plan.len());
            for (_, lhs_index, rhs_index) in plan.iter() {
                output.push(power_real_pair_f32(lhs[lhs_index], rhs[rhs_index]));
            }
            finish_real_power_f32(output, output_shape)
        }
        (NumericStorage::F32(lhs), NumericStorage::F64(rhs)) => {
            let mut output = Vec::with_capacity(plan.len());
            for (_, lhs_index, rhs_index) in plan.iter() {
                output.push(power_real_pair_mixed(
                    f64::from(lhs[lhs_index]),
                    rhs[rhs_index],
                ));
            }
            finish_real_power_f32(output, output_shape)
        }
        (NumericStorage::F64(lhs), NumericStorage::F32(rhs)) => {
            let mut output = Vec::with_capacity(plan.len());
            for (_, lhs_index, rhs_index) in plan.iter() {
                output.push(power_real_pair_mixed(
                    lhs[lhs_index],
                    f64::from(rhs[rhs_index]),
                ));
            }
            finish_real_power_f32(output, output_shape)
        }
        (lhs, rhs) => Err(builtin_error(format!(
            "power: integer {} or {} storage did not use the exact integer arithmetic path",
            lhs.class_name(),
            rhs.class_name()
        ))),
    }
}

fn power_real_pair_f64(base: f64, exponent: f64) -> (f64, f64) {
    let value = base.powf(exponent);
    if value.is_nan() {
        complex_pow_scalar(base, 0.0, exponent, 0.0)
    } else {
        (value, 0.0)
    }
}

fn power_real_pair_f32(base: f32, exponent: f32) -> (f32, f32) {
    let value = base.powf(exponent);
    if value.is_nan() {
        complex_pow_scalar_f32(base, 0.0, exponent, 0.0)
    } else {
        (value, 0.0)
    }
}

fn power_real_pair_mixed(base: f64, exponent: f64) -> (f32, f32) {
    let (real, imag) = power_real_pair_f64(base, exponent);
    (real as f32, imag as f32)
}

fn finish_real_power_f64(output: Vec<(f64, f64)>, shape: Vec<usize>) -> BuiltinResult<Value> {
    if output.iter().any(|value| value.1.abs() > 1e-12) {
        let tensor = ComplexTensor::from_complex_storage(ComplexStorage::F64(output.into()), shape)
            .map_err(|e| builtin_error(format!("power: {e}")))?;
        Ok(complex_tensor_into_value(tensor))
    } else {
        let storage = NumericStorage::F64(output.into_iter().map(|value| value.0).collect());
        let tensor = Tensor::from_numeric_storage(storage, shape)
            .map_err(|e| builtin_error(format!("power: {e}")))?;
        Ok(tensor::tensor_into_value(tensor))
    }
}

fn finish_real_power_f32(output: Vec<(f32, f32)>, shape: Vec<usize>) -> BuiltinResult<Value> {
    if output.iter().any(|value| value.1.abs() > 1e-6) {
        let tensor = ComplexTensor::from_complex_storage(ComplexStorage::F32(output.into()), shape)
            .map_err(|e| builtin_error(format!("power: {e}")))?;
        Ok(complex_tensor_into_value(tensor))
    } else {
        let storage = NumericStorage::F32(output.into_iter().map(|value| value.0).collect());
        let tensor = Tensor::from_numeric_storage(storage, shape)
            .map_err(|e| builtin_error(format!("power: {e}")))?;
        Ok(tensor::tensor_into_value(tensor))
    }
}
