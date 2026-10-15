use runmat_builtins::POWER_ERROR_SIZE_MISMATCH;
use runmat_value::{ComplexStorage, ComplexTensor, NumericStorage, Tensor, Value};

use crate::builtins::common::{
    broadcast::BroadcastPlan, random_args::complex_tensor_into_value, tensor,
};
use crate::BuiltinResult;

use super::super::domain::{evaluate_f32, evaluate_f64, PowerResultDomain};
use super::super::{builtin_error, power_error_with_detail};

struct RealPowerOutput<T> {
    domain: PowerResultDomain,
    values: Vec<(T, T)>,
}

impl<T> RealPowerOutput<T> {
    fn with_capacity(len: usize) -> Self {
        Self {
            domain: PowerResultDomain::Real,
            values: Vec::with_capacity(len),
        }
    }
    fn push(&mut self, (domain, value): (PowerResultDomain, (T, T))) {
        self.domain = self.domain.join(domain);
        self.values.push(value);
    }
}

pub(super) fn power_real_real(lhs: Tensor, rhs: Tensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(lhs.shape(), rhs.shape())
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
            let mut output = RealPowerOutput::with_capacity(plan.len());
            for (_, lhs_index, rhs_index) in plan.iter() {
                output.push(evaluate_f64(lhs[lhs_index], rhs[rhs_index]));
            }
            finish_real_power_f64(output, output_shape)
        }
        (NumericStorage::F32(lhs), NumericStorage::F32(rhs)) => {
            let mut output = RealPowerOutput::with_capacity(plan.len());
            for (_, lhs_index, rhs_index) in plan.iter() {
                output.push(evaluate_f32(lhs[lhs_index], rhs[rhs_index]));
            }
            finish_real_power_f32(output, output_shape)
        }
        (NumericStorage::F32(lhs), NumericStorage::F64(rhs)) => {
            let mut output = RealPowerOutput::with_capacity(plan.len());
            let exact = (rhs.len() == 1)
                .then(|| {
                    runmat_accelerate_api::ExactSingleBasePower::for_control(
                        runmat_accelerate_api::PowerScalarControl::DoubleBits(rhs[0].to_bits()),
                    )
                })
                .flatten();
            if let Some(operation) = exact {
                for (_, lhs_index, _) in plan.iter() {
                    output.push((
                        PowerResultDomain::Real,
                        (operation.evaluate(lhs[lhs_index]), 0.0),
                    ));
                }
            } else {
                for (_, lhs_index, rhs_index) in plan.iter() {
                    output.push(power_real_pair_mixed(
                        f64::from(lhs[lhs_index]),
                        rhs[rhs_index],
                    ));
                }
            }
            finish_real_power_f32(output, output_shape)
        }
        (NumericStorage::F64(lhs), NumericStorage::F32(rhs)) => {
            let mut output = RealPowerOutput::with_capacity(plan.len());
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

fn power_real_pair_mixed(base: f64, exponent: f64) -> (PowerResultDomain, (f32, f32)) {
    let (domain, (real, imag)) = evaluate_f64(base, exponent);
    (domain, (real as f32, imag as f32))
}

fn finish_real_power_f64(output: RealPowerOutput<f64>, shape: Vec<usize>) -> BuiltinResult<Value> {
    let RealPowerOutput {
        domain,
        values: output,
    } = output;
    if domain == PowerResultDomain::Complex {
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

fn finish_real_power_f32(output: RealPowerOutput<f32>, shape: Vec<usize>) -> BuiltinResult<Value> {
    let RealPowerOutput {
        domain,
        values: output,
    } = output;
    if domain == PowerResultDomain::Complex {
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
