use num_complex::{Complex32, Complex64};
use runmat_value::{ComplexStorage, ComplexTensor, NumericStorage, Tensor, Value};

use crate::builtins::common::{broadcast::BroadcastPlan, random_args::complex_tensor_into_value};
use crate::BuiltinResult;

use super::DivisionContext;

pub(super) fn division_complex_complex(
    context: DivisionContext,
    lhs: &ComplexTensor,
    rhs: &ComplexTensor,
) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape)
        .map_err(|err| context.error_with_detail(context.size_mismatch, &err))?;
    let output = match (lhs.complex_storage(), rhs.complex_storage()) {
        (ComplexStorage::F64(lhs), ComplexStorage::F64(rhs)) => {
            let mut output = vec![(0.0f64, 0.0f64); plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                output[output_index] = divide_complex_f64(lhs[lhs_index], rhs[rhs_index]);
            }
            ComplexStorage::F64(output.into())
        }
        (ComplexStorage::F32(lhs), ComplexStorage::F32(rhs)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                output[output_index] = divide_complex_f32(lhs[lhs_index], rhs[rhs_index]);
            }
            ComplexStorage::F32(output.into())
        }
        (ComplexStorage::F32(lhs), ComplexStorage::F64(rhs)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                let lhs = (f64::from(lhs[lhs_index].0), f64::from(lhs[lhs_index].1));
                let value = divide_complex_f64(lhs, rhs[rhs_index]);
                output[output_index] = (value.0 as f32, value.1 as f32);
            }
            ComplexStorage::F32(output.into())
        }
        (ComplexStorage::F64(lhs), ComplexStorage::F32(rhs)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                let rhs = (f64::from(rhs[rhs_index].0), f64::from(rhs[rhs_index].1));
                let value = divide_complex_f64(lhs[lhs_index], rhs);
                output[output_index] = (value.0 as f32, value.1 as f32);
            }
            ComplexStorage::F32(output.into())
        }
        _ => return Err(context.internal_error("complex integer arithmetic is not supported")),
    };
    let tensor = ComplexTensor::from_complex_storage(output, plan.output_shape().to_vec())
        .map_err(|error| context.internal_error(error))?;
    Ok(complex_tensor_into_value(tensor))
}

pub(super) fn division_complex_real(
    context: DivisionContext,
    lhs: &ComplexTensor,
    rhs: &Tensor,
) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape)
        .map_err(|err| context.error_with_detail(context.size_mismatch, &err))?;
    let rhs = rhs
        .clone()
        .into_numeric_storage()
        .map_err(|error| context.internal_error(error))?;
    let output = divide_complex_by_real_storage(context, lhs.complex_storage(), &rhs, &plan)?;
    let tensor = ComplexTensor::from_complex_storage(output, plan.output_shape().to_vec())
        .map_err(|error| context.internal_error(error))?;
    Ok(complex_tensor_into_value(tensor))
}

pub(super) fn division_real_complex(
    context: DivisionContext,
    lhs: &Tensor,
    rhs: &ComplexTensor,
) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape)
        .map_err(|err| context.error_with_detail(context.size_mismatch, &err))?;
    let lhs = lhs
        .clone()
        .into_numeric_storage()
        .map_err(|error| context.internal_error(error))?;
    let output = divide_real_by_complex_storage(context, &lhs, rhs.complex_storage(), &plan)?;
    let tensor = ComplexTensor::from_complex_storage(output, plan.output_shape().to_vec())
        .map_err(|error| context.internal_error(error))?;
    Ok(complex_tensor_into_value(tensor))
}

fn divide_complex_by_real_storage(
    context: DivisionContext,
    complex: &ComplexStorage,
    real: &NumericStorage,
    plan: &BroadcastPlan,
) -> BuiltinResult<ComplexStorage> {
    Ok(match (complex, real) {
        (ComplexStorage::F64(complex), NumericStorage::F64(real)) => {
            let mut output = vec![(0.0f64, 0.0f64); plan.len()];
            for (output_index, complex_index, real_index) in plan.iter() {
                let value = complex[complex_index];
                let scalar = real[real_index];
                output[output_index] = (value.0 / scalar, value.1 / scalar);
            }
            ComplexStorage::F64(output.into())
        }
        (ComplexStorage::F32(complex), NumericStorage::F32(real)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, complex_index, real_index) in plan.iter() {
                let value = complex[complex_index];
                let scalar = real[real_index];
                output[output_index] = (value.0 / scalar, value.1 / scalar);
            }
            ComplexStorage::F32(output.into())
        }
        (ComplexStorage::F32(complex), NumericStorage::F64(real)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, complex_index, real_index) in plan.iter() {
                let value = complex[complex_index];
                let scalar = real[real_index];
                output[output_index] = (
                    (f64::from(value.0) / scalar) as f32,
                    (f64::from(value.1) / scalar) as f32,
                );
            }
            ComplexStorage::F32(output.into())
        }
        (ComplexStorage::F64(complex), NumericStorage::F32(real)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, complex_index, real_index) in plan.iter() {
                let value = complex[complex_index];
                let scalar = f64::from(real[real_index]);
                output[output_index] = ((value.0 / scalar) as f32, (value.1 / scalar) as f32);
            }
            ComplexStorage::F32(output.into())
        }
        _ => {
            return Err(context
                .internal_error("integer operands did not use the exact integer arithmetic path"))
        }
    })
}

fn divide_real_by_complex_storage(
    context: DivisionContext,
    real: &NumericStorage,
    complex: &ComplexStorage,
    plan: &BroadcastPlan,
) -> BuiltinResult<ComplexStorage> {
    Ok(match (real, complex) {
        (NumericStorage::F64(real), ComplexStorage::F64(complex)) => {
            let mut output = vec![(0.0f64, 0.0f64); plan.len()];
            for (output_index, real_index, complex_index) in plan.iter() {
                output[output_index] =
                    divide_complex_f64((real[real_index], 0.0), complex[complex_index]);
            }
            ComplexStorage::F64(output.into())
        }
        (NumericStorage::F32(real), ComplexStorage::F32(complex)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, real_index, complex_index) in plan.iter() {
                output[output_index] =
                    divide_complex_f32((real[real_index], 0.0), complex[complex_index]);
            }
            ComplexStorage::F32(output.into())
        }
        (NumericStorage::F32(real), ComplexStorage::F64(complex)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, real_index, complex_index) in plan.iter() {
                let value =
                    divide_complex_f64((f64::from(real[real_index]), 0.0), complex[complex_index]);
                output[output_index] = (value.0 as f32, value.1 as f32);
            }
            ComplexStorage::F32(output.into())
        }
        (NumericStorage::F64(real), ComplexStorage::F32(complex)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, real_index, complex_index) in plan.iter() {
                let complex = (
                    f64::from(complex[complex_index].0),
                    f64::from(complex[complex_index].1),
                );
                let value = divide_complex_f64((real[real_index], 0.0), complex);
                output[output_index] = (value.0 as f32, value.1 as f32);
            }
            ComplexStorage::F32(output.into())
        }
        _ => {
            return Err(context
                .internal_error("integer operands did not use the exact integer arithmetic path"))
        }
    })
}

fn divide_complex_f64(lhs: impl Into<(f64, f64)>, rhs: impl Into<(f64, f64)>) -> (f64, f64) {
    let lhs = lhs.into();
    let rhs = rhs.into();
    let quotient = Complex64::new(lhs.0, lhs.1) / Complex64::new(rhs.0, rhs.1);
    (quotient.re, quotient.im)
}

fn divide_complex_f32(lhs: impl Into<(f32, f32)>, rhs: impl Into<(f32, f32)>) -> (f32, f32) {
    let lhs = lhs.into();
    let rhs = rhs.into();
    let quotient = Complex32::new(lhs.0, lhs.1) / Complex32::new(rhs.0, rhs.1);
    (quotient.re, quotient.im)
}
