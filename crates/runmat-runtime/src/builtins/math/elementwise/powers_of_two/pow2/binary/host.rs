use runmat_value::{ComplexStorage, ComplexTensor, NumericStorage, Tensor, Value};

use crate::builtins::common::{
    broadcast::BroadcastPlan, random_args::complex_tensor_into_value, tensor,
};
use crate::BuiltinResult;

use super::super::{errors, input, numeric};

pub(super) fn evaluate(significand: Value, exponent: Value) -> BuiltinResult<Value> {
    if let (Some(significand), Some(exponent)) = (
        input::scalar_component(&significand),
        input::scalar_component(&exponent),
    ) {
        let result = numeric::multiply_f64(significand, numeric::power_f64(exponent.0, exponent.1));
        return Ok(if result.1 == 0.0 {
            Value::Num(result.0)
        } else {
            Value::Complex(result.0, result.1)
        });
    }

    let significand = input::into_numeric_array(significand)?;
    let exponent = input::into_numeric_array(exponent)?;
    let plan =
        BroadcastPlan::new(significand.shape(), exponent.shape()).map_err(errors::size_mismatch)?;
    let shape = plan.output_shape().to_vec();
    let is_complex = significand.is_complex() || exponent.is_complex();
    if significand.uses_single() || exponent.uses_single() {
        evaluate_f32(significand, exponent, plan, shape, is_complex)
    } else {
        evaluate_f64(significand, exponent, plan, shape, is_complex)
    }
}

fn evaluate_f32(
    significand: input::NumericArray,
    exponent: input::NumericArray,
    plan: BroadcastPlan,
    shape: Vec<usize>,
    is_complex: bool,
) -> BuiltinResult<Value> {
    let significand = significand.into_f32_components()?;
    let exponent = exponent.into_f32_components()?;
    let values = plan
        .iter()
        .map(|(_, left, right)| {
            numeric::multiply_f32(
                significand[left],
                numeric::power_f32(exponent[right].0, exponent[right].1),
            )
        })
        .collect::<Vec<_>>();
    if is_complex {
        ComplexTensor::from_complex_storage(ComplexStorage::F32(values.into()), shape)
            .map(complex_tensor_into_value)
            .map_err(errors::internal)
    } else {
        let values = values.into_iter().map(|(real, _)| real).collect();
        Tensor::from_numeric_storage(NumericStorage::F32(values), shape)
            .map(tensor::tensor_into_value)
            .map_err(errors::internal)
    }
}

fn evaluate_f64(
    significand: input::NumericArray,
    exponent: input::NumericArray,
    plan: BroadcastPlan,
    shape: Vec<usize>,
    is_complex: bool,
) -> BuiltinResult<Value> {
    let significand = significand.into_f64_components()?;
    let exponent = exponent.into_f64_components()?;
    let values = plan
        .iter()
        .map(|(_, left, right)| {
            numeric::multiply_f64(
                significand[left],
                numeric::power_f64(exponent[right].0, exponent[right].1),
            )
        })
        .collect::<Vec<_>>();
    if is_complex {
        ComplexTensor::from_complex_storage(ComplexStorage::F64(values.into()), shape)
            .map(complex_tensor_into_value)
            .map_err(errors::internal)
    } else {
        let values = values.into_iter().map(|(real, _)| real).collect();
        Tensor::from_numeric_storage(NumericStorage::F64(values), shape)
            .map(tensor::tensor_into_value)
            .map_err(errors::internal)
    }
}
