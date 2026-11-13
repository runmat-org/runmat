//! Native host complex windows over borrowed canonical interleaved storage.
use super::*;
use num_traits::Float;
use runmat_value::{ComplexElement, ComplexStorage};
mod action;
mod order;
mod reduce;
mod variance;
#[cfg(test)]
mod tests;

enum Output<T> {
    Complex(Vec<ComplexElement<T>>),
    Real(Vec<T>),
}
struct Evaluation<T> {
    values: Output<T>,
    shape: Vec<usize>,
}

fn number<T: Float>(value: usize) -> T {
    T::from(value).expect("host element counts fit the native floating exponent range")
}

impl<T: Float> Output<T> {
    fn new(operation: MovingOp, count: usize) -> Self {
        if operation.uses_variance_normalization() {
            Self::Real(vec![T::zero(); count])
        } else {
            Self::Complex(vec![ComplexElement(T::zero(), T::zero()); count])
        }
    }
    fn set(&mut self, index: usize, value: (T, T)) {
        match self {
            Self::Real(values) => values[index] = value.0,
            Self::Complex(values) => values[index] = value.into(),
        }
    }
}

pub(super) fn apply(
    name: &'static str,
    operation: MovingOp,
    input: &ComplexTensor,
    parsed: &ParsedMoving,
) -> BuiltinResult<Value> {
    if let Some(values) = input.as_f32_slice() {
        let output = action::evaluate(name, operation, values, input.shape(), parsed)?;
        match output.values {
            Output::Real(values) => {
                Tensor::from_f32(values, output.shape).map(tensor::tensor_into_value)
            }
            Output::Complex(values) => ComplexTensor::from_complex_storage(
                ComplexStorage::F32(values.into()),
                output.shape,
            )
            .map(Value::ComplexTensor),
        }
        .map_err(|error| internal(name, error))
    } else if let Some(values) = input.as_f64_slice() {
        let output = action::evaluate(name, operation, values, input.shape(), parsed)?;
        match output.values {
            Output::Real(values) => {
                Tensor::new(values, output.shape).map(tensor::tensor_into_value)
            }
            Output::Complex(values) => ComplexTensor::from_complex_storage(
                ComplexStorage::F64(values.into()),
                output.shape,
            )
            .map(complex_tensor_into_value),
        }
        .map_err(|error| internal(name, error))
    } else {
        Err(invalid_input(
            name,
            format!("{name}: complex integer storage is unsupported"),
        ))
    }
}
