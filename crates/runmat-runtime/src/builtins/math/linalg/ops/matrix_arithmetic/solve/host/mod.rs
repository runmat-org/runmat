mod complex;
mod conversion;
mod real;
mod solver;
mod validation;

use runmat_value::{ComplexTensor, Tensor, Value};

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::tensor::tensor_into_value;
use crate::BuiltinResult;

use super::SolveOrientation;

pub(super) enum NumericInput {
    Real(Tensor),
    Complex(ComplexTensor),
}

pub(super) fn evaluate(
    orientation: SolveOrientation,
    lhs: Value,
    rhs: Value,
) -> BuiltinResult<Value> {
    let lhs = conversion::classify(orientation, lhs)?;
    let rhs = conversion::classify(orientation, rhs)?;
    match (lhs, rhs) {
        (NumericInput::Real(lhs), NumericInput::Real(rhs)) => {
            real::evaluate(orientation, &lhs, &rhs).map(tensor_into_value)
        }
        (NumericInput::Complex(lhs), NumericInput::Complex(rhs)) => {
            complex::evaluate(orientation, &lhs, &rhs).map(complex_tensor_into_value)
        }
        (NumericInput::Complex(lhs), NumericInput::Real(rhs)) => {
            let rhs = conversion::promote_real(orientation, &rhs)?;
            complex::evaluate(orientation, &lhs, &rhs).map(complex_tensor_into_value)
        }
        (NumericInput::Real(lhs), NumericInput::Complex(rhs)) => {
            let lhs = conversion::promote_real(orientation, &lhs)?;
            complex::evaluate(orientation, &lhs, &rhs).map(complex_tensor_into_value)
        }
    }
}

pub(super) fn real(
    orientation: SolveOrientation,
    lhs: &Tensor,
    rhs: &Tensor,
) -> BuiltinResult<Tensor> {
    real::evaluate(orientation, lhs, rhs)
}
