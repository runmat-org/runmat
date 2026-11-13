use super::*;

mod complex;
mod geometry;
mod real;
mod statistics;

use complex::evaluate_complex_matrix;
use geometry::{ComplexMatrix, Matrix};
use real::evaluate_real_matrix;
use statistics::{clamp_correlation, confidence_bounds, p_values};

#[derive(Debug)]
pub(super) enum NumericInput {
    Real(Tensor),
    Complex(ComplexTensor),
}

impl NumericInput {
    fn shape(&self) -> &[usize] {
        match self {
            Self::Real(tensor) => tensor.shape(),
            Self::Complex(tensor) => tensor.shape(),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Real(tensor) => tensor.len(),
            Self::Complex(tensor) => tensor.len(),
        }
    }

    fn is_complex(&self) -> bool {
        matches!(self, Self::Complex(_))
    }
}

#[derive(Debug)]
pub(super) enum CorrcoefEvaluation {
    Real(RealEvaluation),
    Complex(ComplexTensor),
}

#[derive(Debug)]
pub(super) struct RealEvaluation {
    pub(super) r: Tensor,
    counts: Vec<usize>,
}

impl CorrcoefEvaluation {
    pub(super) fn outputs(
        self,
        requested_outputs: Option<usize>,
        alpha: f64,
    ) -> BuiltinResult<Value> {
        match self {
            Self::Complex(r) => match requested_outputs {
                None => Ok(Value::ComplexTensor(r)),
                Some(1) => Ok(Value::output_list(vec![Value::ComplexTensor(r)])),
                Some(_) => Err(corrcoef_error(&CORRCOEF_ERROR_COMPLEX_OUTPUTS)),
            },
            Self::Real(evaluation) => evaluation.outputs(requested_outputs, alpha),
        }
    }
}

impl RealEvaluation {
    fn outputs(self, requested_outputs: Option<usize>, alpha: f64) -> BuiltinResult<Value> {
        let RealEvaluation { r, counts } = self;
        let Some(count) = requested_outputs else {
            return Ok(Value::Tensor(r));
        };
        if count > 4 {
            return Err(corrcoef_error_with_detail(
                &CORRCOEF_ERROR_INVALID_ARGUMENT,
                "too many output arguments; maximum is 4",
            ));
        }
        let mut outputs = Vec::with_capacity(count);
        if count >= 1 {
            outputs.push(Value::Tensor(r.clone()));
        }
        if count >= 2 {
            outputs.push(Value::Tensor(p_values(&r, &counts)?));
        }
        if count >= 3 {
            let (lower, upper) = confidence_bounds(&r, &counts, alpha)?;
            outputs.push(Value::Tensor(lower));
            if count >= 4 {
                outputs.push(Value::Tensor(upper));
            }
        }
        Ok(Value::output_list(outputs))
    }
}

pub(super) fn evaluate_inputs(
    left: NumericInput,
    right: Option<NumericInput>,
    rows: CorrcoefRows,
) -> BuiltinResult<CorrcoefEvaluation> {
    if left.shape().len() > 2 {
        return Err(corrcoef_error_with_detail(
            &CORRCOEF_ERROR_INVALID_ARGUMENT,
            "A must be a matrix",
        ));
    }
    if let Some(right) = right {
        if !same_size(left.shape(), right.shape()) {
            return Err(corrcoef_error(&CORRCOEF_ERROR_SIZE_MISMATCH));
        }
        if left.is_complex() || right.is_complex() {
            let matrix = ComplexMatrix::from_pair(left, right)?;
            return evaluate_complex_matrix(matrix, rows).map(CorrcoefEvaluation::Complex);
        }
        let (NumericInput::Real(left), NumericInput::Real(right)) = (left, right) else {
            unreachable!("complex inputs handled above")
        };
        evaluate_real_matrix(Matrix::from_pair(left, right), rows).map(CorrcoefEvaluation::Real)
    } else {
        match left {
            NumericInput::Real(tensor) => evaluate_real_matrix(Matrix::from_single(tensor), rows)
                .map(CorrcoefEvaluation::Real),
            NumericInput::Complex(tensor) => {
                evaluate_complex_matrix(ComplexMatrix::from_single(tensor), rows)
                    .map(CorrcoefEvaluation::Complex)
            }
        }
    }
}

pub(super) fn same_size(left: &[usize], right: &[usize]) -> bool {
    canonical_shape(left) == canonical_shape(right)
}

fn canonical_shape(shape: &[usize]) -> Vec<usize> {
    let mut canonical = shape.to_vec();
    while canonical.len() > 2 && canonical.last() == Some(&1) {
        canonical.pop();
    }
    while canonical.len() < 2 {
        canonical.push(1);
    }
    canonical
}

fn set_symmetric<T: Copy>(buffer: &mut [T], dim: usize, row: usize, col: usize, value: T) {
    buffer[row + col * dim] = value;
    buffer[col + row * dim] = value;
}

fn set_hermitian(buffer: &mut [Complex64], dim: usize, row: usize, col: usize, value: Complex64) {
    buffer[row + col * dim] = value;
    buffer[col + row * dim] = value.conj();
}
