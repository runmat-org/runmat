use nalgebra::DMatrix;
use num_complex::Complex64;
use runmat_value::{ComplexTensor, NumericDType};

use crate::BuiltinResult;

use super::super::{errors, SolveOrientation};
use super::{conversion, solver, validation};

pub(super) fn evaluate(
    orientation: SolveOrientation,
    lhs: &ComplexTensor,
    rhs: &ComplexTensor,
) -> BuiltinResult<ComplexTensor> {
    validation::ensure_matrix(orientation, &lhs.shape)?;
    validation::ensure_matrix(orientation, &rhs.shape)?;
    let divisor = match orientation {
        SolveOrientation::Left if conversion::is_complex_scalar(lhs) => Some((lhs, rhs)),
        SolveOrientation::Right if conversion::is_complex_scalar(rhs) => Some((rhs, lhs)),
        _ => None,
    };
    if let Some((scalar, values)) = divisor {
        let (real, imag) = scalar.materialize_f64()[0];
        return conversion::scale_complex(
            orientation,
            values,
            Complex64::new(1.0, 0.0) / Complex64::new(real, imag),
        );
    }

    validation::ensure_matching_complex_dimension(orientation, lhs, rhs)?;
    if validation::has_empty_complex_solve_dimension(orientation, lhs, rhs) {
        let (rows, cols) = validation::complex_output_dimensions(orientation, lhs, rhs);
        return ComplexTensor::new(vec![(0.0, 0.0); rows * cols], vec![rows, cols]).map_err(
            |error| errors::internal(orientation, format!("{}: {error}", orientation.name())),
        );
    }

    let lhs_data = conversion::complex_values(lhs);
    let rhs_data = conversion::complex_values(rhs);
    let lhs_matrix = DMatrix::from_column_slice(lhs.rows, lhs.cols, &lhs_data);
    let rhs_matrix = DMatrix::from_column_slice(rhs.rows, rhs.cols, &rhs_data);
    let solution = solver::solve(orientation, &lhs_matrix, &rhs_matrix)?;
    conversion::complex_tensor(orientation, solution, result_dtype(lhs, rhs))
}

fn result_dtype(lhs: &ComplexTensor, rhs: &ComplexTensor) -> NumericDType {
    if lhs.numeric_dtype() == NumericDType::F32 || rhs.numeric_dtype() == NumericDType::F32 {
        NumericDType::F32
    } else {
        NumericDType::F64
    }
}
