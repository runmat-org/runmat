use nalgebra::DMatrix;
use runmat_value::{NumericDType, Tensor};

use crate::builtins::common::tensor;
use crate::BuiltinResult;

use super::super::{errors, SolveOrientation};
use super::{conversion, solver, validation};

pub(super) fn evaluate(
    orientation: SolveOrientation,
    lhs: &Tensor,
    rhs: &Tensor,
) -> BuiltinResult<Tensor> {
    validation::ensure_matrix(orientation, &lhs.shape)?;
    validation::ensure_matrix(orientation, &rhs.shape)?;
    let divisor = match orientation {
        SolveOrientation::Left if tensor::is_scalar_tensor(lhs) => Some((lhs, rhs)),
        SolveOrientation::Right if tensor::is_scalar_tensor(rhs) => Some((rhs, lhs)),
        _ => None,
    };
    if let Some((scalar, values)) = divisor {
        return conversion::scale_real(
            orientation,
            values,
            tensor::tensor_value_f64(scalar, 0).recip(),
        );
    }

    validation::ensure_matching_real_dimension(orientation, lhs, rhs)?;
    if validation::has_empty_solve_dimension(orientation, lhs, rhs) {
        let (rows, cols) = validation::real_output_dimensions(orientation, lhs, rhs);
        return Tensor::new(vec![0.0; rows * cols], vec![rows, cols]).map_err(|error| {
            errors::internal(orientation, format!("{}: {error}", orientation.name()))
        });
    }

    let lhs_values = tensor::tensor_values_f64_cow(lhs);
    let rhs_values = tensor::tensor_values_f64_cow(rhs);
    let lhs_matrix = DMatrix::from_column_slice(lhs.rows(), lhs.cols(), lhs_values.as_ref());
    let rhs_matrix = DMatrix::from_column_slice(rhs.rows(), rhs.cols(), rhs_values.as_ref());
    let solution = solver::solve(orientation, &lhs_matrix, &rhs_matrix)?;
    conversion::real_tensor(orientation, solution, result_dtype(lhs, rhs))
}

fn result_dtype(lhs: &Tensor, rhs: &Tensor) -> NumericDType {
    if lhs.numeric_dtype() == NumericDType::F32 || rhs.numeric_dtype() == NumericDType::F32 {
        NumericDType::F32
    } else {
        NumericDType::F64
    }
}
