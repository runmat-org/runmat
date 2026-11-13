// General dense solves and shared-factor reciprocal-condition estimation.
use super::*;
use crate::builtins::common::linalg::{
    one_norm::{binary_lu_scale, reciprocal_one_norm_from_lu},
    singular_value_rcond,
};
use nalgebra::linalg::SVD;

pub(super) fn solve_general_real(lhs: &Tensor, rhs: &Tensor) -> BuiltinResult<(Tensor, f64, f64)> {
    let order = lhs.rows();
    if order == lhs.cols() {
        if order == 0 {
            return Ok((
                real_solution_tensor(
                    Vec::new(),
                    vec![0, rhs.cols()],
                    real_solution_dtype(lhs, rhs),
                )?,
                1.0,
                0.0,
            ));
        }
        if let Some((solution, rcond)) = solve_square_real_lu(lhs, rhs)? {
            return Ok((solution, rcond, order as f64));
        }
    }
    solve_general_real_svd(lhs, rhs)
}

fn solve_square_real_lu(lhs: &Tensor, rhs: &Tensor) -> BuiltinResult<Option<(Tensor, f64)>> {
    let mut matrix = real_matrix_from_tensor(lhs);
    let mut right = real_matrix_from_tensor(rhs);
    if matrix
        .iter()
        .chain(right.iter())
        .any(|value| !value.is_finite())
    {
        // Preserve the preexisting nonfinite SVD path until its public
        // convergence/error policy is independently specified.
        return Ok(None);
    }
    let maximum = matrix.iter().copied().map(f64::abs).fold(0.0, f64::max);
    let Some(candidate_scale) = binary_lu_scale(maximum) else {
        return Ok(None);
    };
    let scale = if matrix
        .iter()
        .chain(right.iter())
        .all(|&value| binary_scale_preserves_component(value, candidate_scale))
    {
        candidate_scale
    } else {
        1.0
    };
    if scale != 1.0 {
        matrix.apply(|value| *value /= scale);
        right.apply(|value| *value /= scale);
    }
    let one_norm = matrix
        .column_iter()
        .map(|column| column.iter().map(|value| value.abs()).sum::<f64>())
        .fold(0.0, f64::max);
    let factor = matrix.lu();
    if factor.lu_internal().iter().any(|value| !value.is_finite()) {
        return Err(builtin_error(
            "linsolve: LU factor exceeds finite floating range",
        ));
    }
    if !factor.is_invertible() {
        return Ok(None);
    }
    let Some(solution) = factor.solve(&right) else {
        return Ok(None);
    };
    if solution.iter().any(|value| !value.is_finite()) {
        return Err(builtin_error(
            "linsolve: square solution exceeds finite floating range",
        ));
    }
    let rcond = reciprocal_lu_condition(&factor, one_norm);
    Ok(Some((
        matrix_real_to_tensor(solution, real_solution_dtype(lhs, rhs))?,
        rcond,
    )))
}

fn solve_general_real_svd(lhs: &Tensor, rhs: &Tensor) -> BuiltinResult<(Tensor, f64, f64)> {
    let a = real_matrix_from_tensor(lhs);
    let b = real_matrix_from_tensor(rhs);
    let svd = SVD::new(a, true, true);
    let rcond = singular_value_rcond(svd.singular_values.as_slice());
    let tol = compute_svd_tolerance(svd.singular_values.as_slice(), lhs.rows(), lhs.cols());
    let rank = svd
        .singular_values
        .iter()
        .filter(|value| **value > tol)
        .count() as f64;
    let solution = svd
        .solve(&b, tol)
        .map_err(|e| builtin_error(format!("{NAME}: {e}")))?;
    let tensor = matrix_real_to_tensor(solution, real_solution_dtype(lhs, rhs))?;
    Ok((tensor, rcond, rank))
}

fn real_matrix_from_tensor(tensor: &Tensor) -> DMatrix<f64> {
    if let Some(values) = tensor.as_f64_slice() {
        DMatrix::from_column_slice(tensor.rows(), tensor.cols(), values)
    } else if let Some(values) = tensor.as_f32_slice() {
        DMatrix::from_iterator(
            tensor.rows(),
            tensor.cols(),
            values.iter().map(|&value| f64::from(value)),
        )
    } else if let Some(values) = tensor.integer_storage() {
        DMatrix::from_iterator(
            tensor.rows(),
            tensor.cols(),
            (0..tensor.len()).map(|index| {
                values
                    .value_at(index)
                    .expect("validated integer tensor storage")
                    .to_f64()
            }),
        )
    } else {
        let values = tensor::tensor_values_f64_cow(tensor);
        DMatrix::from_column_slice(tensor.rows(), tensor.cols(), values.as_ref())
    }
}

pub(super) fn solve_general_complex(
    lhs: &ComplexTensor,
    rhs: &ComplexTensor,
) -> BuiltinResult<(ComplexTensor, f64, f64)> {
    let order = lhs.rows();
    if order == lhs.cols() {
        if order == 0 {
            return Ok((
                complex_solution_tensor(DMatrix::zeros(0, rhs.cols()), lhs, rhs)?,
                1.0,
                0.0,
            ));
        }
        if let Some((solution, rcond)) = solve_square_complex_lu(lhs, rhs)? {
            return Ok((solution, rcond, order as f64));
        }
    }
    solve_general_complex_svd(lhs, rhs)
}

fn solve_square_complex_lu(
    lhs: &ComplexTensor,
    rhs: &ComplexTensor,
) -> BuiltinResult<Option<(ComplexTensor, f64)>> {
    let mut matrix = complex_matrix_from_tensor(lhs);
    if matrix
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Ok(None);
    }
    let maximum = matrix
        .iter()
        .map(|value| value.re.abs().max(value.im.abs()))
        .fold(0.0, f64::max);
    let Some(candidate_scale) = binary_lu_scale(maximum) else {
        return Ok(None);
    };
    let mut right = complex_matrix_from_tensor(rhs);
    if right
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Ok(None);
    }
    let scale = if matrix.iter().chain(right.iter()).all(|value| {
        binary_scale_preserves_component(value.re, candidate_scale)
            && binary_scale_preserves_component(value.im, candidate_scale)
    }) {
        candidate_scale
    } else {
        1.0
    };
    if scale != 1.0 {
        matrix.apply(|value| *value /= scale);
        right.apply(|value| *value /= scale);
    }
    let one_norm = matrix
        .column_iter()
        .map(|column| column.iter().map(|value| value.norm()).sum::<f64>())
        .fold(0.0, f64::max);
    let factor = matrix.lu();
    if factor
        .lu_internal()
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(builtin_error(
            "linsolve: LU factor exceeds finite floating range",
        ));
    }
    if !factor.is_invertible() {
        return Ok(None);
    }
    let Some(solution) = factor.solve(&right) else {
        return Ok(None);
    };
    if solution
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(builtin_error(
            "linsolve: square solution exceeds finite floating range",
        ));
    }
    let rcond = reciprocal_lu_condition(&factor, one_norm);
    Ok(Some((complex_solution_tensor(solution, lhs, rhs)?, rcond)))
}

pub(super) fn binary_scale_preserves_component(value: f64, scale: f64) -> bool {
    let scaled = value / scale;
    scaled.is_finite() && (value == 0.0 || scaled != 0.0) && !(scale > 1.0 && scaled.is_subnormal())
}

fn reciprocal_lu_condition<T>(
    factor: &nalgebra::linalg::LU<T, nalgebra::Dyn, nalgebra::Dyn>,
    matrix_one_norm: f64,
) -> f64
where
    T: nalgebra::ComplexField<RealField = f64> + Copy,
{
    if !matrix_one_norm.is_finite() {
        return 0.0;
    }
    let estimate = reciprocal_one_norm_from_lu(factor, matrix_one_norm);
    if estimate.is_finite() {
        estimate.max(0.0)
    } else {
        0.0
    }
}

fn solve_general_complex_svd(
    lhs: &ComplexTensor,
    rhs: &ComplexTensor,
) -> BuiltinResult<(ComplexTensor, f64, f64)> {
    let a = complex_matrix_from_tensor(lhs);
    let b = complex_matrix_from_tensor(rhs);
    let svd = SVD::new(a, true, true);
    let rcond = singular_value_rcond(svd.singular_values.as_slice());
    let tol = compute_svd_tolerance(svd.singular_values.as_slice(), lhs.rows(), lhs.cols());
    let rank = svd
        .singular_values
        .iter()
        .filter(|value| **value > tol)
        .count() as f64;
    let solution = svd
        .solve(&b, tol)
        .map_err(|e| builtin_error(format!("{NAME}: {e}")))?;
    let tensor = complex_solution_tensor(solution, lhs, rhs)?;
    Ok((tensor, rcond, rank))
}

fn complex_matrix_from_tensor(tensor: &ComplexTensor) -> DMatrix<Complex64> {
    if let Some(values) = tensor.as_f64_slice() {
        DMatrix::from_iterator(
            tensor.rows(),
            tensor.cols(),
            values.iter().map(|value| Complex64::new(value.0, value.1)),
        )
    } else if let Some(values) = tensor.as_f32_slice() {
        DMatrix::from_iterator(
            tensor.rows(),
            tensor.cols(),
            values
                .iter()
                .map(|value| Complex64::new(f64::from(value.0), f64::from(value.1))),
        )
    } else {
        DMatrix::from_iterator(
            tensor.rows(),
            tensor.cols(),
            tensor
                .materialize_f64()
                .into_iter()
                .map(|(re, im)| Complex64::new(re, im)),
        )
    }
}
