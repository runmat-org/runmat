//! Reciprocal 1-norm condition of a real square solve from retained factors.
//! The provider owns its QR/Cholesky result; this host adapter only consumes
//! borrowed, explicitly downloaded factors when the condition is requested.

use crate::builtins::common::{
    linalg::one_norm::{binary_lu_scale, reciprocal_one_norm_from_solves},
    tensor,
};
use crate::BuiltinResult;
use nalgebra::DVector;
use runmat_value::{NumericDType, Tensor};

pub enum ProviderSquareFactor<'a> {
    Qr { q: &'a Tensor, r: &'a Tensor },
    Cholesky { r: &'a Tensor },
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum FactorConditionState {
    Nonsingular,
    Singular,
}

/// Evaluate the same bounded inverse-1-norm estimator used by the LU host
/// route without recomputing a factorization, SVD, or full inverse.
pub fn square_factor_rcond_for_provider(
    effective_matrix: &Tensor,
    factor: ProviderSquareFactor<'_>,
) -> BuiltinResult<f64> {
    let order = effective_matrix.rows();
    validate_square(effective_matrix, order)?;
    if order == 0 {
        return Ok(f64::INFINITY);
    }
    let matrix = tensor::tensor_values_f64_cow(effective_matrix);
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(invalid_factor("nonfinite coefficient matrix"));
    }
    // Retained factors describe A, while the inverse-norm iteration should
    // operate on A / scale. This avoids overflowing finite column sums (and
    // overflowing inverse solves for very small A) without refactoring or
    // materializing a second matrix. A binary scale also preserves components
    // near unity that an arbitrary maximum could round differently.
    let maximum = matrix.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);
    let scale = binary_lu_scale(maximum).unwrap_or(1.0);
    let matrix_norm = matrix
        .chunks_exact(order)
        .map(|column| column.iter().map(|value| value.abs() / scale).sum::<f64>())
        .fold(0.0_f64, f64::max);
    match factor {
        ProviderSquareFactor::Qr { q, r } => {
            validate_square(q, order)?;
            validate_square(r, order)?;
            let q = tensor::tensor_values_f64_cow(q);
            let r = tensor::tensor_values_f64_cow(r);
            if validate_factor(&q, &r, order)? == FactorConditionState::Singular {
                return Ok(0.0);
            }
            Ok(reciprocal_one_norm_from_solves(
                order,
                matrix_norm,
                |rhs| {
                    let projected = multiply_transposed(&q, rhs, order)?;
                    solve_upper(&r, &projected, order, scale)
                },
                |rhs| {
                    let projected = solve_upper_transposed(&r, rhs, order, scale)?;
                    multiply(&q, &projected, order)
                },
            ))
        }
        ProviderSquareFactor::Cholesky { r } => {
            validate_square(r, order)?;
            let r = tensor::tensor_values_f64_cow(r);
            if validate_upper(&r, order)? == FactorConditionState::Singular {
                return Ok(0.0);
            }
            let factor_scale = scale.sqrt();
            Ok(reciprocal_one_norm_from_solves(
                order,
                matrix_norm,
                |rhs| {
                    let projected = solve_upper_transposed(&r, rhs, order, factor_scale)?;
                    solve_upper(&r, &projected, order, factor_scale)
                },
                |rhs| {
                    let projected = solve_upper_transposed(&r, rhs, order, factor_scale)?;
                    solve_upper(&r, &projected, order, factor_scale)
                },
            ))
        }
    }
}

fn invalid_factor(message: &'static str) -> crate::RuntimeError {
    super::builtin_error(format!(
        "linsolve: invalid retained square factor ({message})"
    ))
}

fn validate_square(tensor: &Tensor, order: usize) -> BuiltinResult<()> {
    let expected_len = order
        .checked_mul(order)
        .ok_or_else(|| invalid_factor("shape exceeds addressable storage"))?;
    if tensor.shape().len() > 2
        || tensor.rows() != order
        || tensor.cols() != order
        || tensor.len() != expected_len
        || !matches!(
            tensor.numeric_dtype(),
            NumericDType::F32 | NumericDType::F64
        )
    {
        return Err(invalid_factor("shape or class"));
    }
    Ok(())
}

fn validate_factor(q: &[f64], r: &[f64], order: usize) -> BuiltinResult<FactorConditionState> {
    if q.iter().any(|value| !value.is_finite()) {
        return Err(invalid_factor("nonfinite QR factor"));
    }
    validate_upper(r, order)
}

fn validate_upper(r: &[f64], order: usize) -> BuiltinResult<FactorConditionState> {
    if r.len() != order * order || r.iter().any(|value| !value.is_finite()) {
        return Err(invalid_factor("nonfinite or malformed triangular factor"));
    }
    if (0..order).any(|index| r[index + index * order] == 0.0) {
        return Ok(FactorConditionState::Singular);
    }
    Ok(FactorConditionState::Nonsingular)
}

fn multiply(matrix: &[f64], rhs: &DVector<f64>, order: usize) -> Option<DVector<f64>> {
    let mut result = DVector::<f64>::zeros(order);
    for column in 0..order {
        for row in 0..order {
            result[row] += matrix[row + column * order] * rhs[column];
        }
    }
    result
        .iter()
        .all(|value| value.is_finite())
        .then_some(result)
}

fn multiply_transposed(matrix: &[f64], rhs: &DVector<f64>, order: usize) -> Option<DVector<f64>> {
    let mut result = DVector::<f64>::zeros(order);
    for column in 0..order {
        for row in 0..order {
            result[column] += matrix[row + column * order] * rhs[row];
        }
    }
    result
        .iter()
        .all(|value| value.is_finite())
        .then_some(result)
}

fn solve_upper(
    r: &[f64],
    rhs: &DVector<f64>,
    order: usize,
    factor_scale: f64,
) -> Option<DVector<f64>> {
    let mut result = rhs.clone();
    for row in (0..order).rev() {
        let mut value = result[row];
        for column in row + 1..order {
            value -= (r[row + column * order] / factor_scale) * result[column];
        }
        result[row] = value / (r[row + row * order] / factor_scale);
    }
    result
        .iter()
        .all(|value| value.is_finite())
        .then_some(result)
}

fn solve_upper_transposed(
    r: &[f64],
    rhs: &DVector<f64>,
    order: usize,
    factor_scale: f64,
) -> Option<DVector<f64>> {
    let mut result = rhs.clone();
    for row in 0..order {
        let mut value = result[row];
        for column in 0..row {
            value -= (r[column + row * order] / factor_scale) * result[column];
        }
        result[row] = value / (r[row + row * order] / factor_scale);
    }
    result
        .iter()
        .all(|value| value.is_finite())
        .then_some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_qr_and_cholesky_estimate_inverse_one_norm() {
        let a = Tensor::new(vec![3.0, 1.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let q = Tensor::new(
            vec![
                3.0 / 10.0_f64.sqrt(),
                1.0 / 10.0_f64.sqrt(),
                -1.0 / 10.0_f64.sqrt(),
                3.0 / 10.0_f64.sqrt(),
            ],
            vec![2, 2],
        )
        .unwrap();
        let r = Tensor::new(
            vec![10.0_f64.sqrt(), 0.0, 10.0_f64.sqrt(), 10.0_f64.sqrt()],
            vec![2, 2],
        )
        .unwrap();
        let estimate =
            square_factor_rcond_for_provider(&a, ProviderSquareFactor::Qr { q: &q, r: &r })
                .unwrap();
        assert!((estimate - 1.0 / 3.0).abs() < 1e-12);

        let spd = Tensor::new(vec![4.0, 1.0, 1.0, 3.0], vec![2, 2]).unwrap();
        let chol = Tensor::new(vec![2.0, 0.0, 0.5, 2.75_f64.sqrt()], vec![2, 2]).unwrap();
        let estimate =
            square_factor_rcond_for_provider(&spd, ProviderSquareFactor::Cholesky { r: &chol })
                .unwrap();
        assert!((estimate - 0.44).abs() < 1e-12);

        let singular = Tensor::new(vec![1.0, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
        let identity = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        assert_eq!(
            square_factor_rcond_for_provider(
                &singular,
                ProviderSquareFactor::Qr {
                    q: &identity,
                    r: &singular,
                },
            )
            .unwrap(),
            0.0
        );
        assert_eq!(
            square_factor_rcond_for_provider(
                &singular,
                ProviderSquareFactor::Cholesky { r: &singular },
            )
            .unwrap(),
            0.0
        );
    }

    #[test]
    fn retained_factors_normalize_large_and_small_finite_matrices() {
        // The first A column has a finite pair whose unscaled 1-norm
        // overflows f64, even though A and both QR factors are finite.
        let magnitude = 1.0e308;
        let root_two = 2.0_f64.sqrt();
        let a = Tensor::new(vec![magnitude, magnitude, 0.0, magnitude], vec![2, 2]).unwrap();
        let q = Tensor::new(
            vec![
                1.0 / root_two,
                1.0 / root_two,
                -1.0 / root_two,
                1.0 / root_two,
            ],
            vec![2, 2],
        )
        .unwrap();
        let r = Tensor::new(
            vec![
                root_two * magnitude,
                0.0,
                magnitude / root_two,
                magnitude / root_two,
            ],
            vec![2, 2],
        )
        .unwrap();
        let qr = square_factor_rcond_for_provider(&a, ProviderSquareFactor::Qr { q: &q, r: &r })
            .unwrap();
        // This is a bounded inverse-norm estimate, not the exact condition
        // number. The sign iteration estimates 1; its alternating-vector
        // probe raises that to 4/3 for the unscaled matrix, giving 3/8.
        assert!((qr - 3.0 / 8.0).abs() < 1.0e-12, "QR estimate: {qr}");

        // For a symmetric positive matrix, ||A||_1 = 1.9e308 but its
        // normalized condition number is exactly 19.
        let spd = Tensor::new(
            vec![magnitude, 0.9 * magnitude, 0.9 * magnitude, magnitude],
            vec![2, 2],
        )
        .unwrap();
        let root = magnitude.sqrt();
        let chol = Tensor::new(
            vec![root, 0.0, 0.9 * root, 0.19_f64.sqrt() * root],
            vec![2, 2],
        )
        .unwrap();
        let chol_estimate =
            square_factor_rcond_for_provider(&spd, ProviderSquareFactor::Cholesky { r: &chol })
                .unwrap();
        assert!((chol_estimate - 1.0 / 19.0).abs() < 1.0e-12);

        // Direct inverse solves for this finite tiny matrix have a vector
        // 1-norm above f64::MAX; normalized retained factors do not.
        // Both columns contribute to the sign iteration for this matrix,
        // avoiding an exact-zero sign whose rounding depends on the QR path.
        // ||A||_1 = 1.9*tiny and ||inv(A)||_1 = 1.9/(0.55*tiny).
        let tiny = 1.0e-309;
        let small = Tensor::new(vec![tiny, 0.9 * tiny, 0.5 * tiny, tiny], vec![2, 2]).unwrap();
        let root = 1.81_f64.sqrt();
        let q = Tensor::new(
            vec![1.0 / root, 0.9 / root, -0.9 / root, 1.0 / root],
            vec![2, 2],
        )
        .unwrap();
        let r = Tensor::new(
            vec![root * tiny, 0.0, 1.4 / root * tiny, 0.55 / root * tiny],
            vec![2, 2],
        )
        .unwrap();
        let small_estimate =
            square_factor_rcond_for_provider(&small, ProviderSquareFactor::Qr { q: &q, r: &r })
                .unwrap();
        assert!(
            (small_estimate - 0.55 / 3.61).abs() < 1.0e-12,
            "tiny QR estimate: {small_estimate}"
        );
    }
}
