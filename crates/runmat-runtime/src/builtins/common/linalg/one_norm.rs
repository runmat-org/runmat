//! Reciprocal 1-norm condition estimate from one pivoted LU factorization.
//!
//! The inverse norm is estimated by the Hager/Higham iteration used by
//! LAPACK's xLACN2 family. Each iteration solves with `A` and `Aᴴ`; the
//! adjoint solve uses the same packed LU factors and permutation as the
//! ordinary solve. Only vectors, not an inverse matrix, are materialized.

use nalgebra::{linalg::LU, ComplexField, DMatrix, DVector, Dyn};

const MAX_ESTIMATE_ITERATIONS: usize = 5;

/// A finite binary power near the largest input component. Dividing by this
/// scale does not perturb values near one as division by an arbitrary matrix
/// maximum would. Clamp the exponent because the next power above f64::MAX
/// is not representable, while subnormals share the minimum normal scale.
pub(crate) fn binary_lu_scale(max_component: f64) -> Option<f64> {
    if !max_component.is_finite() || max_component <= 0.0 {
        return None;
    }
    let exponent_field = ((max_component.to_bits() >> 52) & 0x7ff) as i32;
    let exponent = if exponent_field == 0 {
        -1022
    } else {
        exponent_field - 1023
    };
    Some(f64::from_bits(((exponent + 1023) as u64) << 52))
}

pub(crate) fn reciprocal_one_norm<T>(mut matrix: DMatrix<T>) -> f64
where
    T: ComplexField<RealField = f64> + Copy,
{
    let order = matrix.nrows();
    if order == 0 {
        return f64::INFINITY;
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return f64::NAN;
    }

    let scale = matrix
        .iter()
        // Complex modulus can overflow although both stored components are
        // finite. Choose the binary scale from authoritative components first.
        .map(|value| (*value).real().abs().max((*value).imaginary().abs()))
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return 0.0;
    }
    if !scale.is_finite() {
        return f64::NAN;
    }
    // Scaling keeps the 1-norm and LU solves away from avoidable overflow;
    // rcond is invariant under multiplication by a nonzero scalar.
    let scale = binary_lu_scale(scale).expect("finite positive matrix maximum");
    // Divide by a real scalar, not a complex value with a zero imaginary
    // component: complex division can overflow/underflow its denominator
    // even when the scaled components are finite.
    matrix.apply(|value| *value = (*value).unscale(scale));
    let matrix_norm = matrix
        .column_iter()
        .map(|column| column.iter().map(|value| (*value).modulus()).sum::<f64>())
        .fold(0.0_f64, f64::max);
    let lu = matrix.lu();
    if !lu.is_invertible() {
        return 0.0;
    }

    reciprocal_one_norm_from_lu(&lu, matrix_norm)
}

/// Estimate reciprocal condition from an already factored, scaled matrix.
/// The caller supplies the 1-norm of that same scaled matrix. This permits
/// linear solves and `rcond` to share one factorization and one estimator.
pub(crate) fn reciprocal_one_norm_from_lu<T>(lu: &LU<T, Dyn, Dyn>, matrix_norm: f64) -> f64
where
    T: ComplexField<RealField = f64> + Copy,
{
    let order = lu.lu_internal().nrows();
    if order == 0 {
        return f64::INFINITY;
    }
    if !lu.is_invertible() {
        return 0.0;
    }

    reciprocal_one_norm_from_solves(
        order,
        matrix_norm,
        |rhs| lu.solve(rhs),
        |rhs| solve_adjoint(lu, rhs),
    )
}

/// The common bounded inverse-1-norm iteration. Factor owners supply solves
/// with the matrix and its adjoint; no inverse matrix is materialized.
pub(crate) fn reciprocal_one_norm_from_solves<T, Solve, Adjoint>(
    order: usize,
    matrix_norm: f64,
    mut solve: Solve,
    mut solve_adjoint: Adjoint,
) -> f64
where
    T: ComplexField<RealField = f64> + Copy,
    Solve: FnMut(&DVector<T>) -> Option<DVector<T>>,
    Adjoint: FnMut(&DVector<T>) -> Option<DVector<T>>,
{
    if order == 0 {
        return f64::INFINITY;
    }
    if matrix_norm == 0.0 {
        return 0.0;
    }

    let mut vector = DVector::from_element(order, T::from_real(1.0 / order as f64));
    let mut estimate = 0.0_f64;
    let mut previous_index = None;
    for _ in 0..MAX_ESTIMATE_ITERATIONS {
        let Some(solution) = solve(&vector) else {
            return 0.0;
        };
        estimate = estimate.max(vector_one_norm(&solution));
        let signs = solution.map(|value| {
            let magnitude = value.modulus();
            if magnitude == 0.0 {
                T::one()
            } else {
                value.unscale(magnitude)
            }
        });
        let Some(adjoint_solution) = solve_adjoint(&signs) else {
            return 0.0;
        };
        let index = adjoint_solution
            .iter()
            .enumerate()
            .max_by(|left, right| (*left.1).modulus().total_cmp(&(*right.1).modulus()))
            .map(|(index, _)| index)
            .expect("nonempty condition estimate");
        if previous_index == Some(index) {
            break;
        }
        previous_index = Some(index);
        vector.fill(T::zero());
        vector[index] = T::one();
    }

    if order > 1 {
        let alternating = DVector::from_iterator(
            order,
            (0..order).map(|index| {
                let magnitude = 1.0 + index as f64 / (order - 1) as f64;
                T::from_real(if index % 2 == 0 {
                    magnitude
                } else {
                    -magnitude
                })
            }),
        );
        let Some(solution) = solve(&alternating) else {
            return 0.0;
        };
        estimate = estimate.max(2.0 * vector_one_norm(&solution) / (3.0 * order as f64));
    }

    if estimate.is_nan() || matrix_norm.is_nan() {
        f64::NAN
    } else {
        (1.0 / matrix_norm) / estimate
    }
}

fn vector_one_norm<T>(vector: &DVector<T>) -> f64
where
    T: ComplexField<RealField = f64> + Copy,
{
    vector.iter().map(|value| (*value).modulus()).sum()
}

fn solve_adjoint<T>(lu: &LU<T, Dyn, Dyn>, rhs: &DVector<T>) -> Option<DVector<T>>
where
    T: ComplexField<RealField = f64> + Copy,
{
    // P*A = L*U, so Aᴴ = Uᴴ*Lᴴ*P. Forward-solve Uᴴ, backward-solve
    // unit-diagonal Lᴴ, then undo P. The packed factors are borrowed.
    let packed = lu.lu_internal();
    let order = rhs.len();
    let mut solution = rhs.clone();
    for row in 0..order {
        let mut value = solution[row];
        for column in 0..row {
            value -= packed[(column, row)].conjugate() * solution[column];
        }
        let diagonal = packed[(row, row)].conjugate();
        if diagonal.modulus() == 0.0 {
            return None;
        }
        solution[row] = value / diagonal;
    }
    for row in (0..order).rev() {
        let mut value = solution[row];
        for column in row + 1..order {
            value -= packed[(column, row)].conjugate() * solution[column];
        }
        solution[row] = value;
    }
    lu.p().inv_permute_rows(&mut solution);
    Some(solution)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn inverse_one_norm_matches_small_independent_reference() {
        let real = DMatrix::from_column_slice(2, 2, &[1.0, 3.0, 2.0, 4.0]);
        let actual = reciprocal_one_norm(real);
        // ||A||_1 = 6; ||A^-1||_1 = 3.5.
        assert!((actual - 1.0 / 21.0).abs() < 1e-12);
        let unfactored = DMatrix::from_column_slice(2, 2, &[1.0, 3.0, 2.0, 4.0]);
        let factor = unfactored.lu();
        assert!((reciprocal_one_norm_from_lu(&factor, 6.0) - actual).abs() < 1e-12);

        assert_eq!(binary_lu_scale(1.0 + 1e-12), Some(1.0));
        assert_eq!(
            binary_lu_scale(f64::MAX),
            Some(f64::from_bits(2046_u64 << 52))
        );
        assert_eq!(binary_lu_scale(f64::from_bits(1)), Some(f64::MIN_POSITIVE));

        let complex = DMatrix::from_column_slice(
            2,
            2,
            &[
                Complex64::new(1.0, 2.0),
                Complex64::new(0.0, 3.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(2.0, -1.0),
            ],
        );
        // This reference independently evaluates the two inverse columns.
        let inverse_norm = (1.0 / Complex64::new(1.0, 2.0).norm()
            + Complex64::new(0.0, 3.0).norm()
                / Complex64::new(1.0, 2.0).norm()
                / Complex64::new(2.0, -1.0).norm())
        .max(1.0 / Complex64::new(2.0, -1.0).norm());
        let matrix_norm =
            (Complex64::new(1.0, 2.0).norm() + 3.0).max(Complex64::new(2.0, -1.0).norm());
        let expected = 1.0 / (matrix_norm * inverse_norm);
        assert!((reciprocal_one_norm(complex) - expected).abs() < 1e-12);
    }

    #[test]
    fn finite_large_complex_scalar_keeps_unit_reciprocal_condition() {
        for component in [1e308, f64::MAX, f64::from_bits(1)] {
            let matrix = DMatrix::from_element(1, 1, Complex64::new(component, component));
            let actual = reciprocal_one_norm(matrix);
            assert!(
                (actual - 1.0).abs() < 1e-12,
                "component {component:e}: reciprocal condition was {actual}"
            );
        }
    }
}
