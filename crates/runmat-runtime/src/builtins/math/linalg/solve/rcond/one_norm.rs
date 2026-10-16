//! Reciprocal 1-norm condition estimate from one pivoted LU factorization.
//!
//! The inverse norm is estimated by the Hager/Higham iteration used by
//! LAPACK's xLACN2 family. Each iteration solves with `A` and `Aᴴ`; the
//! adjoint solve uses the same packed LU factors and permutation as the
//! ordinary solve. Only vectors, not an inverse matrix, are materialized.

use nalgebra::{linalg::LU, ComplexField, DMatrix, DVector, Dyn};

const MAX_ESTIMATE_ITERATIONS: usize = 5;

pub(super) fn reciprocal_one_norm<T>(mut matrix: DMatrix<T>) -> f64
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
        .map(|value| (*value).modulus())
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return 0.0;
    }
    if !scale.is_finite() {
        return f64::NAN;
    }
    // Scaling keeps the 1-norm and LU solves away from avoidable overflow;
    // rcond is invariant under multiplication by a nonzero scalar.
    matrix.apply(|value| *value /= T::from_real(scale));
    let matrix_norm = matrix
        .column_iter()
        .map(|column| column.iter().map(|value| (*value).modulus()).sum::<f64>())
        .fold(0.0_f64, f64::max);
    let lu = matrix.lu();
    if !lu.is_invertible() {
        return 0.0;
    }

    let mut vector = DVector::from_element(order, T::from_real(1.0 / order as f64));
    let mut estimate = 0.0_f64;
    let mut previous_index = None;
    for _ in 0..MAX_ESTIMATE_ITERATIONS {
        let Some(solution) = lu.solve(&vector) else {
            return 0.0;
        };
        estimate = estimate.max(vector_one_norm(&solution));
        let signs = solution.map(|value| {
            let magnitude = value.modulus();
            if magnitude == 0.0 {
                T::one()
            } else {
                value / T::from_real(magnitude)
            }
        });
        let Some(adjoint_solution) = solve_adjoint(&lu, &signs) else {
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
        let Some(solution) = lu.solve(&alternating) else {
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
}
