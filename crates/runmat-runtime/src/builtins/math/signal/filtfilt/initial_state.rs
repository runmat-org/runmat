//! The one initial-state system used by host and resident filtfilt routes.
use num_complex::Complex;

use super::{
    filtfilt_error_with_detail, NormalizedCoefficients, FILTFILT_ERROR_INVALID_COEFFICIENTS,
};
use crate::BuiltinResult;

fn finite(value: Complex<f64>) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

fn range_error() -> crate::RuntimeError {
    filtfilt_error_with_detail(
        &FILTFILT_ERROR_INVALID_COEFFICIENTS,
        "filter initial-state system exceeds the floating-point range",
    )
}

fn singular_error() -> crate::RuntimeError {
    filtfilt_error_with_detail(
        &FILTFILT_ERROR_INVALID_COEFFICIENTS,
        "filter initial-state system is singular",
    )
}

pub(super) fn lfilter_zi(coeffs: &NormalizedCoefficients) -> BuiltinResult<Vec<Complex<f64>>> {
    let n = coeffs.state_len;
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut matrix = vec![vec![Complex::new(0.0, 0.0); n]; n];
    for (row, row_values) in matrix.iter_mut().enumerate() {
        for (col, value) in row_values.iter_mut().enumerate() {
            if row == col {
                *value += Complex::new(1.0, 0.0);
            }
            let a_t = if col == 0 {
                -coeffs.a[row + 1]
            } else if row + 1 == col {
                Complex::new(1.0, 0.0)
            } else {
                Complex::new(0.0, 0.0)
            };
            *value -= a_t;
        }
    }
    let rhs = (0..n)
        .map(|idx| coeffs.b[idx + 1] - coeffs.a[idx + 1] * coeffs.b[0])
        .collect::<Vec<_>>();
    solve_linear(matrix, rhs)
}

fn solve_linear(
    mut matrix: Vec<Vec<Complex<f64>>>,
    mut rhs: Vec<Complex<f64>>,
) -> BuiltinResult<Vec<Complex<f64>>> {
    let n = rhs.len();
    for (row, values) in matrix.iter_mut().enumerate() {
        let scale = values.iter().fold(0.0_f64, |largest, value| {
            largest.max(value.re.abs()).max(value.im.abs())
        });
        if !scale.is_finite() || !finite(rhs[row]) {
            return Err(range_error());
        }
        if scale == 0.0 {
            return Err(singular_error());
        }
        for value in values {
            *value = Complex::new(value.re / scale, value.im / scale);
        }
        rhs[row] = Complex::new(rhs[row].re / scale, rhs[row].im / scale);
        if !finite(rhs[row]) {
            return Err(range_error());
        }
    }

    for pivot_idx in 0..n {
        let Some(pivot_row) = (pivot_idx..n).max_by(|&left, &right| {
            matrix[left][pivot_idx]
                .norm()
                .total_cmp(&matrix[right][pivot_idx].norm())
        }) else {
            return Err(singular_error());
        };
        if matrix[pivot_row][pivot_idx] == Complex::new(0.0, 0.0) {
            return Err(singular_error());
        }
        if pivot_row != pivot_idx {
            matrix.swap(pivot_row, pivot_idx);
            rhs.swap(pivot_row, pivot_idx);
        }
        let pivot = matrix[pivot_idx][pivot_idx];
        let pivot_row_values = matrix[pivot_idx].clone();
        let pivot_rhs = rhs[pivot_idx];
        for row in pivot_idx + 1..n {
            let factor = matrix[row][pivot_idx] / pivot;
            for (col, pivot_value) in pivot_row_values.iter().enumerate().skip(pivot_idx) {
                matrix[row][col] -= factor * *pivot_value;
                if !finite(matrix[row][col]) {
                    return Err(range_error());
                }
            }
            rhs[row] -= factor * pivot_rhs;
            if !finite(rhs[row]) {
                return Err(range_error());
            }
        }
    }

    let mut out = vec![Complex::new(0.0, 0.0); n];
    for row in (0..n).rev() {
        let mut acc = rhs[row];
        for (col, value) in matrix[row].iter().enumerate().skip(row + 1) {
            acc -= *value * out[col];
        }
        let pivot = matrix[row][row];
        if pivot == Complex::new(0.0, 0.0) {
            return Err(singular_error());
        }
        out[row] = acc / pivot;
        if !finite(out[row]) {
            return Err(range_error());
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_nonzero_pivot_remains_admissible_without_fixed_epsilon() {
        let solution = solve_linear(
            vec![vec![Complex::new(1.0e-14, 0.0)]],
            vec![Complex::new(1.0, 0.0)],
        )
        .expect("finite one-state solution");
        assert!((solution[0].re - 1.0e14).abs() <= 1.0);
        assert_eq!(solution[0].im, 0.0);
        let coeffs = NormalizedCoefficients::new(
            &[Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)],
            &[Complex::new(1.0, 0.0), Complex::new(-(1.0 - 1.0e-14), 0.0)],
        )
        .expect("near-unit-root coefficients");
        let state = lfilter_zi(&coeffs).expect("finite near-unit-root initial state");
        assert!(state[0].re.is_finite());
        assert!(state[0].re > 1.0e13);
        let complex = solve_linear(
            vec![vec![Complex::new(1.0e-14, 1.0e-14)]],
            vec![Complex::new(1.0, 1.0)],
        )
        .expect("finite complex initial state");
        assert!((complex[0].re - 1.0e14).abs() <= 1.0);
        assert!(complex[0].im.abs() <= 1.0);
    }

    #[test]
    fn zero_and_nonrepresentable_systems_reject_without_publishing_nan() {
        assert!(solve_linear(
            vec![vec![Complex::new(0.0, 0.0)]],
            vec![Complex::new(1.0, 0.0)],
        )
        .is_err());
        assert!(solve_linear(
            vec![vec![Complex::new(f64::MIN_POSITIVE, 0.0)]],
            vec![Complex::new(f64::MAX, 0.0)],
        )
        .is_err());
    }
}
