use super::*;

pub(super) fn evaluate_complex_matrix(
    matrix: ComplexMatrix,
    rows: CorrcoefRows,
) -> BuiltinResult<ComplexTensor> {
    let matrix = match rows {
        CorrcoefRows::Complete => filter_complete_complex_rows(&matrix),
        CorrcoefRows::All | CorrcoefRows::Pairwise => matrix,
    };
    let cols = matrix.cols;
    let mut coefficients = vec![Complex64::new(f64::NAN, f64::NAN); cols * cols];
    for lhs in 0..cols {
        for rhs in lhs..cols {
            let mut coefficient = complex_pair(&matrix, lhs, rhs, rows == CorrcoefRows::Pairwise);
            if lhs == rhs && complex_is_finite(coefficient) {
                coefficient = Complex64::new(1.0, 0.0);
            }
            set_hermitian(&mut coefficients, cols, lhs, rhs, coefficient);
        }
    }
    ComplexTensor::new(
        coefficients
            .into_iter()
            .map(|value| (value.re, value.im))
            .collect(),
        vec![cols, cols],
    )
    .map_err(corrcoef_internal_error)
}

fn filter_complete_complex_rows(matrix: &ComplexMatrix) -> ComplexMatrix {
    let valid = (0..matrix.rows)
        .filter(|&row| (0..matrix.cols).all(|col| !complex_has_nan(matrix.get(row, col))))
        .collect::<Vec<_>>();
    let mut data = Vec::with_capacity(valid.len() * matrix.cols);
    for col in 0..matrix.cols {
        data.extend(valid.iter().map(|&row| matrix.get(row, col)));
    }
    ComplexMatrix {
        data,
        rows: valid.len(),
        cols: matrix.cols,
    }
}

fn complex_pair(matrix: &ComplexMatrix, lhs: usize, rhs: usize, pairwise: bool) -> Complex64 {
    let mut xs = Vec::with_capacity(matrix.rows);
    let mut ys = Vec::with_capacity(matrix.rows);
    for row in 0..matrix.rows {
        let x = matrix.get(row, lhs);
        let y = matrix.get(row, rhs);
        if !complex_has_nan(x) && !complex_has_nan(y) {
            xs.push(x);
            ys.push(y);
        } else if !pairwise {
            return Complex64::new(f64::NAN, f64::NAN);
        }
    }
    complex_correlation(&xs, &ys)
}

fn complex_correlation(xs: &[Complex64], ys: &[Complex64]) -> Complex64 {
    let count = xs.len().min(ys.len());
    if count < 2 {
        return Complex64::new(f64::NAN, f64::NAN);
    }
    let mean_x = xs.iter().take(count).copied().sum::<Complex64>() / count as f64;
    let mean_y = ys.iter().take(count).copied().sum::<Complex64>() / count as f64;
    let mut sum_xx = 0.0;
    let mut sum_yy = 0.0;
    let mut sum_xy = Complex64::new(0.0, 0.0);
    for index in 0..count {
        let dx = xs[index] - mean_x;
        let dy = ys[index] - mean_y;
        sum_xx += dx.norm_sqr();
        sum_yy += dy.norm_sqr();
        sum_xy += dx.conj() * dy;
    }
    if sum_xx <= 0.0 || sum_yy <= 0.0 {
        Complex64::new(f64::NAN, f64::NAN)
    } else {
        sum_xy / (sum_xx.sqrt() * sum_yy.sqrt())
    }
}

fn complex_is_finite(value: Complex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

fn complex_has_nan(value: Complex64) -> bool {
    value.re.is_nan() || value.im.is_nan()
}
