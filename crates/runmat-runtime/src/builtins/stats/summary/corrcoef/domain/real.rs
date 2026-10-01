use super::*;

pub(super) fn evaluate_real_matrix(
    matrix: Matrix,
    rows: CorrcoefRows,
) -> BuiltinResult<RealEvaluation> {
    let matrix = match rows {
        CorrcoefRows::Complete => filter_complete_rows(&matrix),
        CorrcoefRows::All | CorrcoefRows::Pairwise => matrix,
    };
    let cols = matrix.cols;
    let mut coefficients = vec![f64::NAN; cols * cols];
    let mut counts = vec![0usize; cols * cols];
    for lhs in 0..cols {
        for rhs in lhs..cols {
            let (coefficient, count) = real_pair(&matrix, lhs, rhs, rows == CorrcoefRows::Pairwise);
            set_symmetric(&mut coefficients, cols, lhs, rhs, coefficient);
            set_symmetric(&mut counts, cols, lhs, rhs, count);
        }
    }
    Ok(RealEvaluation {
        r: Tensor::new(coefficients, vec![cols, cols]).map_err(corrcoef_internal_error)?,
        counts,
    })
}

fn filter_complete_rows(matrix: &Matrix) -> Matrix {
    let valid = (0..matrix.rows)
        .filter(|&row| (0..matrix.cols).all(|col| !matrix.get(row, col).is_nan()))
        .collect::<Vec<_>>();
    let mut data = Vec::with_capacity(valid.len() * matrix.cols);
    for col in 0..matrix.cols {
        data.extend(valid.iter().map(|&row| matrix.get(row, col)));
    }
    Matrix {
        data,
        rows: valid.len(),
        cols: matrix.cols,
    }
}

fn real_pair(matrix: &Matrix, lhs: usize, rhs: usize, pairwise: bool) -> (f64, usize) {
    let mut xs = Vec::with_capacity(matrix.rows);
    let mut ys = Vec::with_capacity(matrix.rows);
    for row in 0..matrix.rows {
        let x = matrix.get(row, lhs);
        let y = matrix.get(row, rhs);
        if !x.is_nan() && !y.is_nan() {
            xs.push(x);
            ys.push(y);
        } else if !pairwise {
            return (f64::NAN, matrix.rows);
        }
    }
    (real_correlation(&xs, &ys), xs.len())
}

fn real_correlation(xs: &[f64], ys: &[f64]) -> f64 {
    let count = xs.len().min(ys.len());
    if count < 2 {
        return f64::NAN;
    }
    let mean_x = xs.iter().take(count).sum::<f64>() / count as f64;
    let mean_y = ys.iter().take(count).sum::<f64>() / count as f64;
    let mut sum_xx = 0.0;
    let mut sum_yy = 0.0;
    let mut sum_xy = 0.0;
    for index in 0..count {
        let dx = xs[index] - mean_x;
        let dy = ys[index] - mean_y;
        sum_xx += dx * dx;
        sum_yy += dy * dy;
        sum_xy += dx * dy;
    }
    if sum_xx <= 0.0 || sum_yy <= 0.0 {
        f64::NAN
    } else {
        clamp_correlation(sum_xy / (sum_xx.sqrt() * sum_yy.sqrt()))
    }
}
