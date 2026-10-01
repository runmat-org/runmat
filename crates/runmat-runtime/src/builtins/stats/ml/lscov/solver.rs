use super::*;

#[derive(Debug)]
pub(super) struct RealSolve {
    pub(super) x: DMatrix<f64>,
    pub(super) covariance_base: DMatrix<f64>,
    pub(super) rank: usize,
}

#[derive(Debug)]
pub(super) struct ComplexSolve {
    pub(super) x: DMatrix<Complex64>,
    pub(super) covariance_base: DMatrix<Complex64>,
    pub(super) rank: usize,
}

pub(super) fn solve_real_least_squares(
    a: &DMatrix<f64>,
    b: &DMatrix<f64>,
    original_rows: usize,
    original_cols: usize,
) -> BuiltinResult<RealSolve> {
    let cols = a.ncols();
    let rhs_cols = b.ncols();
    if a.nrows() == 0 {
        return Ok(RealSolve {
            x: DMatrix::zeros(cols, rhs_cols),
            covariance_base: DMatrix::zeros(cols, cols),
            rank: 0,
        });
    }
    let svd = a.clone().svd(true, true);
    let u = svd
        .u
        .ok_or_else(|| numerical("lscov: SVD did not return left singular vectors"))?;
    let v_t = svd
        .v_t
        .ok_or_else(|| numerical("lscov: SVD did not return right singular vectors"))?;
    let singular_values = svd.singular_values.as_slice().to_vec();
    let largest = singular_values.iter().copied().fold(0.0_f64, f64::max);
    let tolerance = (original_rows.max(original_cols) as f64) * f64::EPSILON * largest.max(1.0);
    let mut x = DMatrix::<f64>::zeros(cols, rhs_cols);
    let mut covariance_base = DMatrix::<f64>::zeros(cols, cols);
    let mut rank = 0usize;
    for (idx, singular_value) in singular_values.iter().copied().enumerate() {
        if singular_value.abs() <= tolerance {
            continue;
        }
        rank += 1;
        for rhs_col in 0..rhs_cols {
            let projection = u.column(idx).dot(&b.column(rhs_col)) / singular_value;
            for row in 0..cols {
                x[(row, rhs_col)] += v_t[(idx, row)] * projection;
            }
        }
        let inv_s2 = 1.0 / (singular_value * singular_value);
        for row in 0..cols {
            for col in 0..cols {
                covariance_base[(row, col)] += v_t[(idx, row)] * v_t[(idx, col)] * inv_s2;
            }
        }
    }
    Ok(RealSolve {
        x,
        covariance_base,
        rank,
    })
}

pub(super) fn solve_complex_least_squares(
    a: &DMatrix<Complex64>,
    b: &DMatrix<Complex64>,
    original_rows: usize,
    original_cols: usize,
) -> BuiltinResult<ComplexSolve> {
    let cols = a.ncols();
    let rhs_cols = b.ncols();
    if a.nrows() == 0 {
        return Ok(ComplexSolve {
            x: DMatrix::zeros(cols, rhs_cols),
            covariance_base: DMatrix::zeros(cols, cols),
            rank: 0,
        });
    }
    let svd = a.clone().svd(true, true);
    let u = svd
        .u
        .ok_or_else(|| numerical("lscov: SVD did not return left singular vectors"))?;
    let v_t = svd
        .v_t
        .ok_or_else(|| numerical("lscov: SVD did not return right singular vectors"))?;
    let singular_values = svd.singular_values.as_slice().to_vec();
    let largest = singular_values.iter().copied().fold(0.0_f64, f64::max);
    let tolerance = (original_rows.max(original_cols) as f64) * f64::EPSILON * largest.max(1.0);
    let mut x = DMatrix::<Complex64>::zeros(cols, rhs_cols);
    let mut covariance_base = DMatrix::<Complex64>::zeros(cols, cols);
    let mut rank = 0usize;
    for (idx, singular_value) in singular_values.iter().copied().enumerate() {
        if singular_value.abs() <= tolerance {
            continue;
        }
        rank += 1;
        for rhs_col in 0..rhs_cols {
            let projection = u.column(idx).dotc(&b.column(rhs_col)) / singular_value;
            for row in 0..cols {
                x[(row, rhs_col)] += v_t[(idx, row)].conj() * projection;
            }
        }
        let inv_s2 = 1.0 / (singular_value * singular_value);
        for row in 0..cols {
            for col in 0..cols {
                covariance_base[(row, col)] += v_t[(idx, row)].conj() * v_t[(idx, col)] * inv_s2;
            }
        }
    }
    Ok(ComplexSolve {
        x,
        covariance_base,
        rank,
    })
}
