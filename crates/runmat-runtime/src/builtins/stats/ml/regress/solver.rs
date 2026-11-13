use super::*;

pub(super) fn solve_coefficients(
    prepared: &PreparedData,
    total_cols: usize,
    active_columns: Vec<usize>,
    need_covariance: bool,
) -> BuiltinResult<CoefficientSolve> {
    let rows = prepared.design.nrows();
    let rank = active_columns.len();
    if rank == 0 {
        return Ok(CoefficientSolve {
            beta: vec![0.0; total_cols],
            fitted: DVector::zeros(rows),
            inv_xtx_active: need_covariance.then(|| DMatrix::zeros(0, 0)),
            active_columns,
            rank,
        });
    }
    let active_design = active_design(&prepared.design, &active_columns);
    let svd = active_design.clone().svd(true, true);
    let u = svd
        .u
        .ok_or_else(|| numerical("regress: SVD did not return left singular vectors"))?;
    let v_t = svd
        .v_t
        .ok_or_else(|| numerical("regress: SVD did not return right singular vectors"))?;
    let singular_values = svd.singular_values.iter().copied().collect::<Vec<_>>();
    let largest = singular_values.iter().copied().fold(0.0_f64, f64::max);
    let tolerance = (rows.max(rank) as f64) * f64::EPSILON * largest.max(1.0);
    let mut beta_active = DVector::zeros(rank);
    let mut inv_xtx_active = need_covariance.then(|| DMatrix::<f64>::zeros(rank, rank));
    for (index, singular_value) in singular_values.iter().copied().enumerate() {
        if singular_value.abs() <= tolerance {
            continue;
        }
        let uty = u.column(index).dot(&prepared.y);
        let contribution = uty / singular_value;
        for row in 0..rank {
            beta_active[row] += v_t[(index, row)] * contribution;
        }
        if let Some(inv_xtx) = inv_xtx_active.as_mut() {
            let inv_s2 = 1.0 / (singular_value * singular_value);
            for row in 0..rank {
                for col in 0..rank {
                    inv_xtx[(row, col)] += v_t[(index, row)] * v_t[(index, col)] * inv_s2;
                }
            }
        }
    }
    let fitted = &active_design * &beta_active;
    let mut beta = vec![0.0; total_cols];
    for (idx, col) in active_columns.iter().copied().enumerate() {
        beta[col] = beta_active[idx];
    }
    Ok(CoefficientSolve {
        beta,
        fitted,
        inv_xtx_active,
        active_columns,
        rank,
    })
}
