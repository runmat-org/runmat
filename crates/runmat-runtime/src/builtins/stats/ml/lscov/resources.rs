use super::*;

pub(super) fn validate_symmetric_covariance(covariance: &DMatrix<f64>) -> BuiltinResult<()> {
    for row in 0..covariance.nrows() {
        for col in (row + 1)..covariance.ncols() {
            let left = covariance[(row, col)];
            let right = covariance[(col, row)];
            let scale = left.abs().max(right.abs()).max(1.0);
            if (left - right).abs() > EPS * scale {
                return Err(invalid("lscov: V covariance matrix must be symmetric"));
            }
        }
    }
    Ok(())
}

pub(super) fn scaled_psd_tolerance(values: &[f64]) -> f64 {
    let scale = values
        .iter()
        .map(|value| value.abs())
        .fold(1.0_f64, f64::max);
    EPS * scale * values.len().max(1) as f64
}

pub(super) fn ensure_budget(rows: usize, cols: usize, label: &str) -> BuiltinResult<()> {
    let cells = rows
        .checked_mul(cols)
        .ok_or_else(|| invalid(format!("lscov: {label} is too large")))?;
    if cells > MAX_LSCOV_CELLS {
        return Err(invalid(format!("lscov: {label} is too large")));
    }
    Ok(())
}

pub(super) fn ensure_work_budget(cols: usize) -> BuiltinResult<()> {
    let cells = cols
        .checked_mul(cols)
        .ok_or_else(|| invalid("lscov: covariance work array is too large"))?;
    if cells > MAX_LSCOV_CELLS {
        return Err(invalid("lscov: covariance work array is too large"));
    }
    Ok(())
}

pub(super) fn ensure_output_budget(
    cols: usize,
    rhs_cols: usize,
    requested_outputs: usize,
) -> BuiltinResult<()> {
    let primary = cols
        .checked_mul(rhs_cols)
        .ok_or_else(|| invalid("lscov: output is too large"))?;
    let mut cells = primary;
    if requested_outputs >= 2 {
        cells = cells
            .checked_add(primary)
            .ok_or_else(|| invalid("lscov: output is too large"))?;
    }
    if requested_outputs >= 3 {
        cells = cells
            .checked_add(rhs_cols)
            .ok_or_else(|| invalid("lscov: output is too large"))?;
    }
    if requested_outputs >= 4 {
        cells = cells
            .checked_add(
                cols.checked_mul(cols)
                    .ok_or_else(|| invalid("lscov: output is too large"))?,
            )
            .ok_or_else(|| invalid("lscov: output is too large"))?;
    }
    if cells > MAX_LSCOV_CELLS {
        return Err(invalid("lscov: output is too large"));
    }
    Ok(())
}
