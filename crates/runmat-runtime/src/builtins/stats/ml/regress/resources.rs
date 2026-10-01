use super::*;

pub(super) fn ensure_input_budget(rows: usize, cols: usize) -> BuiltinResult<()> {
    let input_cells = rows
        .checked_mul(cols)
        .ok_or_else(|| invalid("regress: input is too large"))?;
    if input_cells > MAX_REGRESS_CELLS {
        return Err(invalid("regress: input is too large"));
    }
    Ok(())
}

pub(super) fn ensure_output_budget(
    rows: usize,
    cols: usize,
    requested_outputs: usize,
) -> BuiltinResult<()> {
    let mut output_cells = cols;
    if requested_outputs >= 2 {
        output_cells = output_cells
            .checked_add(
                cols.checked_mul(2)
                    .ok_or_else(|| invalid("regress: output is too large"))?,
            )
            .ok_or_else(|| invalid("regress: output is too large"))?;
    }
    if requested_outputs >= 3 {
        output_cells = output_cells
            .checked_add(rows)
            .ok_or_else(|| invalid("regress: output is too large"))?;
    }
    if requested_outputs >= 4 {
        output_cells = output_cells
            .checked_add(
                rows.checked_mul(2)
                    .ok_or_else(|| invalid("regress: output is too large"))?,
            )
            .ok_or_else(|| invalid("regress: output is too large"))?;
    }
    if requested_outputs >= 5 {
        output_cells = output_cells
            .checked_add(4)
            .ok_or_else(|| invalid("regress: output is too large"))?;
    }
    if output_cells > MAX_REGRESS_CELLS {
        return Err(invalid("regress: output is too large"));
    }
    Ok(())
}

pub(super) fn ensure_diagnostic_budget(
    active_cols: usize,
    requested_outputs: usize,
) -> BuiltinResult<()> {
    if requested_outputs <= 1 {
        return Ok(());
    }
    let covariance_cells = active_cols
        .checked_mul(active_cols)
        .ok_or_else(|| invalid("regress: diagnostic work array is too large"))?;
    if covariance_cells > MAX_REGRESS_CELLS {
        return Err(invalid("regress: diagnostic work array is too large"));
    }
    Ok(())
}
