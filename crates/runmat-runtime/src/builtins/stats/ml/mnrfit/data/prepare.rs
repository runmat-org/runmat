use super::*;
use crate::builtins::common::tensor;
use runmat_value::Tensor;
use std::borrow::Cow;

pub(in super::super) async fn prepare_data(
    x: &Tensor,
    response: ResponseOutcomes,
    options: &Options,
    check: &impl Fn() -> BuiltinResult<()>,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<PreparedData> {
    check()?;
    let rows = x.rows();
    let values = predictor_values(x, check, cooperation).await?;
    let mut design = design_matrix(x, &values, options.constant, check, cooperation).await?;
    let cols = if options.constant {
        x.cols()
            .checked_add(1)
            .ok_or_else(|| internal("mnrfit: design matrix dimension overflow"))?
    } else {
        x.cols()
    };
    let ResponseOutcomes {
        mut outcomes,
        classes,
        mut weights,
    } = response;
    if let Some(option_weights) = &options.weights {
        validate_option_weight_length(option_weights.len(), rows)?;
        if invalid_weights(option_weights, check, cooperation).await? {
            return Err(invalid(
                "mnrfit: weights must be nonnegative finite values with positive total",
            ));
        }
        for (index, (weight, option_weight)) in
            weights.iter_mut().zip(option_weights.iter()).enumerate()
        {
            *weight *= *option_weight;
            if !weight.is_finite() {
                return Err(invalid(
                    "mnrfit: weighted observation count exceeds finite double range",
                ));
            }
            if (index + 1).is_multiple_of(4096) {
                check()?;
                cooperation.completed(4096).await?;
            }
        }
        cooperation.completed((weights.len() % 4096) as u64).await?;
    }
    let kept_rows = compact_complete_cases(
        CompleteCaseBuffers {
            x,
            values: &values,
            cols,
            design: &mut design,
            outcomes: &mut outcomes,
            weights: &mut weights,
        },
        check,
        cooperation,
    )
    .await?;
    if kept_rows == 0 {
        return Err(invalid("mnrfit: no complete observations remain"));
    }
    if invalid_weights(&weights, check, cooperation).await? {
        return Err(invalid(
            "mnrfit: weights must be nonnegative finite values with positive total",
        ));
    }
    let observations = finite_total(weights.iter().copied(), "weights", check, cooperation).await?;
    if observations <= 0.0 {
        return Err(invalid(
            "mnrfit: weights must be nonnegative finite values with positive total",
        ));
    }
    Ok(PreparedData {
        design,
        rows: kept_rows,
        cols,
        outcomes,
        weights,
        classes,
        observations,
    })
}

async fn invalid_weights(
    weights: &[f64],
    check: &impl Fn() -> BuiltinResult<()>,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<bool> {
    for (index, weight) in weights.iter().enumerate() {
        if !weight.is_finite() || *weight < 0.0 {
            return Ok(true);
        }
        if (index + 1).is_multiple_of(4096) {
            check()?;
            cooperation.completed(4096).await?;
        }
    }
    cooperation.completed((weights.len() % 4096) as u64).await?;
    Ok(false)
}

async fn predictor_values<'a>(
    x: &'a Tensor,
    check: &impl Fn() -> BuiltinResult<()>,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<Cow<'a, [f64]>> {
    check()?;
    if let Some(values) = x.as_f64_slice() {
        return Ok(Cow::Borrowed(values));
    }
    let mut values = reserved(x.len(), "predictor conversion")?;
    for index in 0..x.len() {
        values.push(tensor::tensor_value_f64(x, index));
        if (index + 1).is_multiple_of(16_384) {
            check()?;
            cooperation.completed(16_384).await?;
        }
    }
    cooperation.completed((x.len() % 16_384) as u64).await?;
    Ok(Cow::Owned(values))
}

async fn finite_total(
    values: impl Iterator<Item = f64>,
    role: &str,
    check: &impl Fn() -> BuiltinResult<()>,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<f64> {
    let mut total = 0.0;
    let mut count = 0usize;
    for (index, value) in values.enumerate() {
        total = checked_finite_add(total, value, role)?;
        count += 1;
        if (index + 1).is_multiple_of(4096) {
            check()?;
            cooperation.completed(4096).await?;
        }
    }
    cooperation.completed((count % 4096) as u64).await?;
    Ok(total)
}

async fn design_matrix(
    x: &Tensor,
    values: &[f64],
    constant: bool,
    check: &impl Fn() -> BuiltinResult<()>,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<Vec<f64>> {
    check()?;
    let rows = x.rows();
    let x_cols = x.cols();
    let cols = if constant {
        x_cols
            .checked_add(1)
            .ok_or_else(|| internal("mnrfit: design matrix dimension overflow"))?
    } else {
        x_cols
    };
    let len = checked_elements(rows, cols, "design matrix")?;
    let mut design = filled_cooperatively(len, 0.0, "design matrix", check, cooperation).await?;
    let mut out_col = 0usize;
    if constant {
        for row in 0..rows {
            design[row] = 1.0;
            if (row + 1).is_multiple_of(16_384) {
                check()?;
                cooperation.completed(16_384).await?;
            }
        }
        cooperation.completed((rows % 16_384) as u64).await?;
        out_col = 1;
    }
    for col in 0..x_cols {
        let source = &values[col * rows..(col + 1) * rows];
        let destination = &mut design[(out_col + col) * rows..(out_col + col + 1) * rows];
        for (to, from) in destination.chunks_mut(16_384).zip(source.chunks(16_384)) {
            check()?;
            to.copy_from_slice(from);
            cooperation.completed(to.len() as u64).await?;
        }
    }
    Ok(design)
}

pub(super) struct CompleteCaseBuffers<'a> {
    pub(super) x: &'a Tensor,
    pub(super) values: &'a [f64],
    pub(super) cols: usize,
    pub(super) design: &'a mut Vec<f64>,
    pub(super) outcomes: &'a mut [Vec<f64>],
    pub(super) weights: &'a mut Vec<f64>,
}

pub(super) async fn compact_complete_cases(
    buffers: CompleteCaseBuffers<'_>,
    check: &impl Fn() -> BuiltinResult<()>,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<usize> {
    check()?;
    let CompleteCaseBuffers {
        x,
        values,
        cols,
        design,
        outcomes,
        weights,
    } = buffers;
    let rows = x.rows();
    let mut keep = Vec::new();
    keep.try_reserve_exact(rows)
        .map_err(|_| invalid("mnrfit: row selection exceeds available host resources"))?;
    for row in 0..rows {
        let mut x_complete = true;
        for col in 0..x.cols() {
            x_complete &= values[row + col * rows].is_finite();
            if (col + 1).is_multiple_of(16_384) {
                cooperation.completed(16_384).await?;
            }
        }
        cooperation.completed((x.cols() % 16_384) as u64).await?;
        let mut y_complete = weights[row].is_finite() && weights[row] > 0.0;
        if y_complete {
            for (index, outcome) in outcomes.iter().enumerate() {
                y_complete &= outcome[row].is_finite();
                if (index + 1).is_multiple_of(16_384) {
                    cooperation.completed(16_384).await?;
                }
            }
            cooperation
                .completed((outcomes.len() % 16_384) as u64)
                .await?;
        }
        if x_complete && y_complete {
            keep.push(row);
        }
        if (row + 1).is_multiple_of(16_384) {
            cooperation.completed(16_384).await?;
        }
    }
    cooperation.completed((rows % 16_384) as u64).await?;
    if keep.len() == rows {
        return Ok(rows);
    }
    let new_len = checked_elements(keep.len(), cols, "compacted design matrix")?;
    let mut new_design =
        filled_cooperatively(new_len, 0.0, "compacted design matrix", check, cooperation).await?;
    for col in 0..cols {
        for (new_row, old_row) in keep.iter().enumerate() {
            new_design[new_row + col * keep.len()] = design[old_row + col * rows];
            if (new_row + 1).is_multiple_of(4096) {
                check()?;
                cooperation.completed(4096).await?;
            }
        }
        cooperation.completed((keep.len() % 4096) as u64).await?;
    }
    for outcome in outcomes.iter_mut() {
        check()?;
        let mut compacted =
            filled_cooperatively(keep.len(), 0.0, "compacted outcomes", check, cooperation).await?;
        for (new_row, old_row) in keep.iter().enumerate() {
            compacted[new_row] = outcome[*old_row];
            if (new_row + 1).is_multiple_of(4096) {
                check()?;
                cooperation.completed(4096).await?;
            }
        }
        cooperation.completed((keep.len() % 4096) as u64).await?;
        *outcome = compacted;
    }
    let mut compacted_weights =
        filled_cooperatively(keep.len(), 0.0, "compacted weights", check, cooperation).await?;
    for (new_row, old_row) in keep.iter().enumerate() {
        compacted_weights[new_row] = weights[*old_row];
        if (new_row + 1).is_multiple_of(4096) {
            check()?;
            cooperation.completed(4096).await?;
        }
    }
    cooperation.completed((keep.len() % 4096) as u64).await?;
    *weights = compacted_weights;
    *design = new_design;
    Ok(keep.len())
}
