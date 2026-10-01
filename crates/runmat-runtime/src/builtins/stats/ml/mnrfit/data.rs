use super::*;
use crate::builtins::common::tensor;
use runmat_value::{CellArray, CharArray, Tensor};
use std::borrow::Cow;

#[derive(Clone, Debug)]
pub(super) struct PreparedData {
    pub(super) design: Vec<f64>,
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) outcomes: Vec<Vec<f64>>,
    pub(super) weights: Vec<f64>,
    pub(super) classes: ClassNames,
    pub(super) observations: f64,
}

pub(super) struct ResponseOutcomes {
    outcomes: Vec<Vec<f64>>,
    classes: ClassNames,
    weights: Vec<f64>,
}

pub(super) fn validate_option_weight_length(count: usize, rows: usize) -> BuiltinResult<()> {
    if count != rows {
        return Err(invalid(
            "mnrfit: Weights vector length must match the number of rows in X",
        ));
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub(super) enum ClassNames {
    Numeric(Vec<f64>),
    Text(Vec<String>),
}

impl ClassNames {
    pub(super) fn len(&self) -> usize {
        match self {
            ClassNames::Numeric(values) => values.len(),
            ClassNames::Text(values) => values.len(),
        }
    }
}

pub(super) async fn prepare_data(
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
        x,
        &values,
        cols,
        &mut design,
        &mut outcomes,
        &mut weights,
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

fn checked_finite_add(total: f64, value: f64, role: &str) -> BuiltinResult<f64> {
    let next = total + value;
    if next.is_finite() {
        Ok(next)
    } else {
        Err(invalid(format!(
            "mnrfit: {role} total exceeds finite double range"
        )))
    }
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

async fn compact_complete_cases(
    x: &Tensor,
    values: &[f64],
    cols: usize,
    design: &mut Vec<f64>,
    outcomes: &mut [Vec<f64>],
    weights: &mut Vec<f64>,
    check: &impl Fn() -> BuiltinResult<()>,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<usize> {
    check()?;
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
        cooperation.completed(1).await?;
    }
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

pub(super) async fn response_outcomes(
    value: Value,
    rows: usize,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<ResponseOutcomes> {
    match value {
        Value::Tensor(tensor) if tensor.cols() >= 2 && tensor.rows() == rows => {
            ensure_exact_integer_tensor(&tensor, "Y")?;
            count_outcomes(&tensor, cooperation).await
        }
        Value::Tensor(tensor) => {
            ensure_exact_integer_tensor(&tensor, "Y")?;
            let shape = tensor.shape.clone();
            let mut values = reserved(tensor.len(), "numeric responses")?;
            for index in 0..tensor.len() {
                values.push(tensor::tensor_value_f64(&tensor, index));
                if (index + 1).is_multiple_of(16_384) {
                    cooperation.completed(16_384).await?;
                }
            }
            cooperation.completed((tensor.len() % 16_384) as u64).await?;
            numeric_outcomes(values, shape, rows, cooperation).await
        }
        Value::LogicalArray(logical) => {
            let mut values = reserved(logical.data.len(), "logical responses")?;
            for (index, value) in logical.data.iter().enumerate() {
                values.push(f64::from(*value != 0));
                if (index + 1).is_multiple_of(16_384) {
                    cooperation.completed(16_384).await?;
                }
            }
            cooperation.completed((logical.data.len() % 16_384) as u64).await?;
            numeric_outcomes(values, logical.shape, rows, cooperation).await
        }
        Value::StringArray(array) => text_outcomes(array.data, array.shape, rows, cooperation).await,
        Value::String(text) if rows == 1 => text_outcomes(vec![text], vec![1, 1], rows, cooperation).await,
        Value::CharArray(array) => {
            let shape = vec![array.rows, 1];
            text_outcomes(char_rows_cooperatively(&array, cooperation).await?, shape, rows, cooperation).await
        }
        Value::Cell(cell) => cellstr_outcomes(&cell, rows, cooperation).await,
        Value::Object(object) if object.is_class(runmat_types::standard::CATEGORICAL) => {
            let labels = crate::builtins::table::categorical_labels(&Value::Object(object))?;
            text_outcomes(labels, vec![rows, 1], rows, cooperation).await
        }
        Value::Num(value) if rows == 1 => numeric_outcomes(vec![value], vec![1, 1], rows, cooperation).await,
        Value::Bool(value) if rows == 1 => {
            numeric_outcomes(vec![f64::from(value)], vec![1, 1], rows, cooperation).await
        }
        other => Err(invalid(format!(
            "mnrfit: Y must be a numeric, logical, text, categorical, cellstr, or grouped counts response, got {other:?}"
        ))),
    }
}

async fn count_outcomes(
    tensor: &Tensor,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<ResponseOutcomes> {
    let rows = tensor.rows();
    let cols = tensor.cols();
    let logits = cols - 1;
    let mut outcomes = columns(logits, rows, "grouped outcomes", &|| Ok(()), cooperation).await?;
    let mut weights =
        filled_cooperatively(rows, 0.0, "grouped weights", &|| Ok(()), cooperation).await?;
    let mut observations = 0.0;
    for row in 0..rows {
        let mut missing = false;
        for col in 0..cols {
            missing |= tensor::tensor_value_f64(tensor, row + col * rows).is_nan();
            if (col + 1).is_multiple_of(16_384) {
                cooperation.completed(16_384).await?;
            }
        }
        cooperation.completed((cols % 16_384) as u64).await?;
        if missing {
            continue;
        }
        let mut total = 0.0;
        for col in 0..cols {
            let count = tensor::tensor_value_f64(tensor, row + col * rows);
            if !count.is_finite() || count < 0.0 {
                return Err(invalid(
                    "mnrfit: grouped counts must be nonnegative finite values",
                ));
            }
            total = checked_finite_add(total, count, "grouped count")?;
            if (col + 1).is_multiple_of(16_384) {
                cooperation.completed(16_384).await?;
            }
        }
        cooperation.completed((cols % 16_384) as u64).await?;
        if total > 0.0 {
            for (col, outcome) in outcomes.iter_mut().enumerate().take(logits) {
                outcome[row] = tensor::tensor_value_f64(tensor, row + col * rows) / total;
                if (col + 1).is_multiple_of(16_384) {
                    cooperation.completed(16_384).await?;
                }
            }
            cooperation.completed((logits % 16_384) as u64).await?;
            weights[row] = total;
            observations = checked_finite_add(observations, total, "grouped count")?;
        }
    }
    if observations <= 0.0 {
        return Err(invalid("mnrfit: grouped counts must contain observations"));
    }
    let mut classes =
        filled_cooperatively(cols, 0.0, "grouped class labels", &|| Ok(()), cooperation).await?;
    for (index, class) in classes.iter_mut().enumerate() {
        *class = (index + 1) as f64;
    }
    Ok(ResponseOutcomes {
        outcomes,
        classes: ClassNames::Numeric(classes),
        weights,
    })
}

async fn numeric_outcomes(
    values: Vec<f64>,
    shape: Vec<usize>,
    rows: usize,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<ResponseOutcomes> {
    if values.len() != rows || shape.iter().copied().filter(|dim| *dim > 1).count() > 1 {
        return Err(invalid(
            "mnrfit: categorical Y must be a vector with one response per row of X",
        ));
    }
    let mut classes = Vec::new();
    classes
        .try_reserve_exact(values.len())
        .map_err(|_| invalid("mnrfit: class census exceeds available host resources"))?;
    for (index, value) in values.iter().copied().enumerate() {
        if value.is_finite() {
            classes.push(value);
        }
        if (index + 1).is_multiple_of(16_384) {
            cooperation.completed(16_384).await?;
        }
    }
    cooperation
        .completed((values.len() % 16_384) as u64)
        .await?;
    let classes = class_census::sort_unique(
        classes,
        f64::total_cmp,
        |left, right| left == right,
        "numeric class census",
        cooperation,
    )
    .await?;
    if classes.len() < 2 {
        return Err(invalid("mnrfit: Y must contain at least two classes"));
    }
    let logits = classes.len() - 1;
    let mut outcomes = columns(
        logits,
        rows,
        "categorical outcomes",
        &|| Ok(()),
        cooperation,
    )
    .await?;
    let mut weights =
        filled_cooperatively(rows, 1.0, "categorical weights", &|| Ok(()), cooperation).await?;
    for (row, value) in values.iter().enumerate() {
        cooperation.completed(1).await?;
        if value.is_nan() {
            weights[row] = 0.0;
            continue;
        }
        if !value.is_finite() {
            return Err(invalid("mnrfit: Y classes must be finite or NaN"));
        }
        let class_idx = classes
            .binary_search_by(|class| {
                class
                    .partial_cmp(value)
                    .expect("numeric response classes were filtered to finite values")
            })
            .map_err(|_| internal("mnrfit: failed to map response class"))?;
        if class_idx < logits {
            outcomes[class_idx][row] = 1.0;
        }
    }
    Ok(ResponseOutcomes {
        outcomes,
        classes: ClassNames::Numeric(classes),
        weights,
    })
}

async fn text_outcomes(
    values: Vec<String>,
    shape: Vec<usize>,
    rows: usize,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<ResponseOutcomes> {
    if values.len() != rows || shape.iter().copied().filter(|dim| *dim > 1).count() > 1 {
        return Err(invalid(
            "mnrfit: categorical Y must be a vector with one response per row of X",
        ));
    }
    for (index, value) in values.iter().enumerate() {
        if value.is_empty() {
            return Err(invalid("mnrfit: text response classes must be nonempty"));
        }
        if (index + 1).is_multiple_of(16_384) {
            cooperation.completed(16_384).await?;
        }
    }
    cooperation
        .completed((values.len() % 16_384) as u64)
        .await?;
    let mut classes = Vec::new();
    classes
        .try_reserve_exact(values.len())
        .map_err(|_| invalid("mnrfit: text class census exceeds available host resources"))?;
    for (index, value) in values.iter().enumerate() {
        classes.push(value.clone());
        if (index + 1).is_multiple_of(16_384) {
            cooperation.completed(16_384).await?;
        }
    }
    cooperation
        .completed((values.len() % 16_384) as u64)
        .await?;
    let classes = class_census::sort_unique(
        classes,
        String::cmp,
        |left, right| left == right,
        "text class census",
        cooperation,
    )
    .await?;
    if classes.len() < 2 {
        return Err(invalid("mnrfit: Y must contain at least two classes"));
    }
    let logits = classes.len() - 1;
    let mut outcomes = columns(logits, rows, "text outcomes", &|| Ok(()), cooperation).await?;
    for (row, value) in values.iter().enumerate() {
        cooperation.completed(1).await?;
        let class_idx = classes
            .binary_search(value)
            .map_err(|_| internal("mnrfit: failed to map response class"))?;
        if class_idx < logits {
            outcomes[class_idx][row] = 1.0;
        }
    }
    Ok(ResponseOutcomes {
        outcomes,
        classes: ClassNames::Text(classes),
        weights: filled_cooperatively(rows, 1.0, "text weights", &|| Ok(()), cooperation).await?,
    })
}

async fn cellstr_outcomes(
    cell: &CellArray,
    rows: usize,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<ResponseOutcomes> {
    let mut labels = Vec::new();
    labels
        .try_reserve_exact(cell.data.len())
        .map_err(|_| invalid("mnrfit: text response labels exceed available host resources"))?;
    for (index, value) in cell.data.iter().enumerate() {
        labels.push(
            scalar_text(value).ok_or_else(|| invalid("mnrfit: Y cell entries must be text"))?,
        );
        if (index + 1).is_multiple_of(16_384) {
            cooperation.completed(16_384).await?;
        }
    }
    cooperation
        .completed((cell.data.len() % 16_384) as u64)
        .await?;
    text_outcomes(labels, cell.shape.clone(), rows, cooperation).await
}

async fn char_rows_cooperatively(
    array: &CharArray,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<Vec<String>> {
    let mut rows = Vec::new();
    rows.try_reserve_exact(array.rows)
        .map_err(|_| invalid("mnrfit: text response rows exceed available host resources"))?;
    for row in 0..array.rows {
        let start = row * array.cols;
        let mut text = String::new();
        text.try_reserve(array.cols)
            .map_err(|_| invalid("mnrfit: text response row exceeds available host resources"))?;
        for (index, character) in array.data[start..start + array.cols].iter().enumerate() {
            text.push(*character);
            if (index + 1).is_multiple_of(16_384) {
                cooperation.completed(16_384).await?;
            }
        }
        cooperation.completed((array.cols % 16_384) as u64).await?;
        rows.push(text);
    }
    Ok(rows)
}

#[cfg(test)]
#[path = "data/tests.rs"]
mod tests;
