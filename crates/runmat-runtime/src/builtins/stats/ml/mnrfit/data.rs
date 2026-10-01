use super::*;
use crate::builtins::common::tensor;
use runmat_value::{CellArray, Tensor};
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

pub(super) fn prepare_data(
    x: &Tensor,
    response: ResponseOutcomes,
    options: &Options,
    check: &impl Fn() -> BuiltinResult<()>,
) -> BuiltinResult<PreparedData> {
    check()?;
    let rows = x.rows();
    let values = predictor_values(x, check)?;
    let mut design = design_matrix(x, &values, options.constant, check)?;
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
        if invalid_weights(option_weights, check)? {
            return Err(invalid(
                "mnrfit: weights must be nonnegative finite values with positive total",
            ));
        }
        for (index, (weight, option_weight)) in
            weights.iter_mut().zip(option_weights.iter()).enumerate()
        {
            if index.is_multiple_of(4096) {
                check()?;
            }
            *weight *= *option_weight;
            if !weight.is_finite() {
                return Err(invalid(
                    "mnrfit: weighted observation count exceeds finite double range",
                ));
            }
        }
    }
    let kept_rows = compact_complete_cases(
        x,
        &values,
        cols,
        &mut design,
        &mut outcomes,
        &mut weights,
        check,
    )?;
    if kept_rows == 0 {
        return Err(invalid("mnrfit: no complete observations remain"));
    }
    if invalid_weights(&weights, check)? {
        return Err(invalid(
            "mnrfit: weights must be nonnegative finite values with positive total",
        ));
    }
    let observations = finite_total(weights.iter().copied(), "weights", check)?;
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

fn invalid_weights(weights: &[f64], check: &impl Fn() -> BuiltinResult<()>) -> BuiltinResult<bool> {
    for (index, weight) in weights.iter().enumerate() {
        if index.is_multiple_of(4096) {
            check()?;
        }
        if !weight.is_finite() || *weight < 0.0 {
            return Ok(true);
        }
    }
    Ok(false)
}

fn predictor_values<'a>(
    x: &'a Tensor,
    check: &impl Fn() -> BuiltinResult<()>,
) -> BuiltinResult<Cow<'a, [f64]>> {
    check()?;
    if let Some(values) = x.as_f64_slice() {
        return Ok(Cow::Borrowed(values));
    }
    let mut values = reserved(x.len(), "predictor conversion")?;
    for index in 0..x.len() {
        if index.is_multiple_of(16_384) {
            check()?;
        }
        values.push(tensor::tensor_value_f64(x, index));
    }
    Ok(Cow::Owned(values))
}

fn finite_total(
    values: impl Iterator<Item = f64>,
    role: &str,
    check: &impl Fn() -> BuiltinResult<()>,
) -> BuiltinResult<f64> {
    let mut total = 0.0;
    for (index, value) in values.enumerate() {
        if index.is_multiple_of(4096) {
            check()?;
        }
        total = checked_finite_add(total, value, role)?;
    }
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

fn design_matrix(
    x: &Tensor,
    values: &[f64],
    constant: bool,
    check: &impl Fn() -> BuiltinResult<()>,
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
    let mut design = filled(len, 0.0, "design matrix")?;
    let mut out_col = 0usize;
    if constant {
        for row in 0..rows {
            if row.is_multiple_of(16_384) {
                check()?;
            }
            design[row] = 1.0;
        }
        out_col = 1;
    }
    for col in 0..x_cols {
        let source = &values[col * rows..(col + 1) * rows];
        let destination = &mut design[(out_col + col) * rows..(out_col + col + 1) * rows];
        for (to, from) in destination.chunks_mut(16_384).zip(source.chunks(16_384)) {
            check()?;
            to.copy_from_slice(from);
        }
    }
    Ok(design)
}

fn compact_complete_cases(
    x: &Tensor,
    values: &[f64],
    cols: usize,
    design: &mut Vec<f64>,
    outcomes: &mut [Vec<f64>],
    weights: &mut Vec<f64>,
    check: &impl Fn() -> BuiltinResult<()>,
) -> BuiltinResult<usize> {
    check()?;
    let rows = x.rows();
    let mut keep = Vec::new();
    keep.try_reserve_exact(rows)
        .map_err(|_| invalid("mnrfit: row selection exceeds available host resources"))?;
    for row in 0..rows {
        if row.is_multiple_of(4096) {
            check()?;
        }
        let x_complete = (0..x.cols()).all(|col| values[row + col * rows].is_finite());
        let y_complete = weights[row].is_finite()
            && weights[row] > 0.0
            && outcomes.iter().all(|outcome| outcome[row].is_finite());
        if x_complete && y_complete {
            keep.push(row);
        }
    }
    if keep.len() == rows {
        return Ok(rows);
    }
    let new_len = checked_elements(keep.len(), cols, "compacted design matrix")?;
    let mut new_design = filled(new_len, 0.0, "compacted design matrix")?;
    for col in 0..cols {
        for (new_row, old_row) in keep.iter().enumerate() {
            if new_row.is_multiple_of(4096) {
                check()?;
            }
            new_design[new_row + col * keep.len()] = design[old_row + col * rows];
        }
    }
    for outcome in outcomes.iter_mut() {
        check()?;
        let mut compacted = filled(keep.len(), 0.0, "compacted outcomes")?;
        for (new_row, old_row) in keep.iter().enumerate() {
            if new_row.is_multiple_of(4096) {
                check()?;
            }
            compacted[new_row] = outcome[*old_row];
        }
        *outcome = compacted;
    }
    let mut compacted_weights = filled(keep.len(), 0.0, "compacted weights")?;
    for (new_row, old_row) in keep.iter().enumerate() {
        if new_row.is_multiple_of(4096) {
            check()?;
        }
        compacted_weights[new_row] = weights[*old_row];
    }
    *weights = compacted_weights;
    *design = new_design;
    Ok(keep.len())
}

pub(super) fn response_outcomes(value: Value, rows: usize) -> BuiltinResult<ResponseOutcomes> {
    match value {
        Value::Tensor(tensor) if tensor.cols() >= 2 && tensor.rows() == rows => {
            ensure_exact_integer_tensor(&tensor, "Y")?;
            count_outcomes(&tensor)
        }
        Value::Tensor(tensor) => {
            ensure_exact_integer_tensor(&tensor, "Y")?;
            let shape = tensor.shape.clone();
            numeric_outcomes(tensor::tensor_values_f64(&tensor), shape, rows)
        }
        Value::LogicalArray(logical) => categorical_outcomes(
            logical.data.iter().map(|value| f64::from(*value != 0)).collect(),
            logical.shape,
            rows,
        ),
        Value::StringArray(array) => text_outcomes(array.data, array.shape, rows),
        Value::String(text) if rows == 1 => text_outcomes(vec![text], vec![1, 1], rows),
        Value::CharArray(array) => {
            let shape = vec![array.rows, 1];
            text_outcomes(char_rows(&array), shape, rows)
        }
        Value::Cell(cell) => cellstr_outcomes(&cell, rows),
        Value::Object(object) if object.is_class(runmat_types::standard::CATEGORICAL) => {
            let labels = crate::builtins::table::categorical_labels(&Value::Object(object))?;
            text_outcomes(labels, vec![rows, 1], rows)
        }
        Value::Num(value) if rows == 1 => numeric_outcomes(vec![value], vec![1, 1], rows),
        Value::Bool(value) if rows == 1 => {
            categorical_outcomes(vec![f64::from(value)], vec![1, 1], rows)
        }
        other => Err(invalid(format!(
            "mnrfit: Y must be a numeric, logical, text, categorical, cellstr, or grouped counts response, got {other:?}"
        ))),
    }
}

fn count_outcomes(tensor: &Tensor) -> BuiltinResult<ResponseOutcomes> {
    let rows = tensor.rows();
    let cols = tensor.cols();
    let data = tensor::tensor_values_f64(tensor);
    let logits = cols - 1;
    let mut outcomes = columns(logits, rows, "grouped outcomes")?;
    let mut weights = filled(rows, 0.0, "grouped weights")?;
    let mut observations = 0.0;
    for row in 0..rows {
        if (0..cols).any(|col| data[row + col * rows].is_nan()) {
            continue;
        }
        let mut total = 0.0;
        for col in 0..cols {
            let count = data[row + col * rows];
            if !count.is_finite() || count < 0.0 {
                return Err(invalid(
                    "mnrfit: grouped counts must be nonnegative finite values",
                ));
            }
            total = checked_finite_add(total, count, "grouped count")?;
        }
        if total > 0.0 {
            for (col, outcome) in outcomes.iter_mut().enumerate().take(logits) {
                outcome[row] = data[row + col * rows] / total;
            }
            weights[row] = total;
            observations = checked_finite_add(observations, total, "grouped count")?;
        }
    }
    if observations <= 0.0 {
        return Err(invalid("mnrfit: grouped counts must contain observations"));
    }
    let mut classes = filled(cols, 0.0, "grouped class labels")?;
    for (index, class) in classes.iter_mut().enumerate() {
        *class = (index + 1) as f64;
    }
    Ok(ResponseOutcomes {
        outcomes,
        classes: ClassNames::Numeric(classes),
        weights,
    })
}

fn numeric_outcomes(
    values: Vec<f64>,
    shape: Vec<usize>,
    rows: usize,
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
    classes.extend(values.iter().copied().filter(|value| value.is_finite()));
    classes.sort_by(f64::total_cmp);
    classes.dedup_by(|a, b| *a == *b);
    if classes.len() < 2 {
        return Err(invalid("mnrfit: Y must contain at least two classes"));
    }
    let logits = classes.len() - 1;
    let mut outcomes = columns(logits, rows, "categorical outcomes")?;
    let mut weights = filled(rows, 1.0, "categorical weights")?;
    for (row, value) in values.iter().enumerate() {
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

fn categorical_outcomes(
    values: Vec<f64>,
    shape: Vec<usize>,
    rows: usize,
) -> BuiltinResult<ResponseOutcomes> {
    numeric_outcomes(values, shape, rows)
}

fn text_outcomes(
    values: Vec<String>,
    shape: Vec<usize>,
    rows: usize,
) -> BuiltinResult<ResponseOutcomes> {
    if values.len() != rows || shape.iter().copied().filter(|dim| *dim > 1).count() > 1 {
        return Err(invalid(
            "mnrfit: categorical Y must be a vector with one response per row of X",
        ));
    }
    if values.iter().any(|value| value.is_empty()) {
        return Err(invalid("mnrfit: text response classes must be nonempty"));
    }
    let mut classes = values.clone();
    classes.sort();
    classes.dedup();
    if classes.len() < 2 {
        return Err(invalid("mnrfit: Y must contain at least two classes"));
    }
    let logits = classes.len() - 1;
    let mut outcomes = columns(logits, rows, "text outcomes")?;
    for (row, value) in values.iter().enumerate() {
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
        weights: filled(rows, 1.0, "text weights")?,
    })
}

fn cellstr_outcomes(cell: &CellArray, rows: usize) -> BuiltinResult<ResponseOutcomes> {
    let labels = cell
        .data
        .iter()
        .map(|value| {
            scalar_text(value).ok_or_else(|| invalid("mnrfit: Y cell entries must be text"))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    text_outcomes(labels, cell.shape.clone(), rows)
}

#[cfg(test)]
#[path = "data/tests.rs"]
mod tests;
