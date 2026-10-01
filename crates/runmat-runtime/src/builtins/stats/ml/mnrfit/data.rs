use super::*;

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

type ResponseOutcomes = (Vec<Vec<f64>>, ClassNames, Vec<f64>, f64);

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

pub(super) fn prepare_data(x: &Tensor, y: Value, options: &Options) -> BuiltinResult<PreparedData> {
    let rows = x.rows();
    let mut design = design_matrix(x, options.constant)?;
    let cols = if options.constant {
        x.cols()
            .checked_add(1)
            .ok_or_else(|| internal("mnrfit: design matrix dimension overflow"))?
    } else {
        x.cols()
    };
    let (mut outcomes, classes, mut weights, _) = response_outcomes(y, rows)?;
    if let Some(option_weights) = &options.weights {
        if option_weights.len() != rows {
            return Err(invalid(
                "mnrfit: Weights vector length must match the number of rows in X",
            ));
        }
        for (weight, option_weight) in weights.iter_mut().zip(option_weights.iter()) {
            *weight *= *option_weight;
        }
    }
    let kept_rows = compact_complete_cases(x, cols, &mut design, &mut outcomes, &mut weights)?;
    if kept_rows == 0 {
        return Err(invalid("mnrfit: no complete observations remain"));
    }
    if weights
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
        || weights.iter().sum::<f64>() <= 0.0
    {
        return Err(invalid(
            "mnrfit: weights must be nonnegative finite values with positive total",
        ));
    }
    let observations = weights.iter().sum::<f64>();
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

fn design_matrix(x: &Tensor, constant: bool) -> BuiltinResult<Vec<f64>> {
    let rows = x.rows();
    let x_cols = x.cols();
    let cols = if constant {
        x_cols
            .checked_add(1)
            .ok_or_else(|| internal("mnrfit: design matrix dimension overflow"))?
    } else {
        x_cols
    };
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| internal("mnrfit: design matrix dimension overflow"))?;
    let mut design = vec![0.0; len];
    let mut out_col = 0usize;
    if constant {
        for row in 0..rows {
            design[row] = 1.0;
        }
        out_col = 1;
    }
    let values = tensor::tensor_values_f64_cow(x);
    for col in 0..x_cols {
        for row in 0..rows {
            design[row + (out_col + col) * rows] = values[row + col * rows];
        }
    }
    Ok(design)
}

fn compact_complete_cases(
    x: &Tensor,
    cols: usize,
    design: &mut Vec<f64>,
    outcomes: &mut [Vec<f64>],
    weights: &mut Vec<f64>,
) -> BuiltinResult<usize> {
    let rows = x.rows();
    let values = tensor::tensor_values_f64_cow(x);
    let keep = (0..rows)
        .filter(|row| {
            let x_complete = (0..x.cols()).all(|col| values[*row + col * rows].is_finite());
            let y_complete = weights[*row].is_finite()
                && weights[*row] > 0.0
                && outcomes.iter().all(|outcome| outcome[*row].is_finite());
            x_complete && y_complete
        })
        .collect::<Vec<_>>();
    if keep.len() == rows {
        return Ok(rows);
    }
    let new_len = keep
        .len()
        .checked_mul(cols)
        .ok_or_else(|| internal("mnrfit: design matrix dimension overflow"))?;
    let mut new_design = vec![0.0; new_len];
    for col in 0..cols {
        for (new_row, old_row) in keep.iter().enumerate() {
            new_design[new_row + col * keep.len()] = design[old_row + col * rows];
        }
    }
    for outcome in outcomes.iter_mut() {
        *outcome = keep.iter().map(|old_row| outcome[*old_row]).collect();
    }
    *weights = keep.iter().map(|old_row| weights[*old_row]).collect();
    *design = new_design;
    Ok(keep.len())
}

fn response_outcomes(value: Value, rows: usize) -> BuiltinResult<ResponseOutcomes> {
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
    let mut outcomes = vec![vec![0.0; rows]; logits];
    let mut weights = Vec::with_capacity(rows);
    let mut observations = 0.0;
    for row in 0..rows {
        let mut total = 0.0;
        let mut missing = false;
        for col in 0..cols {
            let count = data[row + col * rows];
            if count.is_nan() {
                missing = true;
                break;
            }
            if !count.is_finite() || count < 0.0 {
                return Err(invalid(
                    "mnrfit: grouped counts must be nonnegative finite values",
                ));
            }
            total += count;
        }
        if missing || total <= 0.0 {
            weights.push(0.0);
        } else {
            for (col, outcome) in outcomes.iter_mut().enumerate().take(logits) {
                outcome[row] = data[row + col * rows] / total;
            }
            weights.push(total);
            observations += total;
        }
    }
    if observations <= 0.0 {
        return Err(invalid("mnrfit: grouped counts must contain observations"));
    }
    let classes = (1..=cols).map(|idx| idx as f64).collect::<Vec<_>>();
    Ok((
        outcomes,
        ClassNames::Numeric(classes),
        weights,
        observations,
    ))
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
    let mut classes = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    classes.dedup_by(|a, b| (*a - *b).abs() <= EPS);
    if classes.len() < 2 {
        return Err(invalid("mnrfit: Y must contain at least two classes"));
    }
    let logits = classes.len() - 1;
    let mut outcomes = vec![vec![0.0; rows]; logits];
    let mut weights = vec![1.0; rows];
    for (row, value) in values.iter().enumerate() {
        if value.is_nan() {
            weights[row] = 0.0;
            continue;
        }
        if !value.is_finite() {
            return Err(invalid("mnrfit: Y classes must be finite or NaN"));
        }
        let class_idx = classes
            .iter()
            .position(|class| (*class - *value).abs() <= EPS)
            .ok_or_else(|| internal("mnrfit: failed to map response class"))?;
        if class_idx < logits {
            outcomes[class_idx][row] = 1.0;
        }
    }
    let observations = weights.iter().sum::<f64>();
    Ok((
        outcomes,
        ClassNames::Numeric(classes),
        weights,
        observations,
    ))
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
    let mut outcomes = vec![vec![0.0; rows]; logits];
    for (row, value) in values.iter().enumerate() {
        let class_idx = classes
            .iter()
            .position(|class| class == value)
            .ok_or_else(|| internal("mnrfit: failed to map response class"))?;
        if class_idx < logits {
            outcomes[class_idx][row] = 1.0;
        }
    }
    Ok((
        outcomes,
        ClassNames::Text(classes),
        vec![1.0; rows],
        rows as f64,
    ))
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
