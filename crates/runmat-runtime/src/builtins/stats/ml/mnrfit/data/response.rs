use super::*;
use crate::builtins::common::tensor;
use runmat_value::{decode_utf16_code_units, CellArray, CharArray, Tensor, Utf16DecodeError};

#[derive(Clone, Copy)]
pub(in super::super) enum ResponseExactness {
    PrevalidatedHost,
    ValidateDownloaded,
}

pub(in super::super) async fn response_outcomes(
    value: Value,
    rows: usize,
    exactness: ResponseExactness,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<ResponseOutcomes> {
    match value {
        Value::Tensor(tensor) if tensor.cols() >= 2 && tensor.rows() == rows => {
            if matches!(exactness, ResponseExactness::ValidateDownloaded) {
                ensure_exact_integer_tensor(&tensor, "Y", cooperation).await?;
            }
            count_outcomes(&tensor, cooperation).await
        }
        Value::Tensor(tensor) => {
            if matches!(exactness, ResponseExactness::ValidateDownloaded) {
                ensure_exact_integer_tensor(&tensor, "Y", cooperation).await?;
            }
            let shape = tensor.shape().to_vec();
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
            numeric_outcomes(values, logical.shape.to_vec(), rows, cooperation).await
        }
        Value::StringArray(array) => {
            text_outcomes(array.data(), &array.shape, rows, cooperation).await
        }
        Value::String(text) if rows == 1 => {
            text_outcomes(std::slice::from_ref(&text), &[1, 1], rows, cooperation).await
        }
        Value::CharArray(array) => {
            let shape = vec![array.rows, 1];
            let labels = char_rows_cooperatively(&array, cooperation).await?;
            text_outcomes(&labels, &shape, rows, cooperation).await
        }
        Value::Cell(cell) => cellstr_outcomes(&cell, rows, cooperation).await,
        Value::Object(object) if object.is_class(runmat_types::standard::CATEGORICAL) => {
            let source = crate::builtins::table::categorical_label_view(&object)?;
            let mut labels = Vec::<String>::new();
            labels.try_reserve_exact(source.len()).map_err(|_| {
                invalid("mnrfit: categorical response labels exceed available host resources")
            })?;
            for index in 0..source.len() {
                let label = source.label_at(index)?;
                labels.push(class_census::clone_text(label, cooperation).await?);
                if (index + 1).is_multiple_of(16_384) {
                    cooperation.completed(16_384).await?;
                }
            }
            cooperation.completed((source.len() % 16_384) as u64).await?;
            text_outcomes(&labels, &[rows, 1], rows, cooperation).await
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

pub(super) async fn count_outcomes(
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

pub(super) async fn numeric_outcomes(
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
        if value.is_nan() {
            weights[row] = 0.0;
        } else if !value.is_finite() {
            return Err(invalid("mnrfit: Y classes must be finite or NaN"));
        } else {
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
        if (row + 1).is_multiple_of(16_384) {
            cooperation.completed(16_384).await?;
        }
    }
    cooperation.completed((rows % 16_384) as u64).await?;
    Ok(ResponseOutcomes {
        outcomes,
        classes: ClassNames::Numeric(classes),
        weights,
    })
}

trait ResponseText {
    fn response_text(&self) -> BuiltinResult<&str>;
}

impl ResponseText for runmat_value::StringElement {
    fn response_text(&self) -> BuiltinResult<&str> {
        self.as_str()
            .ok_or_else(|| invalid("mnrfit: text response classes must be nonmissing"))
    }
}

impl ResponseText for String {
    fn response_text(&self) -> BuiltinResult<&str> {
        Ok(self.as_str())
    }
}

async fn text_outcomes<T: ResponseText>(
    values: &[T],
    shape: &[usize],
    rows: usize,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<ResponseOutcomes> {
    if values.len() != rows || shape.iter().copied().filter(|dim| *dim > 1).count() > 1 {
        return Err(invalid(
            "mnrfit: categorical Y must be a vector with one response per row of X",
        ));
    }
    for (index, value) in values.iter().enumerate() {
        if value.response_text()?.is_empty() {
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
        classes.push(class_census::clone_text(value.response_text()?, cooperation).await?);
        if (index + 1).is_multiple_of(16_384) {
            cooperation.completed(16_384).await?;
        }
    }
    cooperation
        .completed((values.len() % 16_384) as u64)
        .await?;
    let classes = class_census::sort_unique_text(classes, cooperation).await?;
    if classes.len() < 2 {
        return Err(invalid("mnrfit: Y must contain at least two classes"));
    }
    let logits = classes.len() - 1;
    let mut outcomes = columns(logits, rows, "text outcomes", &|| Ok(()), cooperation).await?;
    for (row, value) in values.iter().enumerate() {
        let class_idx =
            class_census::find_text_class(&classes, value.response_text()?, cooperation)
                .await?
                .ok_or_else(|| internal("mnrfit: failed to map response class"))?;
        if class_idx < logits {
            outcomes[class_idx][row] = 1.0;
        }
        if (row + 1).is_multiple_of(16_384) {
            cooperation.completed(16_384).await?;
        }
    }
    cooperation.completed((rows % 16_384) as u64).await?;
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
        .try_reserve_exact(cell.data().len())
        .map_err(|_| invalid("mnrfit: text response labels exceed available host resources"))?;
    for (index, value) in cell.data().iter().enumerate() {
        let label = match value {
            Value::String(runmat_value::StringElement::Text(text)) => {
                class_census::clone_text(text, cooperation).await?
            }
            Value::CharArray(chars) if chars.rows == 1 => {
                char_rows_cooperatively(chars, cooperation)
                    .await?
                    .pop()
                    .ok_or_else(|| internal("mnrfit: missing character row"))?
            }
            _ => return Err(invalid("mnrfit: Y cell entries must be text")),
        };
        labels.push(label);
        if (index + 1).is_multiple_of(16_384) {
            cooperation.completed(16_384).await?;
        }
    }
    cooperation
        .completed((cell.data().len() % 16_384) as u64)
        .await?;
    text_outcomes(&labels, cell.shape(), rows, cooperation).await
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
        for (index, _) in array.data()[start..start + array.cols].iter().enumerate() {
            if (index + 1).is_multiple_of(16_384) {
                cooperation.completed(16_384).await?;
            }
        }
        cooperation.completed((array.cols % 16_384) as u64).await?;
        let text = decode_utf16_code_units(&array.data()[start..start + array.cols])
            .map_err(invalid_utf16_response)?;
        rows.push(text);
    }
    Ok(rows)
}

fn invalid_utf16_response(error: Utf16DecodeError) -> RuntimeError {
    let mut builder = build_runtime_error(format!(
        "mnrfit: response label is not valid UTF-16 text: {error}"
    ))
    .with_builtin(NAME)
    .with_source(error);
    if let Some(identifier) = ERROR_INVALID_ARGUMENT.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
