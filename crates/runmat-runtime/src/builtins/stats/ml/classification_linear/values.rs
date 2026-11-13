use super::*;
use runmat_value::decode_utf16_code_units;

pub(super) fn scalar_text(value: &Value) -> BuiltinResult<Option<String>> {
    decode_scalar_text(value).map_err(fitclinear_utf16_error)
}

pub(super) fn predict_scalar_text(value: &Value) -> BuiltinResult<Option<String>> {
    decode_scalar_text(value).map_err(predict_utf16_error)
}

fn decode_scalar_text(value: &Value) -> Result<Option<String>, runmat_value::Utf16DecodeError> {
    match value {
        Value::String(text) => Ok(text.as_str().map(str::to_owned)),
        Value::CharArray(array) if array.rows == 1 => {
            decode_utf16_code_units(array.data()).map(Some)
        }
        Value::StringArray(array) if array.data().len() == 1 => {
            Ok(array.data()[0].as_str().map(str::to_owned))
        }
        _ => Ok(None),
    }
}

pub(super) fn text_list(value: &Value, name: &str) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => {
            Ok(vec![text.as_str().to_owned()])
        }
        Value::CharArray(array) => fit_char_rows(array),
        Value::StringArray(array) => array
            .data()
            .iter()
            .map(|text| {
                text.as_str().map(str::to_owned).ok_or_else(|| {
                    fitclinear_invalid("fitclinear: text entries must be nonmissing")
                })
            })
            .collect(),
        Value::Cell(cell) => cell
            .data()
            .iter()
            .map(|value| {
                scalar_text(value)?.ok_or_else(|| {
                    fitclinear_invalid(format!("fitclinear: {name} entries must be text"))
                })
            })
            .collect(),
        _ => Err(fitclinear_invalid(format!(
            "fitclinear: {name} must be text or a text collection"
        ))),
    }
}

pub(super) fn char_rows(array: &CharArray) -> BuiltinResult<Vec<String>> {
    decode_char_rows(array).map_err(predict_utf16_error)
}

fn fit_char_rows(array: &CharArray) -> BuiltinResult<Vec<String>> {
    decode_char_rows(array).map_err(fitclinear_utf16_error)
}

fn decode_char_rows(array: &CharArray) -> Result<Vec<String>, runmat_value::Utf16DecodeError> {
    let mut out = Vec::with_capacity(array.rows);
    for row in 0..array.rows {
        let start = row * array.cols;
        out.push(decode_utf16_code_units(
            &array.data()[start..start + array.cols],
        )?);
    }
    Ok(out)
}

fn predict_utf16_error(error: runmat_value::Utf16DecodeError) -> RuntimeError {
    let mut builder = build_runtime_error(format!(
        "predict: character value is not valid UTF-16 text: {error}"
    ))
    .with_builtin(PREDICT_NAME)
    .with_source(error);
    if let Some(identifier) = ERROR_PREDICT_INVALID_ARGUMENT.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn fitclinear_utf16_error(error: runmat_value::Utf16DecodeError) -> RuntimeError {
    let mut builder = build_runtime_error(format!(
        "fitclinear: character value is not valid UTF-16 text: {error}"
    ))
    .with_builtin(FITCLINEAR_NAME)
    .with_source(error);
    if let Some(identifier) = ERROR_FITCLINEAR_INVALID_ARGUMENT.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(super) fn numeric_vector(value: &Value, name: &str) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Num(value) => Ok(vec![*value]),
        Value::Int(value) => {
            ensure_int_exact_f64(value, name)?;
            Ok(vec![value.to_f64()])
        }
        Value::Tensor(tensor) => {
            ensure_integer_tensor_exact(tensor, name)?;
            vector_values(tensor, name)
        }
        Value::LogicalArray(array) => Ok(array
            .data
            .iter()
            .map(|value| if *value == 0 { 0.0 } else { 1.0 })
            .collect()),
        Value::Bool(value) => Ok(vec![if *value { 1.0 } else { 0.0 }]),
        _ => Err(fitclinear_invalid(format!(
            "fitclinear: {name} must be numeric"
        ))),
    }
}

pub(super) fn vector_values(tensor: &Tensor, name: &str) -> BuiltinResult<Vec<f64>> {
    vector_values_predict(tensor, name).map_err(|err| fitclinear_invalid(err.message))
}

pub(super) fn vector_values_predict(tensor: &Tensor, name: &str) -> BuiltinResult<Vec<f64>> {
    let (rows, columns) = classification_matrix_dimensions(tensor);
    if tensor.shape().len() > 2 || (rows != 1 && columns != 1) {
        return Err(predict_invalid(format!("predict: {name} must be a vector")));
    }
    Ok(tensor::tensor_values_f64(tensor))
}

pub(super) fn numeric_scalar(value: &Value, name: &str) -> BuiltinResult<f64> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        ensure_int_exact_f64(&integer, name)?;
        return Ok(integer.to_f64());
    }
    match value {
        Value::Num(value) => Ok(*value),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            Ok(tensor::tensor_values_f64(tensor)[0])
        }
        Value::Bool(value) => Ok(if *value { 1.0 } else { 0.0 }),
        _ => Err(fitclinear_invalid(format!(
            "fitclinear: {name} must be a numeric scalar"
        ))),
    }
}

pub(super) fn ensure_integer_tensor_exact(tensor: &Tensor, name: &str) -> BuiltinResult<()> {
    if let Some(storage) = tensor.integer_storage() {
        for index in 0..storage.len() {
            let value = storage
                .value_at(index)
                .expect("index is within integer storage length");
            ensure_int_exact_f64(&value, name)?;
        }
    }
    Ok(())
}

pub(super) fn ensure_integer_tensor_exact_predict(
    tensor: &Tensor,
    name: &str,
) -> BuiltinResult<()> {
    if let Some(storage) = tensor.integer_storage() {
        for index in 0..storage.len() {
            let value = storage
                .value_at(index)
                .expect("index is within integer storage length");
            if !crate::builtins::common::validation::integer_is_exact_f64(&value) {
                return Err(predict_invalid(format!(
                    "predict: integer {name} must be exactly representable as double"
                )));
            }
        }
    }
    Ok(())
}

fn ensure_int_exact_f64(value: &IntValue, name: &str) -> BuiltinResult<()> {
    if crate::builtins::common::validation::integer_is_exact_f64(value) {
        Ok(())
    } else {
        Err(fitclinear_invalid(format!(
            "fitclinear: integer {name} must be exactly representable as double"
        )))
    }
}

pub(super) fn logical_scalar(value: &Value, name: &str) -> BuiltinResult<bool> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return match integer.try_to_usize() {
            Some(0) => Ok(false),
            Some(1) => Ok(true),
            _ => Err(fitclinear_invalid(format!(
                "fitclinear: {name} must be logical scalar"
            ))),
        };
    }
    match value {
        Value::Bool(value) => Ok(*value),
        Value::LogicalArray(array) if array.data.len() == 1 => Ok(array.data[0] != 0),
        Value::Num(value) if *value == 0.0 || *value == 1.0 => Ok(*value != 0.0),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            let raw = tensor::tensor_values_f64(tensor)[0];
            if raw == 0.0 || raw == 1.0 {
                Ok(raw != 0.0)
            } else {
                Err(fitclinear_invalid(format!(
                    "fitclinear: {name} must be logical scalar"
                )))
            }
        }
        _ => Err(fitclinear_invalid(format!(
            "fitclinear: {name} must be logical scalar"
        ))),
    }
}

pub(super) fn positive_integer(value: &Value, name: &str) -> BuiltinResult<usize> {
    let value = numeric_scalar(value, name)?;
    let upper_exclusive = (1u128 << usize::BITS) as f64;
    if !value.is_finite() || value < 1.0 || value >= upper_exclusive || value.fract() != 0.0 {
        return Err(fitclinear_invalid(format!(
            "fitclinear: {name} must be a positive integer"
        )));
    }
    Ok(value as usize)
}

pub(super) fn columns_to_tensor(columns: Vec<Vec<f64>>) -> BuiltinResult<Tensor> {
    if columns.is_empty() {
        return Err(fitclinear_invalid("fitclinear: no predictor columns"));
    }
    let rows = columns[0].len();
    let cols = columns.len();
    let mut data = Vec::with_capacity(rows * cols);
    for column in columns {
        if column.len() != rows {
            return Err(fitclinear_invalid(
                "fitclinear: predictor variables must have the same height",
            ));
        }
        data.extend(column);
    }
    Tensor::new(data, vec![rows, cols])
        .map_err(|err| fitclinear_internal(format!("fitclinear: {err}")))
}

pub(super) fn columns_to_tensor_predict(columns: Vec<Vec<f64>>) -> BuiltinResult<Tensor> {
    columns_to_tensor(columns).map_err(|err| predict_invalid(err.message))
}

pub(super) fn rows_to_tensor(rows: Vec<Vec<f64>>) -> BuiltinResult<Tensor> {
    if rows.is_empty() {
        return Err(fitclinear_invalid("fitclinear: no complete predictor rows"));
    }
    let row_count = rows.len();
    let col_count = rows[0].len();
    let mut data = Vec::with_capacity(row_count * col_count);
    for col in 0..col_count {
        for row in &rows {
            data.push(row[col]);
        }
    }
    Tensor::new(data, vec![row_count, col_count])
        .map_err(|err| fitclinear_internal(format!("fitclinear: {err}")))
}

pub(super) fn beta_value(models: &[TrainedModel], predictors: usize) -> BuiltinResult<Value> {
    let mut data = Vec::with_capacity(predictors * models.len());
    for model in models {
        data.extend(model.beta.iter().copied());
    }
    Ok(Value::Tensor(
        Tensor::new(data, vec![predictors, models.len()])
            .map_err(|err| fitclinear_internal(format!("fitclinear: {err}")))?,
    ))
}

pub(super) fn row_tensor(values: Vec<f64>) -> BuiltinResult<Value> {
    let cols = values.len();
    Ok(Value::Tensor(Tensor::new(values, vec![1, cols]).map_err(
        |err| fitclinear_internal(format!("fitclinear: {err}")),
    )?))
}

pub(super) fn x_value(x: &Tensor, row: usize, col: usize) -> f64 {
    let rows = classification_matrix_dimensions(x).0;
    tensor::tensor_value_f64(x, row + col * rows)
}

pub(super) fn dot_row(x: &Tensor, row: usize, beta: &[f64]) -> f64 {
    beta.iter()
        .enumerate()
        .map(|(col, coef)| x_value(x, row, col) * coef)
        .sum()
}

pub(super) fn soft_threshold(value: f64, threshold: f64) -> f64 {
    if value > threshold {
        value - threshold
    } else if value < -threshold {
        value + threshold
    } else {
        0.0
    }
}

pub(super) fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp_value = value.exp();
        exp_value / (1.0 + exp_value)
    }
}
