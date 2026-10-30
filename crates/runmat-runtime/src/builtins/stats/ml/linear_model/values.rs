use super::*;
use runmat_value::ObjectInstance;

pub(super) fn terms_from_object(object: &ObjectInstance) -> BuiltinResult<Vec<Vec<u32>>> {
    let tensor = match object.properties().get("__RunMatTerms") {
        Some(Value::Tensor(tensor)) => tensor,
        _ => return Err(predict_invalid("predict: model is missing term metadata")),
    };
    let (rows, cols) = fitlm_matrix_dimensions(tensor);
    let mut terms = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut term = Vec::with_capacity(cols);
        for col in 0..cols {
            let value = x_value(tensor, row, col);
            if value < 0.0 || value.fract().abs() > EPS {
                return Err(predict_invalid("predict: model term metadata is invalid"));
            }
            term.push(value as u32);
        }
        terms.push(term);
    }
    Ok(terms)
}

pub(super) fn numeric_property(object: &ObjectInstance, name: &str) -> BuiltinResult<Vec<f64>> {
    match object.properties().get(name) {
        Some(Value::Tensor(tensor)) => Ok(tensor::tensor_values_f64(tensor)),
        Some(Value::Num(value)) => Ok(vec![*value]),
        _ => Err(predict_invalid(format!(
            "predict: model is missing numeric property '{name}'"
        ))),
    }
}

pub(super) fn matrix_property(object: &ObjectInstance, name: &str) -> BuiltinResult<Vec<f64>> {
    match object.properties().get(name) {
        Some(Value::Tensor(tensor))
            if fitlm_matrix_dimensions(tensor).0 == fitlm_matrix_dimensions(tensor).1 =>
        {
            Ok(tensor::tensor_values_f64(tensor))
        }
        _ => Err(predict_invalid(format!(
            "predict: model is missing matrix property '{name}'"
        ))),
    }
}

pub(super) fn numeric_scalar_property(object: &ObjectInstance, name: &str) -> BuiltinResult<f64> {
    match object.properties().get(name) {
        Some(Value::Num(value)) => Ok(*value),
        Some(Value::Tensor(tensor)) if tensor::is_scalar_tensor(tensor) => {
            Ok(tensor::tensor_values_f64(tensor)[0])
        }
        _ => Err(predict_invalid(format!(
            "predict: model is missing scalar property '{name}'"
        ))),
    }
}

pub(super) fn string_property(object: &ObjectInstance, name: &str) -> BuiltinResult<Vec<String>> {
    match object.properties().get(name) {
        Some(Value::StringArray(array)) => array
            .data()
            .iter()
            .map(|text| {
                text.as_str().map(str::to_owned).ok_or_else(|| {
                    predict_invalid("predict: model text metadata must be nonmissing")
                })
            })
            .collect(),
        Some(Value::Cell(cell)) => cell
            .data()
            .iter()
            .map(|value| -> BuiltinResult<String> {
                scalar_text(value)?
                    .map(std::borrow::Cow::into_owned)
                    .ok_or_else(|| {
                        predict_invalid(format!("predict: property '{name}' is invalid"))
                    })
            })
            .collect(),
        _ => Err(predict_invalid(format!(
            "predict: model is missing string property '{name}'"
        ))),
    }
}

pub(super) fn vector_values(tensor: &Tensor, label: &str) -> BuiltinResult<Vec<f64>> {
    let (rows, cols) = fitlm_matrix_dimensions(tensor);
    if tensor.shape().len() > 2 || !(rows == 1 || cols == 1) {
        return Err(fitlm_invalid(format!("fitlm: {label} must be a vector")));
    }
    Ok(tensor::tensor_values_f64(tensor))
}

pub(super) fn ensure_exact_fitlm_integer_tensor_boundary(
    tensor: &Tensor,
    role: &str,
) -> BuiltinResult<()> {
    let Some(storage) = tensor.integer_storage() else {
        return Ok(());
    };
    for index in 0..storage.len() {
        let integer = storage
            .value_at(index)
            .expect("index is within integer storage length");
        if !crate::builtins::common::validation::integer_is_exact_f64(&integer) {
            return Err(fitlm_invalid(format!(
                "fitlm: integer {role} values must be exactly representable as double"
            )));
        }
    }
    Ok(())
}

pub(super) fn ensure_exact_fitlm_integer_scalar_boundary(
    value: NumericScalar,
    role: &str,
) -> BuiltinResult<()> {
    let Some(integer) = value.into_int_value() else {
        return Ok(());
    };
    if !crate::builtins::common::validation::integer_is_exact_f64(&integer) {
        return Err(fitlm_invalid(format!(
            "fitlm: integer {role} values must be exactly representable as double"
        )));
    }
    Ok(())
}

pub(super) fn vector_values_predict(tensor: &Tensor, label: &str) -> BuiltinResult<Vec<f64>> {
    let (rows, cols) = fitlm_matrix_dimensions(tensor);
    if tensor.shape().len() > 2 || !(rows == 1 || cols == 1) {
        return Err(predict_invalid(format!(
            "predict: {label} must be a vector"
        )));
    }
    Ok(tensor::tensor_values_f64(tensor))
}

pub(super) fn numeric_vector(value: &Value, name: &str) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Num(value) => Ok(vec![*value]),
        Value::Int(value) => {
            ensure_exact_fitlm_integer_scalar_boundary(NumericScalar::from(value.clone()), name)?;
            Ok(vec![value.to_f64()])
        }
        Value::Tensor(tensor) => {
            ensure_exact_fitlm_integer_tensor_boundary(tensor, name)?;
            vector_values(tensor, name)
        }
        _ => Err(fitlm_invalid(format!("fitlm: {name} must be numeric"))),
    }
}

pub(super) fn numeric_scalar(value: &Value, name: &str) -> BuiltinResult<f64> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return Ok(integer.to_f64());
    }
    match value {
        Value::Num(value) => Ok(*value),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            Ok(tensor::tensor_values_f64(tensor)[0])
        }
        _ => Err(fitlm_invalid(format!("fitlm: {name} must be a scalar"))),
    }
}

pub(super) fn exclude_spec(value: &Value) -> BuiltinResult<ExcludeSpec> {
    match value {
        Value::Bool(value) => Ok(ExcludeSpec::Mask(vec![*value])),
        Value::LogicalArray(array) => Ok(ExcludeSpec::Mask(
            array.data.iter().map(|value| *value != 0).collect(),
        )),
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            let indices = tensor
                .integer_storage()
                .expect("integer storage checked above")
                .exact_values()
                .iter()
                .map(|value| value.try_to_usize().filter(|value| *value > 0))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| fitlm_invalid("fitlm: Exclude indices must be positive integers"))?;
            Ok(ExcludeSpec::Indices(indices))
        }
        Value::Tensor(tensor) => {
            let values = tensor::tensor_values_f64(tensor);
            let mut indices = Vec::with_capacity(values.len());
            for value in values {
                if value < 1.0 || value.fract().abs() > EPS {
                    return Err(fitlm_invalid(
                        "fitlm: Exclude indices must be positive integers",
                    ));
                }
                indices.push(value as usize);
            }
            Ok(ExcludeSpec::Indices(indices))
        }
        _ => Err(fitlm_invalid(
            "fitlm: Exclude must be logical mask or row indices",
        )),
    }
}

pub(super) fn exclude_mask(
    spec: Option<&ExcludeSpec>,
    rows: usize,
) -> BuiltinResult<Option<Vec<bool>>> {
    let Some(spec) = spec else {
        return Ok(None);
    };
    let mut mask = vec![false; rows];
    match spec {
        ExcludeSpec::Mask(values) => {
            if values.len() == 1 {
                mask.fill(values[0]);
            } else if values.len() == rows {
                mask.clone_from(values);
            } else {
                return Err(fitlm_invalid(
                    "fitlm: Exclude mask length must match the number of observations",
                ));
            }
        }
        ExcludeSpec::Indices(indices) => {
            for index in indices {
                if *index == 0 || *index > rows {
                    return Err(fitlm_invalid("fitlm: Exclude index exceeds observations"));
                }
                mask[*index - 1] = true;
            }
        }
    }
    Ok(Some(mask))
}

pub(super) fn bool_scalar(value: &Value, name: &str) -> BuiltinResult<bool> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return Ok(!integer.is_zero());
    }
    match value {
        Value::Bool(value) => Ok(*value),
        Value::Num(value) => Ok(*value != 0.0),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            Ok(tensor::tensor_values_f64(tensor)[0] != 0.0)
        }
        _ => Err(fitlm_invalid(format!(
            "fitlm: {name} must be logical scalar"
        ))),
    }
}

pub(super) fn scalar_text(value: &Value) -> BuiltinResult<Option<std::borrow::Cow<'_, str>>> {
    Ok(match value {
        Value::String(runmat_value::StringElement::Text(text)) => {
            Some(std::borrow::Cow::Borrowed(text.as_str()))
        }
        Value::CharArray(chars) => Some(std::borrow::Cow::Owned(chars_to_string(chars)?)),
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            array
                .data()
                .first()
                .and_then(|text| text.as_str().map(std::borrow::Cow::Borrowed))
        }
        _ => None,
    })
}

pub(super) fn text_list(value: &Value, name: &str) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => {
            Ok(vec![text.as_str().to_owned()])
        }
        Value::CharArray(chars) => Ok(vec![chars_to_string(chars)?]),
        Value::StringArray(array) => array
            .data()
            .iter()
            .map(|text| {
                text.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| fitlm_invalid("fitlm: text entries must be nonmissing"))
            })
            .collect(),
        Value::Cell(cell) => cell
            .data()
            .iter()
            .map(|value| {
                scalar_text(value)?
                    .map(std::borrow::Cow::into_owned)
                    .ok_or_else(|| fitlm_invalid(format!("fitlm: {name} entries must be text")))
            })
            .collect(),
        _ => Err(fitlm_invalid(format!(
            "fitlm: {name} must be text or a text collection"
        ))),
    }
}

fn chars_to_string(chars: &CharArray) -> BuiltinResult<String> {
    crate::builtins::stats::decode_utf16_with_error(chars.data(), || {
        fitlm_invalid("fitlm: character value is not valid UTF-16 text")
    })
}

pub(super) fn columns_to_tensor(columns: Vec<Vec<f64>>) -> BuiltinResult<Tensor> {
    if columns.is_empty() {
        return Tensor::new(Vec::new(), vec![0, 0])
            .map_err(|err| fitlm_internal(format!("fitlm: {err}")));
    }
    let rows = columns[0].len();
    let cols = columns.len();
    let mut data = Vec::with_capacity(rows * cols);
    for column in columns {
        if column.len() != rows {
            return Err(fitlm_invalid(
                "fitlm: predictor variables must have the same height",
            ));
        }
        data.extend(column);
    }
    Tensor::new(data, vec![rows, cols]).map_err(|err| fitlm_internal(format!("fitlm: {err}")))
}

pub(super) fn columns_to_tensor_predict(columns: Vec<Vec<f64>>) -> BuiltinResult<Tensor> {
    if columns.is_empty() {
        return Tensor::new(Vec::new(), vec![0, 0])
            .map_err(|err| predict_internal(format!("predict: {err}")));
    }
    let rows = columns[0].len();
    let cols = columns.len();
    let mut data = Vec::with_capacity(rows * cols);
    for column in columns {
        data.extend(column);
    }
    Tensor::new(data, vec![rows, cols]).map_err(|err| predict_internal(format!("predict: {err}")))
}

pub(super) fn x_value(tensor: &Tensor, row: usize, col: usize) -> f64 {
    tensor::tensor_value_f64(tensor, col * fitlm_matrix_dimensions(tensor).0 + row)
}

pub(super) fn terms_tensor(terms: &[Vec<u32>]) -> BuiltinResult<Value> {
    let rows = terms.len();
    let cols = terms.first().map(|term| term.len()).unwrap_or(0);
    let mut data = Vec::with_capacity(rows * cols);
    for col in 0..cols {
        for term in terms {
            data.push(term[col] as f64);
        }
    }
    Ok(Value::Tensor(Tensor::new(data, vec![rows, cols]).map_err(
        |err| fitlm_internal(format!("fitlm: terms metadata error: {err}")),
    )?))
}

pub(super) fn numeric_column(values: &[f64]) -> BuiltinResult<Value> {
    Ok(Value::Tensor(
        Tensor::new(values.to_vec(), vec![values.len(), 1])
            .map_err(|err| fitlm_internal(format!("fitlm: numeric column error: {err}")))?,
    ))
}

pub(super) fn string_row(values: Vec<String>) -> BuiltinResult<Value> {
    let len = values.len();
    Ok(Value::StringArray(
        StringArray::new(values, vec![1, len])
            .map_err(|err| fitlm_internal(format!("fitlm: string row error: {err}")))?,
    ))
}

pub(super) fn string_column(values: Vec<String>) -> BuiltinResult<Value> {
    let len = values.len();
    Ok(Value::StringArray(
        StringArray::new(values, vec![len, 1])
            .map_err(|err| fitlm_internal(format!("fitlm: string column error: {err}")))?,
    ))
}

pub(super) fn has_intercept(terms: &[Vec<u32>]) -> bool {
    terms
        .iter()
        .any(|term| term.iter().all(|power| *power == 0))
}
