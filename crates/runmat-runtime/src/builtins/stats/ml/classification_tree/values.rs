use super::*;
use runmat_value::decode_utf16_code_units;

pub(super) fn scalar_text(value: &Value) -> BuiltinResult<Option<String>> {
    decode_scalar_text(value).map_err(fitctree_utf16_error)
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
                text.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| fitctree_invalid("fitctree: text entries must be nonmissing"))
            })
            .collect(),
        Value::Cell(cell) => cell
            .data()
            .iter()
            .map(|value| {
                scalar_text(value)?.ok_or_else(|| {
                    fitctree_invalid(format!("fitctree: {name} entries must be text"))
                })
            })
            .collect(),
        _ => Err(fitctree_invalid(format!(
            "fitctree: {name} must be text or a text collection"
        ))),
    }
}

pub(super) fn char_rows(array: &CharArray) -> BuiltinResult<Vec<String>> {
    decode_char_rows(array).map_err(predict_utf16_error)
}

fn fit_char_rows(array: &CharArray) -> BuiltinResult<Vec<String>> {
    decode_char_rows(array).map_err(fitctree_utf16_error)
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

fn fitctree_utf16_error(error: runmat_value::Utf16DecodeError) -> RuntimeError {
    let mut builder = build_runtime_error(format!(
        "fitctree: character value is not valid UTF-16 text: {error}"
    ))
    .with_builtin(FITCTREE_NAME)
    .with_source(error);
    if let Some(identifier) = ERROR_FITCTREE_INVALID_ARGUMENT.identifier {
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
        _ => Err(fitctree_invalid(format!(
            "fitctree: {name} must be numeric"
        ))),
    }
}

pub(super) fn vector_values(tensor: &Tensor, name: &str) -> BuiltinResult<Vec<f64>> {
    vector_values_predict(tensor, name).map_err(|err| fitctree_invalid(err.message))
}

pub(super) fn vector_values_predict(tensor: &Tensor, name: &str) -> BuiltinResult<Vec<f64>> {
    let (rows, columns) = classification_tree_matrix_dimensions(tensor);
    if tensor.shape().len() > 2 || (rows != 1 && columns != 1) {
        return Err(predict_invalid(format!("predict: {name} must be a vector")));
    }
    Ok(tensor::tensor_values_f64(tensor))
}

pub(super) fn nonnegative_integer(value: &Value, name: &str) -> BuiltinResult<usize> {
    let value = numeric_scalar(value, name)?;
    let upper_exclusive = (1u128 << usize::BITS) as f64;
    if !value.is_finite() || value < 0.0 || value >= upper_exclusive || value.fract() != 0.0 {
        return Err(fitctree_invalid(format!(
            "fitctree: {name} must be a nonnegative integer"
        )));
    }
    Ok(value as usize)
}

pub(super) fn positive_integer(value: &Value, name: &str) -> BuiltinResult<usize> {
    let value = numeric_scalar(value, name)?;
    let upper_exclusive = (1u128 << usize::BITS) as f64;
    if !value.is_finite() || value < 1.0 || value >= upper_exclusive || value.fract() != 0.0 {
        return Err(fitctree_invalid(format!(
            "fitctree: {name} must be a positive integer"
        )));
    }
    Ok(value as usize)
}

fn numeric_scalar(value: &Value, name: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(value) => Ok(*value),
        Value::Int(value) => {
            ensure_int_exact_f64(value, name)?;
            Ok(value.to_f64())
        }
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            ensure_integer_tensor_exact(tensor, name)?;
            Ok(tensor::tensor_values_f64(tensor)[0])
        }
        Value::Bool(value) => Ok(if *value { 1.0 } else { 0.0 }),
        _ => Err(fitctree_invalid(format!(
            "fitctree: {name} must be a numeric scalar"
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
        Err(fitctree_invalid(format!(
            "fitctree: integer {name} must be exactly representable as double"
        )))
    }
}

pub(super) fn columns_to_tensor(columns: Vec<Vec<f64>>) -> BuiltinResult<Tensor> {
    if columns.is_empty() {
        return Err(fitctree_invalid("fitctree: no predictor columns"));
    }
    let rows = columns[0].len();
    let cols = columns.len();
    let mut data = Vec::with_capacity(rows * cols);
    for column in columns {
        if column.len() != rows {
            return Err(fitctree_invalid(
                "fitctree: predictor variables must have the same height",
            ));
        }
        data.extend(column);
    }
    Tensor::new(data, vec![rows, cols]).map_err(|err| fitctree_internal(format!("fitctree: {err}")))
}

pub(super) fn columns_to_tensor_predict(columns: Vec<Vec<f64>>) -> BuiltinResult<Tensor> {
    if columns.is_empty() {
        return Err(predict_invalid("predict: no predictor columns"));
    }
    let rows = columns[0].len();
    let cols = columns.len();
    let mut data = Vec::with_capacity(rows * cols);
    for column in columns {
        if column.len() != rows {
            return Err(predict_invalid(
                "predict: predictor table variables must have the same height",
            ));
        }
        data.extend(column);
    }
    Tensor::new(data, vec![rows, cols]).map_err(|err| predict_internal(format!("predict: {err}")))
}

pub(super) fn x_value(tensor: &Tensor, row: usize, col: usize) -> f64 {
    tensor::tensor_value_f64(
        tensor,
        col * classification_tree_matrix_dimensions(tensor).0 + row,
    )
}

pub(super) fn string_column(values: &[String]) -> BuiltinResult<Value> {
    Ok(Value::StringArray(
        StringArray::new(values.to_vec(), vec![values.len(), 1]).map_err(predict_internal)?,
    ))
}

pub(super) fn numeric_property(object: &ObjectInstance, name: &str) -> BuiltinResult<Vec<f64>> {
    match object.properties().get(name) {
        Some(Value::Tensor(tensor)) => Ok(tensor::tensor_values_f64(tensor)),
        _ => Err(predict_invalid(format!("predict: tree is missing {name}"))),
    }
}

pub(super) fn matrix_property(object: &ObjectInstance, name: &str) -> BuiltinResult<Vec<f64>> {
    numeric_property(object, name)
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
        Some(Value::String(runmat_value::StringElement::Text(text))) => {
            Ok(vec![text.as_str().to_owned()])
        }
        _ => Err(predict_invalid(format!("predict: tree is missing {name}"))),
    }
}

pub(super) fn tree_children_tensor(nodes: &[Node]) -> BuiltinResult<Value> {
    let rows = nodes.len();
    let mut data = Vec::with_capacity(rows * 2);
    for node in nodes {
        data.push(node.left.map(|idx| idx as f64 + 1.0).unwrap_or(0.0));
    }
    for node in nodes {
        data.push(node.right.map(|idx| idx as f64 + 1.0).unwrap_or(0.0));
    }
    Ok(Value::Tensor(Tensor::new(data, vec![rows, 2]).map_err(
        |err| fitctree_internal(format!("fitctree: {err}")),
    )?))
}

pub(super) fn tree_predictor_tensor(nodes: &[Node]) -> BuiltinResult<Value> {
    Ok(Value::Tensor(
        Tensor::new(
            nodes
                .iter()
                .map(|node| node.predictor.map(|idx| idx as f64 + 1.0).unwrap_or(0.0))
                .collect(),
            vec![nodes.len(), 1],
        )
        .map_err(|err| fitctree_internal(format!("fitctree: {err}")))?,
    ))
}

pub(super) fn tree_threshold_tensor(nodes: &[Node]) -> BuiltinResult<Value> {
    Ok(Value::Tensor(
        Tensor::new(
            nodes.iter().map(|node| node.threshold).collect(),
            vec![nodes.len(), 1],
        )
        .map_err(|err| fitctree_internal(format!("fitctree: {err}")))?,
    ))
}

pub(super) fn tree_class_tensor(nodes: &[Node]) -> BuiltinResult<Value> {
    Ok(Value::Tensor(
        Tensor::new(
            nodes
                .iter()
                .map(|node| node.class_index as f64 + 1.0)
                .collect(),
            vec![nodes.len(), 1],
        )
        .map_err(|err| fitctree_internal(format!("fitctree: {err}")))?,
    ))
}

pub(super) fn tree_prob_tensor(nodes: &[Node], class_count: usize) -> BuiltinResult<Value> {
    let rows = nodes.len();
    let mut data = Vec::with_capacity(rows * class_count);
    for class in 0..class_count {
        for node in nodes {
            data.push(node.probabilities.get(class).copied().unwrap_or(0.0));
        }
    }
    Ok(Value::Tensor(
        Tensor::new(data, vec![rows, class_count])
            .map_err(|err| fitctree_internal(format!("fitctree: {err}")))?,
    ))
}
