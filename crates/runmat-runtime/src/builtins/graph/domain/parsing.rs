use super::*;

pub(super) fn is_numeric_weight_arg(value: &Value) -> bool {
    matches!(
        value,
        Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::Tensor(_)
            | Value::LogicalArray(_)
            | Value::GpuTensor(_)
    )
}

pub(super) fn weights_from_value(
    value: &Value,
    len: usize,
    name: &'static str,
) -> BuiltinResult<Vec<f64>> {
    let values = numeric_f64_vector(value, name)?;
    if values.len() != len {
        return Err(graph_error(
            name,
            format!("{name}: weights must have one value per edge"),
        ));
    }
    for weight in &values {
        if !weight.is_finite() || *weight < 0.0 {
            return Err(graph_error(
                name,
                format!("{name}: edge weights must be finite nonnegative values"),
            ));
        }
    }
    Ok(values)
}

pub(super) fn numeric_vector(value: &Value, name: &'static str) -> BuiltinResult<Vec<usize>> {
    if let Value::Tensor(tensor) = value {
        if !is_vector_shape(tensor.shape()) {
            return Err(graph_error(
                name,
                format!("{name}: expected a vector of node indices"),
            ));
        }
    }
    if let Value::LogicalArray(array) = value {
        if !is_vector_shape(&array.shape) {
            return Err(graph_error(
                name,
                format!("{name}: expected a vector of node indices"),
            ));
        }
    }
    if let Some(values) = exact_integer_vector(value, name) {
        return values;
    }
    let values = numeric_f64_vector(value, name)?;
    values
        .into_iter()
        .map(|value| {
            if !value.is_finite()
                || value < 0.0
                || value.fract() != 0.0
                || value > usize::MAX as f64
                || (usize::BITS == 64 && value == usize::MAX as f64)
            {
                Err(graph_error(
                    name,
                    format!("{name}: node indices must be nonnegative integers"),
                ))
            } else {
                Ok(value as usize)
            }
        })
        .collect()
}

pub(super) fn numeric_f64_vector(value: &Value, name: &'static str) -> BuiltinResult<Vec<f64>> {
    if let Value::Tensor(tensor) = value {
        if !is_vector_shape(tensor.shape()) {
            return Err(graph_error(
                name,
                format!("{name}: expected a numeric vector"),
            ));
        }
    }
    if let Value::LogicalArray(array) = value {
        if !is_vector_shape(&array.shape) {
            return Err(graph_error(
                name,
                format!("{name}: expected a numeric vector"),
            ));
        }
    }
    if !crate::builtins::common::validation::native_integer_value_is_exact_f64(value) {
        return Err(graph_error(
            name,
            format!("{name}: integer values cannot be represented exactly as graph weights"),
        ));
    }
    match value {
        Value::Num(n) => Ok(vec![*n]),
        Value::Int(i) => Ok(vec![i.to_f64()]),
        Value::Bool(b) => Ok(vec![if *b { 1.0 } else { 0.0 }]),
        Value::Tensor(tensor) => Ok(tensor::tensor_values_f64(tensor)),
        Value::LogicalArray(array) => Ok(array
            .data
            .iter()
            .map(|&v| if v != 0 { 1.0 } else { 0.0 })
            .collect()),
        other => Err(graph_error(
            name,
            format!("{name}: expected numeric vector, got {other:?}"),
        )),
    }
}

pub(super) fn exact_integer_vector(
    value: &Value,
    name: &'static str,
) -> Option<BuiltinResult<Vec<usize>>> {
    match value {
        Value::Int(value) => Some(parse_nonnegative_integer(value, name).map(|value| vec![value])),
        Value::Tensor(tensor) => tensor.integer_storage().map(|storage| {
            storage
                .exact_values()
                .into_iter()
                .map(|value| parse_nonnegative_integer(&value, name))
                .collect()
        }),
        _ => None,
    }
}

pub(super) fn parse_nonnegative_integer(
    value: &IntValue,
    name: &'static str,
) -> BuiltinResult<usize> {
    value.try_to_usize().ok_or_else(|| {
        graph_error(
            name,
            format!("{name}: node indices must be nonnegative integers"),
        )
    })
}

pub(super) fn parse_node_names(
    value: &Value,
    expected_len: usize,
    name: &'static str,
) -> BuiltinResult<Vec<String>> {
    let names = parse_string_vector(value).map_err(|err| graph_error(name, err))?;
    if names.len() != expected_len {
        return Err(graph_error(
            name,
            format!("{name}: node name list length must match node count"),
        ));
    }
    Ok(names)
}

pub(super) fn parse_string_vector(value: &Value) -> Result<Vec<String>, String> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => {
            Ok(vec![text.as_str().to_owned()])
        }
        Value::StringArray(array) if is_vector_shape(&array.shape) => array
            .data()
            .iter()
            .map(|text| {
                text.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| "expected nonmissing text".to_owned())
            })
            .collect(),
        Value::CharArray(chars) if chars.rows == 1 => {
            Ok(vec![runmat_value::decode_utf16_code_units(chars.data())
                .map_err(|source| source.to_string())?])
        }
        Value::CharArray(chars) => (0..chars.rows)
            .map(|row| {
                Ok(runmat_value::decode_utf16_code_units(
                    &chars.data()[row * chars.cols..(row + 1) * chars.cols],
                )
                .map_err(|source| source.to_string())?
                .trim_end()
                .to_owned())
            })
            .collect(),
        Value::Cell(cell) if is_vector_shape(&cell.shape) => cell
            .data()
            .iter()
            .map(|entry| {
                scalar_text(entry)?.ok_or_else(|| {
                    "expected cell array of string or character node names".to_string()
                })
            })
            .collect(),
        other => Err(format!(
            "expected string, character, or cellstr vector, got {other:?}"
        )),
    }
}

pub(super) fn is_vector_shape(shape: &[usize]) -> bool {
    shape.len() <= 2 && (shape.len() < 2 || shape[0] <= 1 || shape[1] <= 1)
}

pub(super) fn scalar_text(value: &Value) -> Result<Option<String>, String> {
    Ok(match value {
        Value::String(runmat_value::StringElement::Text(text)) => Some(text.as_str().to_owned()),
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Some(
                array.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string")
                    .as_str()
                    .to_owned(),
            )
        }
        Value::CharArray(chars) if chars.rows == 1 => Some(
            runmat_value::decode_utf16_code_units(chars.data())
                .map_err(|source| source.to_string())?,
        ),
        _ => None,
    })
}

pub(super) fn canonical(text: &str) -> String {
    text.chars()
        .filter(|ch| *ch != '_' && *ch != '-' && !ch.is_whitespace())
        .flat_map(char::to_lowercase)
        .collect()
}

pub(super) fn positive_integer_scalar(value: &Value, name: &'static str) -> BuiltinResult<usize> {
    let values = numeric_vector(value, name)?;
    if values.len() != 1 {
        return Err(graph_error(
            name,
            format!("{name}: expected scalar node count"),
        ));
    }
    Ok(values[0])
}

pub(super) fn node_index_from_tensor(
    tensor: &Tensor,
    index: usize,
    node_count: usize,
    name: &'static str,
) -> BuiltinResult<usize> {
    if let Some(storage) = tensor.integer_storage() {
        let value = storage
            .value_at(index)
            .ok_or_else(|| graph_error(name, format!("{name}: endpoint index is out of bounds")))?;
        return node_index_from_integer(&value, node_count, name);
    }
    node_index_from_f64(tensor::tensor_value_f64(tensor, index), node_count, name)
}

pub(super) fn node_index_from_integer(
    value: &IntValue,
    node_count: usize,
    name: &'static str,
) -> BuiltinResult<usize> {
    let Some(value) = value.try_to_usize() else {
        return Err(graph_error(
            name,
            format!("{name}: edge endpoints must be positive integer node ids"),
        ));
    };
    node_index_from_one_based(value, node_count, name)
}

pub(super) fn node_index_from_f64(
    value: f64,
    node_count: usize,
    name: &'static str,
) -> BuiltinResult<usize> {
    if !value.is_finite()
        || value < 1.0
        || value.fract() != 0.0
        || value > usize::MAX as f64
        || (usize::BITS == 64 && value == usize::MAX as f64)
    {
        return Err(graph_error(
            name,
            format!("{name}: edge endpoints must be positive integer node ids"),
        ));
    }
    node_index_from_one_based(value as usize, node_count, name)
}

pub(super) fn node_index_from_one_based(
    value: usize,
    node_count: usize,
    name: &'static str,
) -> BuiltinResult<usize> {
    if value == 0 || value > node_count {
        return Err(graph_error(
            name,
            format!("{name}: node index is outside the graph"),
        ));
    }
    Ok(value - 1)
}
