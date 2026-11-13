use super::*;
use crate::builtins::io::json::layout;

pub(crate) fn value_from_json(value: &JsonValue) -> BuiltinResult<Value> {
    match value {
        JsonValue::Null => empty_double(),
        JsonValue::Bool(b) => Ok(Value::Bool(*b)),
        JsonValue::Number(num) => parse_json_number(num).map(Value::Num),
        JsonValue::String(s) => {
            let char_array = CharArray::new_row(s);
            Ok(Value::CharArray(char_array))
        }
        JsonValue::Array(arr) => decode_json_array(arr),
        JsonValue::Object(map) => decode_json_object(map),
    }
}

fn decode_json_object(map: &serde_json::Map<String, JsonValue>) -> BuiltinResult<Value> {
    let mut struct_value = runmat_value::UnboundedStructValueBuilder::new();
    for (key, val) in map {
        struct_value
            .fields
            .insert(key.clone(), value_from_json(val)?);
    }
    Ok(Value::Struct(struct_value.finish()))
}

fn parse_json_number(number: &serde_json::Number) -> BuiltinResult<f64> {
    if let Some(value) = number.as_f64() {
        return Ok(value);
    }
    let text = number.to_string();
    match text.parse::<f64>() {
        Ok(value) => {
            if value.is_nan() {
                Err(jsondecode_error_with(
                    &JSONDECODE_ERROR_NUMERIC_LITERAL,
                    format!(
                        "{}: unsupported numeric literal ({text})",
                        JSONDECODE_ERROR_NUMERIC_LITERAL.message
                    ),
                ))
            } else {
                Ok(value)
            }
        }
        Err(_) => {
            let display = if text.len() > 64 {
                format!("{}...", &text[..64])
            } else {
                text
            };
            Err(jsondecode_error_with(
                &JSONDECODE_ERROR_NUMERIC_LITERAL,
                format!(
                    "{}: numeric literal out of range ({display})",
                    JSONDECODE_ERROR_NUMERIC_LITERAL.message
                ),
            ))
        }
    }
}

fn decode_json_array(values: &[JsonValue]) -> BuiltinResult<Value> {
    if values.is_empty() {
        let tensor = Tensor::new(Vec::new(), vec![0, 0]).map_err(|e| {
            jsondecode_error_with(
                &JSONDECODE_ERROR_INTERNAL,
                format!("{} ({e})", JSONDECODE_ERROR_INTERNAL.message),
            )
        })?;
        return Ok(Value::Tensor(tensor));
    }

    if let Some(numeric) = parse_numeric_array(values)? {
        let tensor = Tensor::new(numeric.data, numeric.shape).map_err(|e| {
            jsondecode_error_with(
                &JSONDECODE_ERROR_INTERNAL,
                format!("{} ({e})", JSONDECODE_ERROR_INTERNAL.message),
            )
        })?;
        return Ok(Value::Tensor(tensor));
    }

    if let Some(logical) = parse_logical_array(values) {
        let array = LogicalArray::new(logical.data, logical.shape).map_err(|e| {
            jsondecode_error_with(
                &JSONDECODE_ERROR_INTERNAL,
                format!("{} ({e})", JSONDECODE_ERROR_INTERNAL.message),
            )
        })?;
        return Ok(Value::LogicalArray(array));
    }

    if let Some(strings) = parse_string_array(values) {
        let array = StringArray::new(strings.data, strings.shape).map_err(|e| {
            jsondecode_error_with(
                &JSONDECODE_ERROR_INTERNAL,
                format!("{} ({e})", JSONDECODE_ERROR_INTERNAL.message),
            )
        })?;
        return Ok(Value::StringArray(array));
    }

    if let Some(structures) = parse_struct_array(values)? {
        return Ok(structures);
    }

    if let Some(cell) = parse_rectangular_cell_array(values)? {
        return Ok(cell);
    }

    let mut elements = Vec::with_capacity(values.len());
    for element in values {
        elements.push(value_from_json(element)?);
    }
    if values.iter().all(JsonValue::is_object) {
        cell_matrix(elements, values.len(), 1)
    } else {
        cell_row(elements)
    }
}

fn parse_struct_array(values: &[JsonValue]) -> BuiltinResult<Option<Value>> {
    let Some((json_shape, leaves)) = layout::rectangular_leaves(values) else {
        return Ok(None);
    };
    let Some(first) = leaves.first().and_then(|value| value.as_object()) else {
        return Ok(None);
    };
    let field_names = first.keys().map(ToOwned::to_owned).collect::<Vec<_>>();
    let expected = field_names.iter().collect::<std::collections::HashSet<_>>();
    let mut row_major = Vec::with_capacity(leaves.len());
    for leaf in leaves {
        let Some(object) = leaf.as_object() else {
            return Ok(None);
        };
        if object.len() != field_names.len()
            || object.keys().collect::<std::collections::HashSet<_>>() != expected
        {
            return Ok(None);
        }
        let mut structure = runmat_value::UnboundedStructValueBuilder::new();
        for name in &field_names {
            structure.insert(name.clone(), value_from_json(&object[name])?);
        }
        let structure = structure.finish();
        row_major.push(structure);
    }
    let shape = if json_shape.len() == 1 {
        vec![json_shape[0], 1]
    } else {
        json_shape
    };
    let column_major = layout::row_to_column_major(row_major, &shape)
        .map_err(|error| jsondecode_error_with(&JSONDECODE_ERROR_INTERNAL, error))?;
    runmat_value::StructArray::normalize(field_names, column_major, shape)
        .map(Some)
        .map_err(|error| jsondecode_error_with(&JSONDECODE_ERROR_INTERNAL, error))
}

fn empty_double() -> BuiltinResult<Value> {
    Tensor::new(Vec::new(), vec![0, 0])
        .map(Value::Tensor)
        .map_err(|e| {
            jsondecode_error_with(
                &JSONDECODE_ERROR_INTERNAL,
                format!("{} ({e})", JSONDECODE_ERROR_INTERNAL.message),
            )
        })
}

fn cell_matrix(elements: Vec<Value>, rows: usize, cols: usize) -> BuiltinResult<Value> {
    let cell = CellArray::new(elements, rows, cols).map_err(|e| {
        jsondecode_error_with(
            &JSONDECODE_ERROR_INTERNAL,
            format!("{} ({e})", JSONDECODE_ERROR_INTERNAL.message),
        )
    })?;
    Ok(Value::Cell(cell))
}

fn cell_row(elements: Vec<Value>) -> BuiltinResult<Value> {
    let cols = elements.len();
    cell_matrix(elements, 1, cols)
}

struct NumericTensor {
    data: Vec<f64>,
    shape: Vec<usize>,
}

fn parse_numeric_array(values: &[JsonValue]) -> BuiltinResult<Option<NumericTensor>> {
    if values.is_empty() {
        return Ok(None);
    }

    if values.iter().all(|v| v.is_number()) {
        let mut data = Vec::with_capacity(values.len());
        for value in values {
            if let JsonValue::Number(number) = value {
                data.push(parse_json_number(number)?);
            }
        }
        return Ok(Some(NumericTensor {
            data,
            shape: vec![values.len()],
        }));
    }

    if values.iter().all(|v| v.is_array()) {
        let mut children = Vec::with_capacity(values.len());
        for value in values {
            let Some(child_values) = value.as_array() else {
                return Ok(None);
            };
            let Some(child) = parse_numeric_array(child_values)? else {
                return Ok(None);
            };
            children.push(child);
        }
        if children.is_empty() {
            return Ok(None);
        }
        let first_shape = children[0].shape.clone();
        if !children.iter().all(|child| child.shape == first_shape) {
            return Ok(None);
        }
        let mut shape = Vec::with_capacity(first_shape.len() + 1);
        shape.push(children.len());
        shape.extend(first_shape.clone());

        let total = shape.iter().try_fold(1usize, |count, extent| {
            count.checked_mul(*extent).ok_or_else(|| {
                jsondecode_error_with(
                    &JSONDECODE_ERROR_INTERNAL,
                    "jsondecode: numeric array shape exceeds addressable memory",
                )
            })
        })?;
        let rows = shape[0];
        if rows == 0 {
            return Ok(None);
        }
        let inner = total / rows;
        let mut data = vec![0.0; total];
        for (row, child) in children.into_iter().enumerate() {
            if child.data.len() != inner {
                return Ok(None);
            }
            for (idx, value) in child.data.into_iter().enumerate() {
                let offset = row + rows * idx;
                data[offset] = value;
            }
        }
        return Ok(Some(NumericTensor { data, shape }));
    }

    Ok(None)
}

struct LogicalTensor {
    data: Vec<u8>,
    shape: Vec<usize>,
}

fn parse_logical_array(values: &[JsonValue]) -> Option<LogicalTensor> {
    if values.is_empty() {
        return None;
    }

    if values.iter().all(|v| v.is_boolean()) {
        let mut data = Vec::with_capacity(values.len());
        for value in values {
            data.push(if value.as_bool()? { 1 } else { 0 });
        }
        return Some(LogicalTensor {
            data,
            shape: vec![values.len()],
        });
    }

    if values.iter().all(|v| v.is_array()) {
        let mut children = Vec::with_capacity(values.len());
        for value in values {
            let child = parse_logical_array(value.as_array()?)?;
            children.push(child);
        }
        if children.is_empty() {
            return None;
        }
        let first_shape = children[0].shape.clone();
        if !children.iter().all(|child| child.shape == first_shape) {
            return None;
        }
        let mut shape = Vec::with_capacity(first_shape.len() + 1);
        shape.push(children.len());
        shape.extend(first_shape.clone());

        let total = shape
            .iter()
            .try_fold(1usize, |count, extent| count.checked_mul(*extent))?;
        let rows = shape[0];
        if rows == 0 {
            return None;
        }
        let inner = total / rows;
        let mut data = vec![0u8; total];
        for (row, child) in children.into_iter().enumerate() {
            if child.data.len() != inner {
                return None;
            }
            for (idx, value) in child.data.into_iter().enumerate() {
                let offset = row + rows * idx;
                data[offset] = value;
            }
        }
        return Some(LogicalTensor { data, shape });
    }

    None
}

struct StringTensor {
    data: Vec<String>,
    shape: Vec<usize>,
}

fn parse_string_array(values: &[JsonValue]) -> Option<StringTensor> {
    if values.is_empty() {
        return None;
    }

    if values.iter().all(|v| v.is_string()) {
        let mut data = Vec::with_capacity(values.len());
        for value in values {
            data.push(value.as_str()?.to_string());
        }
        return Some(StringTensor {
            data,
            shape: vec![values.len()],
        });
    }

    if values.iter().all(|v| v.is_array()) {
        let mut children = Vec::with_capacity(values.len());
        for value in values {
            let child = parse_string_array(value.as_array()?)?;
            children.push(child);
        }
        if children.is_empty() {
            return None;
        }
        let first_shape = children[0].shape.clone();
        if !children.iter().all(|child| child.shape == first_shape) {
            return None;
        }
        let mut shape = Vec::with_capacity(first_shape.len() + 1);
        shape.push(children.len());
        shape.extend(first_shape.clone());

        let total = shape
            .iter()
            .try_fold(1usize, |count, extent| count.checked_mul(*extent))?;
        let rows = shape[0];
        if rows == 0 {
            return None;
        }
        let inner = total / rows;
        let mut data = vec![String::new(); total];
        for (row, mut child) in children.into_iter().enumerate() {
            if child.data.len() != inner {
                return None;
            }
            for (idx, value) in child.data.drain(..).enumerate() {
                let offset = row + rows * idx;
                data[offset] = value;
            }
        }
        return Some(StringTensor { data, shape });
    }

    None
}

fn parse_rectangular_cell_array(values: &[JsonValue]) -> BuiltinResult<Option<Value>> {
    if values.is_empty() || !values.iter().all(|v| v.is_array()) {
        return Ok(None);
    }

    let mut expected_len: Option<usize> = None;
    for value in values {
        let arr = value.as_array().ok_or_else(|| {
            // Should be unreachable due to the `all(|v| v.is_array())` guard above.
            jsondecode_error_with(
                &JSONDECODE_ERROR_INTERNAL,
                "jsondecode: inconsistent array state",
            )
        })?;
        match expected_len {
            Some(len) if arr.len() != len => return Ok(None),
            None => expected_len = Some(arr.len()),
            _ => {}
        }
    }

    let cols = expected_len.unwrap_or(0);
    let rows = values.len();

    let mut elements = Vec::with_capacity(rows.saturating_mul(cols));
    if cols > 0 {
        for value in values {
            let arr = value.as_array().expect("validated array value");
            for element in arr {
                elements.push(value_from_json(element)?);
            }
        }
    }

    let cell = cell_matrix(elements, rows, cols)?;
    Ok(Some(cell))
}
