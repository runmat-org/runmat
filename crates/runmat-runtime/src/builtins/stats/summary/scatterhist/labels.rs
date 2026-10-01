//! Scatterhist labels domain behavior.
use super::*;

pub(super) fn group_labels(
    value: &Value,
    expected_len: usize,
) -> BuiltinResult<Vec<Option<String>>> {
    match value {
        Value::Tensor(tensor) if tensor::tensor_element_len(tensor) == expected_len => {
            if let Some(storage) = tensor.integer_storage() {
                return Ok(storage
                    .exact_values()
                    .into_iter()
                    .map(|value| Some(integer_label(&value)))
                    .collect());
            }
            Ok(tensor
                .materialize_f64()
                .into_iter()
                .map(|value| value.is_finite().then(|| number_label(value)))
                .collect())
        }
        Value::StringArray(array) if array.data.len() == expected_len => {
            Ok(array.data.iter().map(|s| Some(s.clone())).collect())
        }
        Value::CharArray(chars) if chars.rows == expected_len => Ok((0..chars.rows)
            .map(|row| {
                let start = row * chars.cols;
                let end = start + chars.cols;
                Some(chars.data[start..end].iter().collect::<String>().trim().to_string())
            })
            .collect()),
        Value::Cell(cell) => group_labels_from_cell(cell, expected_len),
        Value::Object(object) if object.is_class(runmat_types::standard::CATEGORICAL) => (0..expected_len)
            .map(|row| {
                crate::builtins::table::categorical_label_at(object, row)
                    .map(Some)
                    .ok_or_else(|| invalid("scatterhist: invalid categorical Group value"))
            })
            .collect(),
        _ => Err(invalid(
            "scatterhist: Group must have one numeric, text, cell, or categorical label per observation",
        )),
    }
}

pub(super) fn group_labels_from_cell(
    cell: &CellArray,
    expected_len: usize,
) -> BuiltinResult<Vec<Option<String>>> {
    if cell.data.len() != expected_len {
        return Err(invalid(
            "scatterhist: Group must have one label per observation",
        ));
    }
    cell.data
        .iter()
        .map(|value| match value {
            Value::Num(value) => Ok(value.is_finite().then(|| number_label(*value))),
            Value::Int(value) => Ok(Some(integer_label(value))),
            Value::String(value) => Ok(Some(value.clone())),
            Value::CharArray(chars) => Ok(Some(chars.data.iter().collect())),
            Value::StringArray(array) if array.data.len() == 1 => Ok(Some(array.data[0].clone())),
            _ => Err(invalid("scatterhist: cell Group labels must be scalar")),
        })
        .collect()
}

pub(super) fn number_label(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{value:.0}")
    } else {
        value.to_string()
    }
}

pub(super) fn integer_label(value: &IntValue) -> String {
    match value {
        IntValue::I8(value) => value.to_string(),
        IntValue::I16(value) => value.to_string(),
        IntValue::I32(value) => value.to_string(),
        IntValue::I64(value) => value.to_string(),
        IntValue::U8(value) => value.to_string(),
        IntValue::U16(value) => value.to_string(),
        IntValue::U32(value) => value.to_string(),
        IntValue::U64(value) => value.to_string(),
    }
}

pub(super) fn integer_values(tensor: &Tensor) -> Option<Vec<IntValue>> {
    tensor
        .integer_storage()
        .map(|storage| storage.exact_values())
}

pub(super) fn tensor_values_f64(tensor: &Tensor) -> Vec<f64> {
    tensor::tensor_values_f64(tensor)
}
