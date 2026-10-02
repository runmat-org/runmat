//! Typed group labels, exact numeric identity, and multifactor decoding.
use super::super::*;
use crate::builtins::common::grouping_keys::KeyAtom;

pub(super) type GroupLabel = Vec<KeyAtom>;

pub(in crate::builtins::stats::summary::boxplot) fn group_label_text(label: &[KeyAtom]) -> String {
    label
        .iter()
        .map(KeyAtom::label)
        .collect::<Vec<_>>()
        .join(",")
}

fn numeric_group_label(value: NumericScalar) -> Option<GroupLabel> {
    let key = KeyAtom::from_numeric(value)?;
    match key {
        KeyAtom::Number(value) if !value.is_finite() => None,
        key => Some(vec![key]),
    }
}

pub(super) fn group_labels(
    value: &Value,
    expected_len: usize,
) -> BuiltinResult<Vec<Option<GroupLabel>>> {
    match value {
        Value::Cell(cell) => group_labels_from_cell(cell, expected_len),
        Value::Object(object) if object.is_class(runmat_types::standard::CATEGORICAL) => {
            let labels =
                crate::builtins::table::categorical_observation_labels(object).map_err(|err| {
                    invalid_with_source("boxplot: invalid categorical grouping variable", err)
                })?;
            if labels.len() != expected_len {
                return Err(invalid("boxplot: grouping variable length must match X"));
            }
            Ok(labels
                .into_iter()
                .map(|label| {
                    label
                        .filter(|text| missing_text_label(text))
                        .map(|text| vec![KeyAtom::Text(text)])
                })
                .collect())
        }
        Value::String(text) if expected_len == 1 => {
            Ok(vec![Some(vec![KeyAtom::Text(text.clone())])])
        }
        Value::StringArray(array) if array.data.len() == expected_len => Ok(array
            .data
            .iter()
            .map(|text| missing_text_label(text).then(|| vec![KeyAtom::Text(text.clone())]))
            .collect()),
        Value::LogicalArray(array) if array.len() == expected_len => Ok(array
            .data
            .iter()
            .map(|value| Some(vec![KeyAtom::Logical(*value != 0)]))
            .collect()),
        Value::CharArray(array) if array.rows == expected_len => Ok(char_rows(array)
            .into_iter()
            .map(|text| missing_text_label(&text).then_some(vec![KeyAtom::Text(text)]))
            .collect()),
        Value::CharArray(array) if array.rows == 1 && expected_len == 1 => {
            let text: String = array.data.iter().collect();
            Ok(vec![
                missing_text_label(&text).then_some(vec![KeyAtom::Text(text)])
            ])
        }
        _ => {
            let tensor = tensor::value_into_tensor_for(NAME, value.clone()).map_err(|err| {
                invalid(format!(
                    "boxplot: grouping variable must be numeric, text, cell, or categorical ({err})"
                ))
            })?;
            if tensor::tensor_element_len(&tensor) != expected_len {
                return Err(invalid(
                    "boxplot: grouping variable length must match the number of X elements",
                ));
            }
            Ok((0..expected_len)
                .map(|index| tensor.numeric_value_at(index).and_then(numeric_group_label))
                .collect())
        }
    }
}

pub(super) fn validate_group_like(tensor: &Tensor, value: &Value) -> BuiltinResult<()> {
    match group_labels(value, tensor::tensor_element_len(tensor)) {
        Ok(_) => Ok(()),
        Err(err) if tensor.rows > 1 && tensor.cols > 1 => group_labels(value, tensor.rows)
            .map(|_| ())
            .map_err(|_| err),
        Err(err) => Err(err),
    }
}

fn group_labels_from_cell(
    cell: &CellArray,
    expected_len: usize,
) -> BuiltinResult<Vec<Option<GroupLabel>>> {
    if cell.data.len() == expected_len {
        let scalar_labels = cell
            .data
            .iter()
            .map(scalar_group_label)
            .collect::<BuiltinResult<Vec<_>>>();
        if let Ok(labels) = scalar_labels {
            return Ok(labels);
        }
    }
    let mut variables = Vec::with_capacity(cell.data.len());
    for value in &cell.data {
        let labels = group_labels(value, expected_len)?;
        if labels.len() != expected_len {
            return Err(invalid(
                "boxplot: grouping variables in a cell array must share the X length",
            ));
        }
        variables.push(labels);
    }
    let mut labels = Vec::with_capacity(expected_len);
    for row in 0..expected_len {
        let mut parts = Vec::with_capacity(variables.len());
        let mut missing = false;
        for variable in &variables {
            match &variable[row] {
                Some(label) => parts.extend(label.iter().cloned()),
                None => {
                    missing = true;
                    break;
                }
            }
        }
        labels.push((!missing).then_some(parts));
    }
    Ok(labels)
}

fn scalar_group_label(value: &Value) -> BuiltinResult<Option<GroupLabel>> {
    match value {
        Value::String(text) => {
            Ok(missing_text_label(text).then(|| vec![KeyAtom::Text(text.clone())]))
        }
        Value::StringArray(array) if array.data.len() == 1 => {
            let text = array.data[0].clone();
            Ok(missing_text_label(&text).then_some(vec![KeyAtom::Text(text)]))
        }
        Value::CharArray(array) if array.rows == 1 => {
            let text: String = array.data.iter().collect();
            Ok(missing_text_label(&text).then_some(vec![KeyAtom::Text(text)]))
        }
        Value::Num(value) => Ok(value.is_finite().then_some(vec![KeyAtom::Number(*value)])),
        Value::Int(value) => Ok(Some(vec![KeyAtom::Integer(value.clone())])),
        Value::Bool(value) => Ok(Some(vec![KeyAtom::Logical(*value)])),
        Value::LogicalArray(array) if array.len() == 1 => {
            Ok(Some(vec![KeyAtom::Logical(array.data[0] != 0)]))
        }
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            Ok(tensor.numeric_value_at(0).and_then(numeric_group_label))
        }
        _ => Err(invalid("boxplot: cell grouping labels must be scalar")),
    }
}

fn missing_text_label(text: &str) -> bool {
    let trimmed = text.trim();
    !trimmed.is_empty()
        && trimmed != "<missing>"
        && trimmed != "<undefined>"
        && !trimmed.eq_ignore_ascii_case("missing")
}

pub(in crate::builtins::stats::summary::boxplot) fn char_rows(array: &CharArray) -> Vec<String> {
    if array.rows <= 1 {
        return vec![array.data.iter().collect()];
    }
    (0..array.rows)
        .map(|row| {
            (0..array.cols)
                .filter_map(|col| array.data.get(row * array.cols + col).copied())
                .collect::<String>()
                .trim_end()
                .to_string()
        })
        .collect()
}

pub(in crate::builtins::stats::summary::boxplot) fn number_label(value: &f64) -> String {
    if value.fract().abs() < f64::EPSILON {
        format!("{value:.0}")
    } else {
        let text = format!("{value:.15}");
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

pub(in crate::builtins::stats::summary::boxplot) fn numeric_scalar_label(
    value: NumericScalar,
) -> String {
    match value {
        NumericScalar::F64(value) => number_label(&value),
        NumericScalar::F32(value) => number_label(&f64::from(value)),
        NumericScalar::I8(value) => value.to_string(),
        NumericScalar::I16(value) => value.to_string(),
        NumericScalar::I32(value) => value.to_string(),
        NumericScalar::I64(value) => value.to_string(),
        NumericScalar::U8(value) => value.to_string(),
        NumericScalar::U16(value) => value.to_string(),
        NumericScalar::U32(value) => value.to_string(),
        NumericScalar::U64(value) => value.to_string(),
    }
}
