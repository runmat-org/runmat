//! Typed group labels, exact numeric identity, and multifactor decoding.
use super::super::*;
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(super) enum GroupLabel {
    Floating(OrderedF64),
    Integer(i128),
    Logical(bool),
    Text(String),
    Tuple(Vec<GroupLabel>),
}

impl GroupLabel {
    pub(super) fn text(&self) -> String {
        match self {
            Self::Floating(value) => number_label(&value.0),
            Self::Integer(value) => value.to_string(),
            Self::Logical(value) => value.to_string(),
            Self::Text(value) => value.clone(),
            Self::Tuple(parts) => parts.iter().map(Self::text).collect::<Vec<_>>().join(","),
        }
    }
}

pub(in crate::builtins::stats::summary::boxplot) fn int_group_key(value: &IntValue) -> i128 {
    match value {
        IntValue::I8(value) => i128::from(*value),
        IntValue::I16(value) => i128::from(*value),
        IntValue::I32(value) => i128::from(*value),
        IntValue::I64(value) => i128::from(*value),
        IntValue::U8(value) => i128::from(*value),
        IntValue::U16(value) => i128::from(*value),
        IntValue::U32(value) => i128::from(*value),
        IntValue::U64(value) => i128::from(*value),
    }
}

fn numeric_group_label(value: NumericScalar) -> Option<GroupLabel> {
    match value {
        NumericScalar::F64(value) => value
            .is_finite()
            .then_some(GroupLabel::Floating(OrderedF64(value))),
        NumericScalar::F32(value) => value
            .is_finite()
            .then_some(GroupLabel::Floating(OrderedF64(f64::from(value)))),
        NumericScalar::I8(value) => Some(GroupLabel::Integer(i128::from(value))),
        NumericScalar::I16(value) => Some(GroupLabel::Integer(i128::from(value))),
        NumericScalar::I32(value) => Some(GroupLabel::Integer(i128::from(value))),
        NumericScalar::I64(value) => Some(GroupLabel::Integer(i128::from(value))),
        NumericScalar::U8(value) => Some(GroupLabel::Integer(i128::from(value))),
        NumericScalar::U16(value) => Some(GroupLabel::Integer(i128::from(value))),
        NumericScalar::U32(value) => Some(GroupLabel::Integer(i128::from(value))),
        NumericScalar::U64(value) => Some(GroupLabel::Integer(i128::from(value))),
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
                        .map(GroupLabel::Text)
                })
                .collect())
        }
        Value::String(text) if expected_len == 1 => Ok(vec![Some(GroupLabel::Text(text.clone()))]),
        Value::StringArray(array) if array.data.len() == expected_len => Ok(array
            .data
            .iter()
            .map(|text| missing_text_label(text).then(|| GroupLabel::Text(text.clone())))
            .collect()),
        Value::CharArray(array) if array.rows == expected_len => Ok(char_rows(array)
            .into_iter()
            .map(|text| missing_text_label(&text).then_some(GroupLabel::Text(text)))
            .collect()),
        Value::CharArray(array) if array.rows == 1 && expected_len == 1 => {
            let text: String = array.data.iter().collect();
            Ok(vec![
                missing_text_label(&text).then_some(GroupLabel::Text(text))
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
                Some(label) => parts.push(label.clone()),
                None => {
                    missing = true;
                    break;
                }
            }
        }
        labels.push((!missing).then(|| GroupLabel::Tuple(parts)));
    }
    Ok(labels)
}

fn scalar_group_label(value: &Value) -> BuiltinResult<Option<GroupLabel>> {
    match value {
        Value::String(text) => Ok(missing_text_label(text).then(|| GroupLabel::Text(text.clone()))),
        Value::StringArray(array) if array.data.len() == 1 => {
            let text = array.data[0].clone();
            Ok(missing_text_label(&text).then_some(GroupLabel::Text(text)))
        }
        Value::CharArray(array) if array.rows == 1 => {
            let text: String = array.data.iter().collect();
            Ok(missing_text_label(&text).then_some(GroupLabel::Text(text)))
        }
        Value::Num(value) => Ok(value
            .is_finite()
            .then_some(GroupLabel::Floating(OrderedF64(*value)))),
        Value::Int(value) => Ok(Some(GroupLabel::Integer(int_group_key(value)))),
        Value::Bool(value) => Ok(Some(GroupLabel::Logical(*value))),
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct OrderedF64(pub(super) f64);

impl Eq for OrderedF64 {}

impl PartialOrd for OrderedF64 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}
