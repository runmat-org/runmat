//! Typed group labels, exact numeric identity, and multifactor decoding.
use super::super::*;
use crate::builtins::common::grouping_keys::KeyAtom;

mod builder;
#[cfg(test)]
mod tests;
mod text_copy;
use builder::{GroupLabelBuilder, LabelProgress};

pub(super) type GroupLabel = Vec<KeyAtom>;

#[derive(Debug)]
pub(super) enum CooperativeLabelError {
    Semantic(RuntimeError),
    Service(RuntimeError),
}

impl CooperativeLabelError {
    pub(super) fn into_runtime(self) -> RuntimeError {
        match self {
            Self::Semantic(error) | Self::Service(error) => error,
        }
    }
}

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

#[cfg(test)]
pub(super) fn group_labels(
    value: &Value,
    expected_len: usize,
) -> BuiltinResult<Vec<Option<GroupLabel>>> {
    let mut builder = GroupLabelBuilder::new(value, expected_len)?;
    loop {
        match builder.step()? {
            LabelProgress::Work(_) => {}
            LabelProgress::Complete(labels) => return Ok(labels),
        }
    }
}

pub(super) async fn group_labels_cooperative(
    value: &Value,
    expected_len: usize,
    cooperation: &mut super::super::cooperation::BoxplotCooperation,
) -> Result<Vec<Option<GroupLabel>>, CooperativeLabelError> {
    cooperation
        .completed(0)
        .await
        .map_err(CooperativeLabelError::Service)?;
    let mut builder =
        GroupLabelBuilder::new(value, expected_len).map_err(CooperativeLabelError::Semantic)?;
    loop {
        match builder.step().map_err(CooperativeLabelError::Semantic)? {
            LabelProgress::Work(units) => cooperation
                .completed(units)
                .await
                .map_err(CooperativeLabelError::Service)?,
            LabelProgress::Complete(labels) => return Ok(labels),
        }
    }
}

fn scalar_group_label(value: &Value) -> BuiltinResult<Option<GroupLabel>> {
    match value {
        Value::String(value) if value.is_missing() => Ok(None),
        Value::StringArray(value) if value.data().len() == 1 && value.data()[0].is_missing() => {
            Ok(None)
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

pub(in crate::builtins::stats::summary::boxplot) fn char_rows(
    array: &CharArray,
) -> BuiltinResult<Vec<String>> {
    (0..array.rows.max(1))
        .map(|row| char_row(array, row))
        .collect()
}

fn char_row(array: &CharArray, row: usize) -> BuiltinResult<String> {
    let units = if array.rows <= 1 {
        array.data()
    } else {
        let start = row * array.cols;
        &array.data()[start..start + array.cols]
    };
    crate::builtins::stats::decode_utf16_with_error(units, || {
        invalid("boxplot: group label is not valid UTF-16 text")
    })
    .map(|text| text.trim_end().to_string())
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
