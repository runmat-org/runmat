use super::*;
use runmat_value::{CellArray, CharArray, LogicalArray, NumericStorage, StringArray};
use std::cmp::Ordering;

pub(super) fn labels_from_value(value: Value, name: &str) -> BuiltinResult<LabelVector> {
    match value {
        Value::Tensor(tensor) => {
            let dtype = tensor.numeric_dtype();
            Ok(LabelVector {
                labels: vector_values(&tensor, name)?
                    .into_iter()
                    .map(ClassLabel::Numeric)
                    .collect(),
                kind: LabelKind::Numeric(dtype),
            })
        }
        Value::Num(value) => Ok(LabelVector {
            labels: vec![ClassLabel::Numeric(NumericScalar::F64(value))],
            kind: LabelKind::Numeric(NumericDType::F64),
        }),
        Value::Int(value) => {
            let value = NumericScalar::from(value);
            Ok(LabelVector {
                labels: vec![ClassLabel::Numeric(value)],
                kind: LabelKind::Numeric(value.numeric_dtype()),
            })
        }
        Value::String(text) => Ok(LabelVector {
            labels: vec![ClassLabel::Text(text)],
            kind: LabelKind::String,
        }),
        Value::CharArray(array) => Ok(LabelVector {
            labels: char_rows(&array)?
                .into_iter()
                .map(|text| ClassLabel::Text(text.into()))
                .collect(),
            kind: LabelKind::Char,
        }),
        Value::StringArray(array) => {
            let (data, _shape, _rows, _cols) = array.into_parts();
            let labels = match data.try_into_owned() {
                Ok(values) => values.into_iter().map(ClassLabel::Text).collect(),
                Err(values) => values.iter().cloned().map(ClassLabel::Text).collect(),
            };
            Ok(LabelVector {
                labels,
                kind: LabelKind::String,
            })
        },
        Value::Cell(cell) => {
            let mut labels = Vec::with_capacity(cell.data().len());
            for item in cell.data() {
                labels.push(ClassLabel::Text(match item {
                    Value::String(text) => text.clone(),
                    Value::StringArray(array) if array.data().len() == 1 => array.data()[0].clone(),
                    _ => scalar_text(item, name)?.into(),
                }));
            }
            Ok(LabelVector {
                labels,
                kind: LabelKind::Cell,
            })
        }
        Value::Bool(value) => Ok(LabelVector {
            labels: vec![ClassLabel::Logical(value)],
            kind: LabelKind::Logical,
        }),
        Value::LogicalArray(array) => Ok(LabelVector {
            labels: array
                .data
                .into_iter()
                .map(|value| ClassLabel::Logical(value != 0))
                .collect(),
            kind: LabelKind::Logical,
        }),
        other => Err(invalid_argument(format!(
            "classify: {name} must be numeric, logical, string, character, or cellstr labels; got {other:?}"
        ))),
    }
}

pub(super) fn vector_values(tensor: &Tensor, name: &str) -> BuiltinResult<Vec<NumericScalar>> {
    if tensor.shape().iter().filter(|dim| **dim > 1).count() > 1 {
        return Err(invalid_argument(format!(
            "classify: {name} must be a vector"
        )));
    }
    (0..tensor.len())
        .map(|index| {
            tensor.numeric_value_at(index).ok_or_else(|| {
                invalid_argument(format!("classify: {name} numeric storage is invalid"))
            })
        })
        .collect()
}

pub(super) fn numeric_vector(value: &Value, name: &str) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Tensor(tensor) => {
            if tensor.shape().iter().filter(|dim| **dim > 1).count() > 1 {
                return Err(invalid_argument(format!(
                    "classify: {name} must be a vector"
                )));
            }
            ensure_exact_integer_tensor_boundary(tensor, name)?;
            Ok(tensor::tensor_values_f64(tensor))
        }
        Value::Num(value) => Ok(vec![*value]),
        Value::Int(value) => {
            ensure_exact_integer_scalar_boundary(NumericScalar::from(value.clone()), name)?;
            Ok(vec![value.to_f64()])
        }
        other => Err(invalid_argument(format!(
            "classify: {name} must be a numeric vector, got {other:?}"
        ))),
    }
}

pub(super) fn ensure_exact_integer_tensor_boundary(
    tensor: &Tensor,
    name: &str,
) -> BuiltinResult<()> {
    if tensor.integer_storage().is_none() {
        return Ok(());
    }
    for index in 0..tensor.len() {
        if let Some(value) = tensor.numeric_value_at(index) {
            ensure_exact_integer_scalar_boundary(value, name)?;
        }
    }
    Ok(())
}

pub(super) fn ensure_exact_integer_scalar_boundary(
    value: NumericScalar,
    name: &str,
) -> BuiltinResult<()> {
    if value
        .into_int_value()
        .is_some_and(|integer| !crate::builtins::common::validation::integer_is_exact_f64(&integer))
    {
        return Err(invalid_argument(format!(
            "classify: {name} integer values must be exactly representable as double"
        )));
    }
    Ok(())
}

pub(super) fn unique_labels(labels: &[ClassLabel], kind: LabelKind) -> Vec<ClassLabel> {
    let mut out = Vec::new();
    for label in labels {
        if is_missing_label(label) {
            continue;
        }
        if !out.iter().any(|existing| same_label(existing, label)) {
            out.push(label.clone());
        }
    }
    match kind {
        LabelKind::Numeric(_) => out.sort_by(|left, right| match (left, right) {
            (ClassLabel::Numeric(a), ClassLabel::Numeric(b)) => {
                numeric_label_cmp(*a, *b).unwrap_or(Ordering::Equal)
            }
            _ => Ordering::Equal,
        }),
        LabelKind::String | LabelKind::Char | LabelKind::Cell => {
            out.sort_by(|left, right| match (left, right) {
                (ClassLabel::Text(a), ClassLabel::Text(b)) => a.cmp(b),
                _ => Ordering::Equal,
            })
        }
        LabelKind::Logical => out.sort_by_key(|label| match label {
            ClassLabel::Logical(value) => *value,
            _ => false,
        }),
    }
    out
}

pub(super) fn numeric_label_cmp(left: NumericScalar, right: NumericScalar) -> Option<Ordering> {
    match (left, right) {
        (NumericScalar::F64(left), NumericScalar::F64(right)) => left.partial_cmp(&right),
        (NumericScalar::F32(left), NumericScalar::F32(right)) => left.partial_cmp(&right),
        (NumericScalar::I8(left), NumericScalar::I8(right)) => Some(left.cmp(&right)),
        (NumericScalar::I16(left), NumericScalar::I16(right)) => Some(left.cmp(&right)),
        (NumericScalar::I32(left), NumericScalar::I32(right)) => Some(left.cmp(&right)),
        (NumericScalar::I64(left), NumericScalar::I64(right)) => Some(left.cmp(&right)),
        (NumericScalar::U8(left), NumericScalar::U8(right)) => Some(left.cmp(&right)),
        (NumericScalar::U16(left), NumericScalar::U16(right)) => Some(left.cmp(&right)),
        (NumericScalar::U32(left), NumericScalar::U32(right)) => Some(left.cmp(&right)),
        (NumericScalar::U64(left), NumericScalar::U64(right)) => Some(left.cmp(&right)),
        _ => None,
    }
}

pub(super) fn numeric_label_is_nan(value: NumericScalar) -> bool {
    match value {
        NumericScalar::F64(value) => value.is_nan(),
        NumericScalar::F32(value) => value.is_nan(),
        NumericScalar::I8(_)
        | NumericScalar::I16(_)
        | NumericScalar::I32(_)
        | NumericScalar::I64(_)
        | NumericScalar::U8(_)
        | NumericScalar::U16(_)
        | NumericScalar::U32(_)
        | NumericScalar::U64(_) => false,
    }
}

pub(super) fn same_label(left: &ClassLabel, right: &ClassLabel) -> bool {
    match (left, right) {
        (ClassLabel::Numeric(a), ClassLabel::Numeric(b)) => {
            (a == b) || (numeric_label_is_nan(*a) && numeric_label_is_nan(*b))
        }
        (ClassLabel::Text(a), ClassLabel::Text(b)) => a == b,
        (ClassLabel::Logical(a), ClassLabel::Logical(b)) => a == b,
        _ => false,
    }
}

pub(super) fn is_missing_label(label: &ClassLabel) -> bool {
    match label {
        ClassLabel::Numeric(value) => numeric_label_is_nan(*value),
        ClassLabel::Text(value) => value.is_missing() || value.as_str().is_some_and(str::is_empty),
        ClassLabel::Logical(_) => false,
    }
}

pub(super) fn label_kinds_compatible(left: LabelKind, right: LabelKind) -> bool {
    left == right || (is_text_kind(left) && is_text_kind(right))
}

pub(super) fn is_text_kind(kind: LabelKind) -> bool {
    matches!(kind, LabelKind::String | LabelKind::Char | LabelKind::Cell)
}

pub(super) fn labels_value(
    classes: &[ClassStats],
    kind: LabelKind,
    indices: &[usize],
) -> BuiltinResult<Value> {
    let labels = classes
        .iter()
        .map(|class| class.label.clone())
        .collect::<Vec<_>>();
    labels_from_indices(&labels, kind, indices)
}

pub(super) fn labels_from_indices(
    labels: &[ClassLabel],
    kind: LabelKind,
    indices: &[usize],
) -> BuiltinResult<Value> {
    match kind {
        LabelKind::Numeric(dtype) => {
            let values = indices
                .iter()
                .map(|idx| match labels.get(*idx) {
                    Some(ClassLabel::Numeric(value)) => Ok(*value),
                    _ => Err(internal_error("classify: invalid numeric label index")),
                })
                .collect::<BuiltinResult<Vec<_>>>()?;
            let mut storage = NumericStorage::zeros(dtype, values.len());
            for (index, value) in values.into_iter().enumerate() {
                storage
                    .set_value(index, value)
                    .map_err(|err| internal_error(format!("classify: {err}")))?;
            }
            Ok(Value::Tensor(
                Tensor::from_numeric_storage(storage, vec![indices.len(), 1])
                    .map_err(|err| internal_error(format!("classify: {err}")))?,
            ))
        }
        LabelKind::String => {
            let data = indices
                .iter()
                .map(|idx| match labels.get(*idx) {
                    Some(ClassLabel::Text(value)) => value.clone(),
                    _ => runmat_value::StringElement::default(),
                })
                .collect::<Vec<_>>();
            Ok(Value::StringArray(
                StringArray::from_elements_unbounded(data, vec![indices.len(), 1])
                    .map_err(|err| internal_error(format!("classify: {err}")))?,
            ))
        }
        LabelKind::Char => {
            let rows = indices
                .iter()
                .map(|idx| match labels.get(*idx) {
                    Some(ClassLabel::Text(value)) => value
                        .as_str()
                        .expect("missing labels excluded before character publication"),
                    _ => "",
                })
                .collect::<Vec<_>>();
            Ok(Value::CharArray(char_array_from_rows(&rows)?))
        }
        LabelKind::Cell => {
            let data = indices
                .iter()
                .map(|idx| match labels.get(*idx) {
                    Some(ClassLabel::Text(value)) => Value::CharArray(CharArray::new_row(
                        value
                            .as_str()
                            .expect("missing labels excluded before character publication"),
                    )),
                    _ => Value::CharArray(CharArray::new_row("")),
                })
                .collect::<Vec<_>>();
            Ok(Value::Cell(
                CellArray::new(data, indices.len(), 1)
                    .map_err(|err| internal_error(format!("classify: {err}")))?,
            ))
        }
        LabelKind::Logical => {
            let data = indices
                .iter()
                .map(|idx| match labels.get(*idx) {
                    Some(ClassLabel::Logical(value)) if *value => 1,
                    _ => 0,
                })
                .collect::<Vec<_>>();
            Ok(Value::LogicalArray(
                LogicalArray::new(data, vec![indices.len(), 1])
                    .map_err(|err| internal_error(format!("classify: {err}")))?,
            ))
        }
    }
}

pub(super) fn char_rows(array: &CharArray) -> BuiltinResult<Vec<String>> {
    let mut rows = Vec::with_capacity(array.rows);
    for row in 0..array.rows {
        let start = row * array.cols;
        rows.push(
            runmat_value::decode_utf16_code_units(&array.data()[start..start + array.cols])
                .map_err(|source| internal_error(format!("classify: {source}")))?,
        );
    }
    Ok(rows)
}

pub(super) fn char_array_from_rows(rows: &[impl AsRef<str>]) -> BuiltinResult<CharArray> {
    let width = rows
        .iter()
        .map(|row| row.as_ref().encode_utf16().count())
        .max()
        .unwrap_or(0);
    let mut data = Vec::with_capacity(rows.len() * width);
    for row in rows {
        let units = row.as_ref().encode_utf16().count();
        data.extend(row.as_ref().chars());
        data.extend(std::iter::repeat_n(' ', width - units));
    }
    CharArray::from_unicode_scalars(data, rows.len(), width)
        .map_err(|err| internal_error(format!("classify: {err}")))
}

pub(super) fn scalar_text(value: &Value, name: &str) -> BuiltinResult<runmat_value::Utf8Text> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => Ok(text.clone()),
        Value::CharArray(chars) => runmat_value::decode_utf16_code_units(chars.data())
            .map(Into::into)
            .map_err(|source| invalid_argument(format!("classify: {source}"))),
        Value::StringArray(array) if array.data().len() == 1 => array.data()[0]
            .text_ref()
            .cloned()
            .ok_or_else(|| invalid_argument(format!("classify: {name} must be a string scalar"))),
        other => Err(invalid_argument(format!(
            "classify: {name} must be a string scalar, got {other:?}"
        ))),
    }
}

pub(super) fn canonical(text: &str) -> String {
    text.chars()
        .filter(|ch| *ch != '_' && *ch != '-' && !ch.is_whitespace())
        .collect::<String>()
        .to_ascii_lowercase()
}
