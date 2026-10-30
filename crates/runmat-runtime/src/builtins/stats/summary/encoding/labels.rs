//! Label construction, exact class identity, and class-order validation.
use super::*;
use runmat_value::{decode_utf16_code_units, Utf16DecodeError};

pub(super) fn labels_from_value(name: EncodingBuiltin, value: Value) -> BuiltinResult<LabelArray> {
    match value {
        Value::Num(value) => Ok(LabelArray {
            labels: vec![numeric_label(NumericScalar::F64(value))],
            shape: vec![1, 1],
            kind: LabelKind::Numeric,
            categories: None,
        }),
        Value::Int(value) => Ok(LabelArray {
            labels: vec![numeric_label(NumericScalar::from(value))],
            shape: vec![1, 1],
            kind: LabelKind::Numeric,
            categories: None,
        }),
        Value::Tensor(tensor) => Ok(LabelArray {
            labels: (0..tensor.len())
                .map(|index| tensor.numeric_value_at(index).and_then(numeric_label))
                .collect(),
            shape: normalized_shape(tensor.shape()),
            kind: LabelKind::Numeric,
            categories: None,
        }),
        Value::Bool(value) => Ok(LabelArray {
            labels: vec![Some(Label::Logical(value))],
            shape: vec![1, 1],
            kind: LabelKind::Logical,
            categories: None,
        }),
        Value::LogicalArray(array) => Ok(LabelArray {
            labels: array
                .data
                .iter()
                .map(|value| Some(Label::Logical(*value != 0)))
                .collect(),
            shape: normalized_shape(&array.shape),
            kind: LabelKind::Logical,
            categories: None,
        }),
        Value::String(text) => Ok(LabelArray {
            labels: vec![text.as_str().and_then(|text| text_label(text.to_owned()))],
            shape: vec![1, 1],
            kind: LabelKind::String,
            categories: None,
        }),
        Value::StringArray(array) => Ok(LabelArray {
            labels: array
                .data()
                .iter()
                .map(|text| text.as_str().and_then(|text| text_label(text.to_owned())))
                .collect(),
            shape: normalized_shape(&array.shape),
            kind: LabelKind::String,
            categories: None,
        }),
        Value::CharArray(array) => Ok(LabelArray {
            labels: char_rows(&array, name)?
                .into_iter()
                .map(text_label)
                .collect(),
            shape: vec![array.rows, 1],
            kind: LabelKind::Char,
            categories: None,
        }),
        Value::Cell(cell) => {
            let mut labels = Vec::with_capacity(cell.data().len());
            for item in cell.data().iter() {
                labels.push(match item {
                    Value::String(text) if text.is_missing() => None,
                    Value::StringArray(array)
                        if array.data().len() == 1 && array.data()[0].is_missing() =>
                    {
                        None
                    }
                    _ => text_label(scalar_text(item, name, name.name())?),
                });
            }
            Ok(LabelArray {
                labels,
                shape: vec![cell.rows, cell.cols],
                kind: LabelKind::Cell,
                categories: None,
            })
        }
        Value::Object(object) if object.is_class(runmat_types::standard::CATEGORICAL) => {
            let labels =
                crate::builtins::table::categorical_labels(&Value::Object(object.clone()))?
                    .into_iter()
                    .map(|label| label.as_str().and_then(|text| text_label(text.to_owned())))
                    .collect();
            let shape = categorical_shape(&object);
            Ok(LabelArray {
                labels,
                shape,
                kind: LabelKind::Categorical,
                categories: Some(crate::builtins::table::categorical_categories(&object)?),
            })
        }
        other => Err(invalid(
            name,
            format!("{name}: unsupported label input {other:?}"),
        )),
    }
}

pub(super) fn dummyvar_groups(value: Value) -> BuiltinResult<Vec<LabelArray>> {
    match value {
        Value::Tensor(tensor)
            if tensor.shape().len() <= 2 && tensor.rows() > 1 && tensor.cols() > 1 =>
        {
            let rows = tensor.rows();
            let cols = tensor.cols();
            let values = tensor::tensor_values_f64_cow(&tensor);
            let mut groups = Vec::with_capacity(cols);
            for col in 0..cols {
                let mut labels = Vec::with_capacity(rows);
                for row in 0..rows {
                    labels.push(numeric_label(NumericScalar::F64(values[row + col * rows])));
                }
                groups.push(LabelArray {
                    labels,
                    shape: vec![rows, 1],
                    kind: LabelKind::Numeric,
                    categories: None,
                });
            }
            Ok(groups)
        }
        Value::Cell(cell) if cell.rows == 1 || cell.cols == 1 => {
            if cell.rows == 1 && cell.cols > 1 && cell.data().iter().any(is_grouping_container) {
                let mut groups = Vec::with_capacity(cell.cols);
                let (values, _shape, _rows, _cols) = cell.into_parts();
                // Keep the original physical storage and its claims throughout
                // the read traversal; cloning each published Value retains its
                // original payload owner without an escaping bare Value Vec.
                for value in values.iter().cloned() {
                    groups.push(labels_from_value(EncodingBuiltin::Dummyvar, value)?);
                }
                Ok(groups)
            } else {
                Ok(vec![labels_from_value(
                    EncodingBuiltin::Dummyvar,
                    Value::Cell(cell),
                )?])
            }
        }
        other => Ok(vec![labels_from_value(EncodingBuiltin::Dummyvar, other)?]),
    }
}

pub(super) fn classes_for_dummyvar(group: &LabelArray, rows: usize) -> BuiltinResult<Vec<Label>> {
    match group.kind {
        LabelKind::Numeric => numeric_dummyvar_classes(&group.labels, rows),
        _ => Ok(classes_for_labels(group)),
    }
}

pub(super) fn numeric_dummyvar_classes(
    labels: &[Option<Label>],
    rows: usize,
) -> BuiltinResult<Vec<Label>> {
    let mut max_level = 0usize;
    for label in labels.iter().flatten() {
        let Label::Numeric(value) = label else {
            continue;
        };
        let Some(value) = positive_numeric_usize(*value) else {
            return Err(invalid(
                EncodingBuiltin::Dummyvar,
                "dummyvar: numeric groups must contain positive integer levels or NaN",
            ));
        };
        max_level = max_level.max(value);
    }
    let row_count = rows.max(1);
    if max_level > MAX_ONEHOT_CELLS / row_count {
        return Err(invalid(
            EncodingBuiltin::Dummyvar,
            "dummyvar: output is too large",
        ));
    }
    Ok((1..=max_level)
        .map(|value| Label::Numeric(NumericScalar::F64(value as f64)))
        .collect())
}

pub(super) fn classes_for_labels(labels: &LabelArray) -> Vec<Label> {
    match labels.kind {
        LabelKind::Numeric => {
            let mut classes = dedupe(labels.labels.iter().flatten().cloned());
            classes.sort_by(|left, right| match (left, right) {
                (Label::Numeric(a), Label::Numeric(b)) => a
                    .materialize_f64()
                    .partial_cmp(&b.materialize_f64())
                    .unwrap_or(Ordering::Equal),
                _ => Ordering::Equal,
            });
            classes
        }
        LabelKind::Logical => vec![Label::Logical(false), Label::Logical(true)],
        LabelKind::Categorical => {
            let mut classes = Vec::new();
            if let Some(categories) = &labels.categories {
                for category in categories {
                    classes.push(Label::Text(category.clone()));
                }
            }
            for label in labels.labels.iter().flatten() {
                if !classes.iter().any(|existing| same_label(existing, label)) {
                    classes.push(label.clone());
                }
            }
            classes
        }
        LabelKind::String | LabelKind::Char | LabelKind::Cell => {
            dedupe(labels.labels.iter().flatten().cloned())
        }
    }
}

pub(super) fn nonmissing_classes(
    name: EncodingBuiltin,
    labels: &LabelArray,
) -> BuiltinResult<Vec<Label>> {
    if labels.labels.iter().any(Option::is_none) {
        return Err(invalid(
            name,
            format!("{name}: class names cannot contain missing values"),
        ));
    }
    let classes = dedupe(labels.labels.iter().flatten().cloned());
    if classes.len() != labels.labels.iter().flatten().count() {
        return Err(invalid(name, format!("{name}: class names must be unique")));
    }
    Ok(classes)
}

pub(super) fn class_index(
    owner: EncodingBuiltin,
    classes: &[Label],
) -> BuiltinResult<HashMap<LabelKey, usize>> {
    let mut index = HashMap::with_capacity(classes.len());
    for (idx, class) in classes.iter().enumerate() {
        if index.insert(label_key(class), idx).is_some() {
            return Err(invalid(owner, format!("{owner}: duplicate class labels")));
        }
    }
    Ok(index)
}

pub(super) fn numeric_label(value: NumericScalar) -> Option<Label> {
    match value {
        NumericScalar::F64(value) if value.is_nan() => None,
        NumericScalar::F32(value) if value.is_nan() => None,
        NumericScalar::F64(0.0) => Some(Label::Numeric(NumericScalar::F64(0.0))),
        NumericScalar::F32(0.0) => Some(Label::Numeric(NumericScalar::F32(0.0))),
        value => Some(Label::Numeric(value)),
    }
}

pub(super) fn text_label(value: String) -> Option<Label> {
    (!value.is_empty()).then_some(Label::Text(value))
}

pub(super) fn same_label(left: &Label, right: &Label) -> bool {
    match (left, right) {
        (Label::Numeric(a), Label::Numeric(b)) => numeric_key(*a) == numeric_key(*b),
        (Label::Logical(a), Label::Logical(b)) => a == b,
        (Label::Text(a), Label::Text(b)) => a == b,
        _ => false,
    }
}

pub(super) fn label_key(label: &Label) -> LabelKey {
    match label {
        Label::Numeric(value) => numeric_key(*value),
        Label::Logical(value) => LabelKey::Logical(*value),
        Label::Text(value) => LabelKey::Text(value.clone()),
    }
}

pub(super) fn numeric_key(value: NumericScalar) -> LabelKey {
    if let Some(value) = value.into_int_value() {
        return integer_key(value);
    }
    let value = match value {
        NumericScalar::F64(value) => value,
        NumericScalar::F32(value) => f64::from(value),
        _ => unreachable!("integer numeric scalar handled above"),
    };
    if value == 0.0 {
        return LabelKey::Integer(0);
    }
    if value.is_finite() && value.fract() == 0.0 {
        if value > 0.0 && value < 18_446_744_073_709_551_616.0 {
            return LabelKey::Integer(i128::from(value as u64));
        }
        if value < 0.0 && value >= i64::MIN as f64 {
            return LabelKey::Integer(i128::from(value as i64));
        }
    }
    LabelKey::FloatingBits(value.to_bits())
}

pub(super) fn integer_key(value: IntValue) -> LabelKey {
    match value {
        IntValue::I8(value) => LabelKey::Integer(i128::from(value)),
        IntValue::I16(value) => LabelKey::Integer(i128::from(value)),
        IntValue::I32(value) => LabelKey::Integer(i128::from(value)),
        IntValue::I64(value) => LabelKey::Integer(i128::from(value)),
        IntValue::U8(value) => LabelKey::Integer(i128::from(value)),
        IntValue::U16(value) => LabelKey::Integer(i128::from(value)),
        IntValue::U32(value) => LabelKey::Integer(i128::from(value)),
        IntValue::U64(value) => LabelKey::Integer(i128::from(value)),
    }
}

pub(super) fn positive_numeric_usize(value: NumericScalar) -> Option<usize> {
    match value {
        NumericScalar::F64(value) => positive_float_usize(value),
        NumericScalar::F32(value) => positive_float_usize(f64::from(value)),
        value => value
            .into_int_value()?
            .try_to_usize()
            .filter(|value| *value > 0),
    }
}

pub(super) fn positive_float_usize(value: f64) -> Option<usize> {
    if !value.is_finite() || value <= 0.0 || value.fract() != 0.0 {
        return None;
    }
    if value > usize::MAX as f64 || (usize::BITS == 64 && value == usize::MAX as f64) {
        return None;
    }
    Some(value as usize)
}

pub(super) fn dedupe<I>(labels: I) -> Vec<Label>
where
    I: IntoIterator<Item = Label>,
{
    let mut out = Vec::new();
    for label in labels {
        if !out.iter().any(|existing| same_label(existing, &label)) {
            out.push(label);
        }
    }
    out
}

pub(super) fn text_values(labels: &[Label]) -> Vec<String> {
    labels
        .iter()
        .map(|label| match label {
            Label::Numeric(value) => numeric_label_text(*value),
            Label::Logical(value) => {
                if *value {
                    "true".to_string()
                } else {
                    "false".to_string()
                }
            }
            Label::Text(value) => value.clone(),
        })
        .collect()
}

pub(super) fn numeric_label_text(value: NumericScalar) -> String {
    match value {
        NumericScalar::F64(value) => format_number(value),
        NumericScalar::F32(value) => format_number(f64::from(value)),
        value => value
            .into_int_value()
            .map(|value| value.decimal_string())
            .unwrap_or_default(),
    }
}

pub(super) fn dedupe_text(values: Vec<String>) -> Vec<String> {
    let mut out = Vec::new();
    for value in values {
        if !out.iter().any(|existing| existing == &value) {
            out.push(value);
        }
    }
    out
}

pub(super) fn label_kinds_compatible(left: LabelKind, right: LabelKind) -> bool {
    left == right || (is_text_kind(left) && is_text_kind(right))
}

pub(super) fn is_grouping_container(value: &Value) -> bool {
    match value {
        Value::Tensor(tensor) => tensor::tensor_element_len(tensor) > 1,
        Value::LogicalArray(array) => array.data.len() > 1,
        Value::StringArray(array) => array.data().len() > 1,
        Value::CharArray(array) => array.rows > 1,
        Value::Cell(cell) => cell.data().len() > 1,
        Value::Object(object) if object.is_class(runmat_types::standard::CATEGORICAL) => {
            matches!(
                object.properties().get("Codes"),
                Some(Value::Tensor(tensor)) if tensor::tensor_element_len(tensor) > 1
            )
        }
        _ => false,
    }
}

pub(super) fn is_text_kind(kind: LabelKind) -> bool {
    matches!(
        kind,
        LabelKind::String | LabelKind::Char | LabelKind::Cell | LabelKind::Categorical
    )
}

pub(super) fn char_rows(array: &CharArray, builtin: EncodingBuiltin) -> BuiltinResult<Vec<String>> {
    let mut rows = Vec::with_capacity(array.rows);
    for row in 0..array.rows {
        let start = row * array.cols;
        rows.push(
            decode_utf16_code_units(&array.data()[start..start + array.cols])
                .map_err(|error| invalid_utf16(builtin, "character label row", error))?,
        );
    }
    Ok(rows)
}

pub(super) fn scalar_text(
    value: &Value,
    builtin: EncodingBuiltin,
    context: &str,
) -> BuiltinResult<String> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => Ok(text.as_str().to_owned()),
        Value::CharArray(chars) => decode_utf16_code_units(chars.data())
            .map_err(|error| invalid_utf16(builtin, context, error)),
        Value::StringArray(array) if array.data().len() == 1 => {
            array.data()[0].as_str().map(str::to_owned).ok_or_else(|| {
                invalid(
                    builtin,
                    format!("{context}: expected string scalar, got {value:?}"),
                )
            })
        }
        other => Err(invalid(
            builtin,
            format!("{context}: expected string scalar, got {other:?}"),
        )),
    }
}

fn invalid_utf16(builtin: EncodingBuiltin, context: &str, error: Utf16DecodeError) -> RuntimeError {
    let mut builder = build_runtime_error(format!(
        "{context}: character value is not valid UTF-16 text: {error}"
    ))
    .with_builtin(builtin.name())
    .with_source(error);
    if let Some(identifier) = builtin.invalid().identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(super) fn canonical(text: &str) -> String {
    text.chars()
        .filter(|ch| *ch != '_' && *ch != '-' && !ch.is_whitespace())
        .collect::<String>()
        .to_ascii_lowercase()
}

pub(super) fn format_number(value: f64) -> String {
    if value.fract() == 0.0 && value.is_finite() {
        format!("{value:.0}")
    } else {
        value.to_string()
    }
}

pub(super) fn categorical_shape(object: &ObjectInstance) -> Vec<usize> {
    match object.properties().get("Codes") {
        Some(Value::Tensor(tensor)) => normalized_shape(tensor.shape()),
        _ => vec![1, 1],
    }
}
