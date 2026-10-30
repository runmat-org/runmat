use super::*;
use runmat_value::{decode_utf16_code_units, Utf16DecodeError};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum LabelKind {
    Numeric,
    String,
    Char,
    Cell,
    Logical,
    Categorical,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Label {
    MissingString,
    Numeric(f64),
    Integer(IntValue),
    Text(String),
    Logical(bool),
}

#[derive(Clone, Debug)]
pub(super) struct LabelVector {
    pub(super) labels: Vec<Label>,
    pub(super) kind: LabelKind,
    pub(super) categories: Option<Vec<String>>,
}

pub(super) fn labels_from_value(value: Value, name: &str) -> BuiltinResult<LabelVector> {
    match value {
        Value::Tensor(tensor) => {
            ensure_vector_shape(tensor.shape(), name)?;
            let labels = if let Some(storage) = tensor.integer_storage() {
                storage
                    .exact_values()
                    .into_iter()
                    .map(Label::Integer)
                    .collect()
            } else {
                tensor::tensor_into_values_f64(tensor)
                    .into_iter()
                    .map(Label::Numeric)
                    .collect()
            };
            Ok(LabelVector {
                labels,
                kind: LabelKind::Numeric,
                categories: None,
            })
        }
        Value::Num(value) => Ok(LabelVector {
            labels: vec![Label::Numeric(value)],
            kind: LabelKind::Numeric,
            categories: None,
        }),
        Value::Int(value) => Ok(LabelVector {
            labels: vec![Label::Integer(value)],
            kind: LabelKind::Numeric,
            categories: None,
        }),
        Value::Bool(value) => Ok(LabelVector {
            labels: vec![Label::Logical(value)],
            kind: LabelKind::Logical,
            categories: None,
        }),
        Value::LogicalArray(array) => {
            ensure_vector_shape(&array.shape, name)?;
            Ok(LabelVector {
                labels: array
                    .data
                    .into_iter()
                    .map(|value| Label::Logical(value != 0))
                    .collect(),
                kind: LabelKind::Logical,
                categories: None,
            })
        }
        Value::String(text) => Ok(LabelVector {
            labels: vec![text.as_str().map(|text| Label::Text(text.to_owned())).unwrap_or(Label::MissingString)],
            kind: LabelKind::String,
            categories: None,
        }),
        Value::StringArray(array) => {
            ensure_vector_shape(&array.shape, name)?;
            Ok(LabelVector {
                labels: array.data().iter().map(|text| text.as_str().map(|text| Label::Text(text.to_owned())).unwrap_or(Label::MissingString)).collect(),
                kind: LabelKind::String,
                categories: None,
            })
        }
        Value::CharArray(array) => Ok(LabelVector {
            labels: char_rows(&array, name)?.into_iter().map(Label::Text).collect(),
            kind: LabelKind::Char,
            categories: None,
        }),
        Value::Cell(cell) => {
            ensure_vector_shape(&[cell.rows, cell.cols], name)?;
            let mut labels = Vec::with_capacity(cell.data().len());
            for item in cell.data().iter() {
                labels.push(match item {
                    Value::String(text) if text.is_missing() => Label::MissingString,
                    Value::StringArray(array) if array.data().len() == 1 && array.data()[0].is_missing() => Label::MissingString,
                    _ => Label::Text(scalar_text(item, name)?),
                });
            }
            Ok(LabelVector {
                labels,
                kind: LabelKind::Cell,
                categories: None,
            })
        }
        Value::Object(object) if object.is_class(runmat_types::standard::CATEGORICAL) => {
            ensure_categorical_vector_shape(&object, name)?;
            let labels = crate::builtins::table::categorical_labels(&Value::Object(object.clone()))?
                .into_iter()
                .map(|label| label.as_str().map(|text| Label::Text(text.to_owned())).unwrap_or(Label::MissingString))
                .collect();
            Ok(LabelVector {
                labels,
                kind: LabelKind::Categorical,
                categories: categorical_categories(&object)?,
            })
        }
        other => Err(invalid(format!(
            "perfcurve: {name} must be numeric, logical, string, character, categorical, or cellstr labels; got {other:?}"
        ))),
    }
}

pub(super) fn scalar_label_from_value(value: Value, name: &str) -> BuiltinResult<Label> {
    let vector = labels_from_value(value, name)?;
    if vector.labels.len() != 1 {
        return Err(invalid(format!("perfcurve: {name} must be a scalar label")));
    }
    Ok(vector.labels[0].clone())
}

pub(super) fn coerce_labels_to_kind(
    labels: Vec<Label>,
    kind: LabelKind,
    name: &str,
) -> BuiltinResult<Vec<Label>> {
    labels
        .into_iter()
        .map(|label| coerce_label_to_kind(label, kind, name))
        .collect()
}

pub(super) fn coerce_label_to_kind(
    label: Label,
    kind: LabelKind,
    name: &str,
) -> BuiltinResult<Label> {
    match (kind, label) {
        (LabelKind::Logical, Label::Logical(value)) => Ok(Label::Logical(value)),
        (LabelKind::Logical, Label::Text(value)) => match canonical(&value).as_str() {
            "true" | "logical1" | "1" => Ok(Label::Logical(true)),
            "false" | "logical0" | "0" => Ok(Label::Logical(false)),
            _ => Err(invalid(format!(
                "perfcurve: {name} must be true or false for logical labels"
            ))),
        },
        (LabelKind::Logical, Label::Numeric(value)) if value == 0.0 || value == 1.0 => {
            Ok(Label::Logical(value != 0.0))
        }
        (LabelKind::Logical, Label::Integer(value)) if value.is_zero() => Ok(Label::Logical(false)),
        (LabelKind::Logical, Label::Integer(value)) if value.try_to_u64() == Some(1) => {
            Ok(Label::Logical(true))
        }
        (LabelKind::Numeric, Label::Numeric(value)) => Ok(Label::Numeric(value)),
        (LabelKind::Numeric, Label::Integer(value)) => Ok(Label::Integer(value)),
        (LabelKind::Numeric, Label::Text(value)) => value
            .parse::<f64>()
            .map(Label::Numeric)
            .map_err(|_| invalid(format!("perfcurve: {name} must be numeric"))),
        (
            LabelKind::String | LabelKind::Char | LabelKind::Cell | LabelKind::Categorical,
            Label::Text(value),
        ) => Ok(Label::Text(value)),
        (
            LabelKind::String | LabelKind::Char | LabelKind::Cell | LabelKind::Categorical,
            Label::Numeric(value),
        ) => Ok(Label::Text(format_number(value))),
        (
            LabelKind::String | LabelKind::Char | LabelKind::Cell | LabelKind::Categorical,
            Label::Integer(value),
        ) => Ok(Label::Text(value.decimal_string())),
        (
            LabelKind::String | LabelKind::Char | LabelKind::Cell | LabelKind::Categorical,
            Label::Logical(value),
        ) => Ok(Label::Text(value.to_string())),
        _ => Err(invalid(format!(
            "perfcurve: {name} label type must match labels"
        ))),
    }
}

pub(super) fn ensure_vector_shape(shape: &[usize], name: &str) -> BuiltinResult<()> {
    if shape.iter().filter(|dim| **dim > 1).count() > 1 {
        return Err(invalid(format!("perfcurve: {name} must be a vector")));
    }
    Ok(())
}

pub(super) fn ensure_categorical_vector_shape(
    object: &ObjectInstance,
    name: &str,
) -> BuiltinResult<()> {
    match object.properties().get("Codes") {
        Some(Value::Tensor(tensor)) => ensure_vector_shape(tensor.shape(), name),
        Some(_) => Err(invalid("perfcurve: categorical Codes must be numeric")),
        None => Ok(()),
    }
}

pub(super) fn categorical_categories(
    object: &ObjectInstance,
) -> BuiltinResult<Option<Vec<String>>> {
    match object.properties().get("Categories") {
        Some(Value::StringArray(array)) => Ok(Some(
            array
                .data()
                .iter()
                .map(|text| {
                    text.as_str().map(str::to_owned).ok_or_else(|| {
                        invalid("perfcurve: categorical Categories must be nonmissing strings")
                    })
                })
                .collect::<BuiltinResult<Vec<_>>>()?,
        )),
        Some(_) => Err(invalid("perfcurve: categorical Categories must be strings")),
        None => Ok(None),
    }
}

pub(super) fn negative_classes(
    labels: &LabelVector,
    posclass: &Label,
    explicit: Option<&Vec<Label>>,
) -> BuiltinResult<Vec<Label>> {
    let mut classes = if let Some(explicit) = explicit {
        explicit.clone()
    } else if labels.kind == LabelKind::Categorical {
        let mut out = Vec::new();
        for category in labels.categories.iter().flatten() {
            let label = Label::Text(category.clone());
            if !same_label(&label, posclass)
                && labels
                    .labels
                    .iter()
                    .any(|candidate| same_label(candidate, &label))
            {
                out.push(label);
            }
        }
        for label in &labels.labels {
            if !same_label(label, posclass)
                && !is_missing_label(label)
                && !out.iter().any(|existing| same_label(existing, label))
            {
                out.push(label.clone());
            }
        }
        out
    } else {
        let mut out = dedupe_labels(labels.labels.clone());
        sort_labels(&mut out, labels.kind);
        out.retain(|label| !same_label(label, posclass));
        out
    };
    if classes.iter().any(|label| same_label(label, posclass)) {
        return Err(invalid("perfcurve: NegClass cannot include posclass"));
    }
    classes.retain(|label| {
        !is_missing_label(label)
            && labels
                .labels
                .iter()
                .any(|candidate| same_label(candidate, label))
    });
    Ok(dedupe_labels(classes))
}

pub(super) fn dedupe_labels<I>(labels: I) -> Vec<Label>
where
    I: IntoIterator<Item = Label>,
{
    let mut out = Vec::new();
    for label in labels {
        if is_missing_label(&label) {
            continue;
        }
        if !out.iter().any(|existing| same_label(existing, &label)) {
            out.push(label);
        }
    }
    out
}

pub(super) fn sort_labels(labels: &mut [Label], kind: LabelKind) {
    match kind {
        LabelKind::Numeric => labels.sort_by(|left, right| match (left, right) {
            (Label::Numeric(a), Label::Numeric(b)) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
            (Label::Integer(a), Label::Integer(b)) => {
                crate::builtins::logical::rel::integer_comparison::compare_integer_values(
                    a.clone(),
                    b.clone(),
                )
            }
            (Label::Integer(a), Label::Numeric(b)) => {
                crate::builtins::logical::rel::integer_comparison::integer_f64_order(a.clone(), *b)
                    .unwrap_or(Ordering::Equal)
            }
            (Label::Numeric(a), Label::Integer(b)) => {
                crate::builtins::logical::rel::integer_comparison::integer_f64_order(b.clone(), *a)
                    .map(Ordering::reverse)
                    .unwrap_or(Ordering::Equal)
            }
            _ => Ordering::Equal,
        }),
        LabelKind::String | LabelKind::Char | LabelKind::Cell | LabelKind::Categorical => labels
            .sort_by(|left, right| match (left, right) {
                (Label::Text(a), Label::Text(b)) => a.cmp(b),
                _ => Ordering::Equal,
            }),
        LabelKind::Logical => labels.sort_by_key(|label| match label {
            Label::Logical(value) => *value,
            _ => false,
        }),
    }
}

pub(super) fn label_key(label: &Label) -> String {
    match label {
        Label::MissingString => "m:".to_string(),
        Label::Numeric(value) if *value == 0.0 => "n:0".to_string(),
        Label::Numeric(value) => format!("n:{:016x}", value.to_bits()),
        Label::Integer(value) => format!("i:{}", value.decimal_string()),
        Label::Text(value) => format!("t:{value}"),
        Label::Logical(value) => format!("l:{}", usize::from(*value)),
    }
}

pub(super) fn same_label(left: &Label, right: &Label) -> bool {
    label_key(left) == label_key(right)
        || matches!(
            (left, right),
            (Label::Integer(integer), Label::Numeric(float))
                | (Label::Numeric(float), Label::Integer(integer))
                if crate::builtins::logical::rel::integer_comparison::integer_f64_order(
                    integer.clone(),
                    *float,
                ) == Some(Ordering::Equal)
        )
        || matches!(
            (left, right),
            (Label::Numeric(a), Label::Numeric(b)) if a.is_nan() && b.is_nan()
        )
}

pub(super) fn is_missing_label(label: &Label) -> bool {
    match label {
        Label::Numeric(value) => value.is_nan(),
        Label::Integer(_) => false,
        Label::MissingString => true,
        Label::Text(value) => value.is_empty(),
        Label::Logical(_) => false,
    }
}

pub(super) fn labels_to_cell_value(labels: &[Label]) -> BuiltinResult<Value> {
    Ok(Value::Cell(
        CellArray::new(
            labels
                .iter()
                .map(|label| Value::CharArray(CharArray::new_row(&label_display(label))))
                .collect(),
            labels.len(),
            1,
        )
        .map_err(|err| internal(format!("perfcurve: {err}")))?,
    ))
}

pub(super) fn label_display(label: &Label) -> String {
    match label {
        Label::MissingString => "<missing>".to_owned(),
        Label::Numeric(value) => format_number(*value),
        Label::Integer(value) => value.decimal_string(),
        Label::Text(value) => value.clone(),
        Label::Logical(value) => value.to_string(),
    }
}

pub(super) fn char_rows(array: &CharArray, name: &str) -> BuiltinResult<Vec<String>> {
    let mut rows = Vec::with_capacity(array.rows);
    for row in 0..array.rows {
        let mut units = Vec::with_capacity(array.cols);
        for col in 0..array.cols {
            units.push(array.data()[row + col * array.rows]);
        }
        let text = decode_utf16_code_units(&units).map_err(|error| invalid_utf16(name, error))?;
        rows.push(text.trim_end().to_string());
    }
    Ok(rows)
}

pub(super) fn scalar_text(value: &Value, name: &str) -> BuiltinResult<String> {
    match value {
        Value::String(runmat_value::StringElement::Text(value)) => Ok(value.as_str().to_owned()),
        Value::StringArray(array) if array.data().len() == 1 => array.data()[0]
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| invalid(format!("perfcurve: {name} must be a string scalar"))),
        Value::CharArray(array) if array.rows == 1 => {
            decode_utf16_code_units(array.data()).map_err(|error| invalid_utf16(name, error))
        }
        Value::Num(value) if value.is_finite() => Ok(format_number(*value)),
        Value::Int(value) => Ok(value.decimal_string()),
        Value::Bool(value) => Ok(value.to_string()),
        other => Err(invalid(format!(
            "perfcurve: {name} must be text scalar; got {other:?}"
        ))),
    }
}

fn invalid_utf16(name: &str, error: Utf16DecodeError) -> RuntimeError {
    let mut builder = build_runtime_error(format!(
        "perfcurve: {name} is not valid UTF-16 text: {error}"
    ))
    .with_builtin(NAME)
    .with_source(error);
    if let Some(identifier) = ERROR_INVALID_ARGUMENT.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(super) fn format_number(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{value:.0}")
    } else {
        value.to_string()
    }
}

pub(super) fn canonical(value: &str) -> String {
    value
        .chars()
        .filter(|ch| *ch != '_' && !ch.is_ascii_whitespace())
        .flat_map(char::to_lowercase)
        .collect()
}
