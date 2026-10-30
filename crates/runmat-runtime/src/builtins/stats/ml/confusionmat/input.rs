use super::*;
pub(super) fn labels_from_value(value: Value, name: &str) -> BuiltinResult<LabelVector> {
    match value {
        Value::Tensor(tensor) => {
            ensure_vector_shape(tensor.shape(), name)?;
            if let Some(storage) = tensor.integer_storage() {
                Ok(LabelVector { labels: storage.exact_values().into_iter().map(Label::Integer).collect(), kind: LabelKind::Integer(storage.numeric_dtype()), categories: None })
            } else {
                Ok(LabelVector { labels: vector_values(&tensor, name)?.into_iter().map(Label::Numeric).collect(), kind: LabelKind::Numeric, categories: None })
            }
        }
        Value::Num(value) => Ok(LabelVector {
            labels: vec![Label::Numeric(value)],
            kind: LabelKind::Numeric,
            categories: None,
        }),
        Value::Int(value) => Ok(LabelVector {
            kind: LabelKind::Integer(int_value_dtype(&value)),
            labels: vec![Label::Integer(value)],
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
        Value::CharArray(array) => {
            if array.shape.iter().skip(2).any(|dimension| *dimension != 1) {
                return Err(invalid("confusionmat: character labels must be two-dimensional"));
            }
            Ok(LabelVector {
            labels: char_rows(&array)?.into_iter().map(Label::Text).collect(),
            kind: LabelKind::Char,
            categories: None,
        })},
        Value::Cell(cell) => {
            ensure_vector_shape(&cell.shape, name)?;
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
            "confusionmat: {name} must be numeric, logical, string, character, categorical, or cellstr labels; got {other:?}"
        ))),
    }
}

pub(super) fn vector_values(tensor: &Tensor, name: &str) -> BuiltinResult<Vec<f64>> {
    ensure_vector_shape(tensor.shape(), name)?;
    Ok(tensor::tensor_values_f64(tensor))
}

pub(super) fn ensure_vector_shape(shape: &[usize], name: &str) -> BuiltinResult<()> {
    if shape.iter().filter(|dim| **dim > 1).count() > 1 {
        return Err(invalid(format!("confusionmat: {name} must be a vector")));
    }
    Ok(())
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
                        invalid("confusionmat: categorical Categories must be nonmissing strings")
                    })
                })
                .collect::<BuiltinResult<Vec<_>>>()?,
        )),
        Some(_) => Err(invalid(
            "confusionmat: categorical Categories must be strings",
        )),
        None => Ok(None),
    }
}

pub(super) fn ensure_categorical_vector_shape(
    object: &ObjectInstance,
    name: &str,
) -> BuiltinResult<()> {
    match object.properties().get("Codes") {
        Some(Value::Tensor(tensor)) => ensure_vector_shape(tensor.shape(), name),
        Some(_) => Err(invalid("confusionmat: categorical Codes must be numeric")),
        None => Ok(()),
    }
}

pub(super) fn scalar_text(value: &Value, name: &str) -> BuiltinResult<String> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => Ok(text.as_str().to_owned()),
        Value::CharArray(chars)
            if chars.rows <= 1 && chars.shape.iter().skip(2).all(|dimension| *dimension == 1) =>
        {
            crate::builtins::stats::decode_utf16_with_error(chars.data(), || {
                invalid(format!("confusionmat: {name} must be valid UTF-16 text"))
            })
        }
        Value::StringArray(array) if array.data().len() == 1 => array.data()[0]
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| invalid(format!("confusionmat: {name} must be a string scalar"))),
        other => Err(invalid(format!(
            "confusionmat: {name} must be a string scalar, got {other:?}"
        ))),
    }
}

pub(super) fn canonical(text: &str) -> String {
    text.chars()
        .filter(|ch| *ch != '_' && *ch != '-' && !ch.is_whitespace())
        .collect::<String>()
        .to_ascii_lowercase()
}
