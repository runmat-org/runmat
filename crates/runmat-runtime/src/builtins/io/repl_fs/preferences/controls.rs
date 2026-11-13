use runmat_value::Value;

use crate::builtins::common::validation::value_text_scalar_string;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

pub(super) fn error(name: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
}

pub(super) fn map_control_flow(name: &str, source: RuntimeError) -> RuntimeError {
    let identifier = source.identifier().map(str::to_string);
    let mut builder = build_runtime_error(format!("{name}: {}", source.message()))
        .with_builtin(name)
        .with_source(source);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(super) fn scalar_text(value: &Value, name: &str, arg: &str) -> BuiltinResult<String> {
    preference_scalar_text(value).ok_or_else(|| {
        error(
            name,
            format!("{name}: {arg} must be a string scalar or character vector"),
        )
    })
}

fn preference_scalar_text(value: &Value) -> Option<String> {
    let mut text = value_text_scalar_string(value)?;
    if matches!(value, Value::CharArray(_)) {
        while text.ends_with(' ') {
            text.pop();
        }
    }
    Some(text)
}

#[derive(Debug)]
pub(super) enum PreferenceNames {
    Scalar(String),
    Array {
        names: Vec<String>,
        shape: Vec<usize>,
    },
}

pub(super) fn preference_names(
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<PreferenceNames> {
    match value {
        Value::String(_) | Value::CharArray(_) => {
            Ok(PreferenceNames::Scalar(scalar_text(value, builtin, "preference")?))
        }
        Value::StringArray(array) if array.data().iter().all(|text| !text.is_missing()) => {
            if let Some(text) = preference_scalar_text(value) {
                Ok(PreferenceNames::Scalar(text))
            } else {
                Ok(PreferenceNames::Array {
                    names: array.data().iter().map(|text| text.as_str().expect("checked nonmissing string").to_owned()).collect(),
                    shape: array.shape.as_slice().to_vec(),
                })
            }
        }
        Value::Cell(cell) => {
            let mut names = Vec::with_capacity(cell.data().len());
            for value in cell.data() {
                if let Value::CharArray(_) = value {
                    if let Some(text) = preference_scalar_text(value) {
                        names.push(text);
                        continue;
                    }
                }
                return Err(error(
                    builtin,
                    format!("{builtin}: preference cell entries must be character vectors"),
                ));
            }
            Ok(PreferenceNames::Array {
                names,
                shape: cell.shape.as_slice().to_vec(),
            })
        }
        _ => Err(error(
            builtin,
            format!("{builtin}: preference must be a character vector, string scalar, cell array of character vectors, or string array"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::{CellArray, CharArray, StringArray};

    #[test]
    fn controls_reject_character_pages_but_keep_scalar_padding_semantics() {
        let page = Value::CharArray(
            CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], vec![1, 1, 2])
                .expect("page"),
        );
        assert!(scalar_text(&page, "setpref", "group").is_err());
        assert!(preference_names(&page, "setpref").is_err());
        let cell = Value::Cell(CellArray::new(vec![page], 1, 1).expect("cell"));
        assert!(preference_names(&cell, "setpref").is_err());

        let trailing_singletons = Value::CharArray(
            CharArray::from_unicode_scalars_with_shape(vec!['a', 'b', ' '], vec![1, 3, 1])
                .expect("row"),
        );
        assert_eq!(
            scalar_text(&trailing_singletons, "setpref", "group").unwrap(),
            "ab"
        );
    }

    #[test]
    fn singleton_string_array_is_scalar_and_multiple_strings_keep_shape() {
        let scalar = Value::StringArray(
            StringArray::new(vec!["one".into()], vec![1, 1, 1]).expect("scalar"),
        );
        assert!(matches!(
            preference_names(&scalar, "getpref").unwrap(),
            PreferenceNames::Scalar(name) if name == "one"
        ));
        let multiple = Value::StringArray(
            StringArray::new(vec!["a".into(), "b".into()], vec![1, 1, 2]).expect("pages"),
        );
        assert!(matches!(
            preference_names(&multiple, "getpref").unwrap(),
            PreferenceNames::Array { shape, .. } if shape == [1, 1, 2]
        ));
    }
}
