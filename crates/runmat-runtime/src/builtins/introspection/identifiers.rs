//! MATLAB language identifier and keyword introspection builtins.

use crate::builtins::common::identifiers::{
    is_matlab_keyword, MATLAB_KEYWORDS, MATLAB_NAME_LENGTH_MAX,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::{
    BuiltinErrorDescriptor, ERROR_ISKEYWORD_ARG_COUNT, ERROR_ISKEYWORD_INTERNAL,
    ERROR_ISKEYWORD_TEXT, ERROR_NAMELENGTHMAX_ARG_COUNT,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, CharArray, Value};

#[runtime_builtin(
    name = "namelengthmax",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::identifiers"
)]
fn namelengthmax_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    if !args.is_empty() {
        return Err(error("namelengthmax", &ERROR_NAMELENGTHMAX_ARG_COUNT));
    }
    Ok(Value::Num(MATLAB_NAME_LENGTH_MAX as f64))
}

#[runtime_builtin(
    name = "iskeyword",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::identifiers"
)]
fn iskeyword_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    match (args.len(), args.first()) {
        (0, _) => keyword_cell(),
        (1, Some(value)) => Ok(Value::Bool(is_matlab_keyword(&text_scalar(value)?))),
        _ => Err(error("iskeyword", &ERROR_ISKEYWORD_ARG_COUNT)),
    }
}

fn keyword_cell() -> BuiltinResult<Value> {
    let values = MATLAB_KEYWORDS
        .iter()
        .map(|keyword| Value::CharArray(CharArray::new_row(keyword)))
        .collect::<Vec<_>>();
    CellArray::new(values, MATLAB_KEYWORDS.len(), 1)
        .map(Value::Cell)
        .map_err(|err| {
            error_with_detail(
                "iskeyword",
                &ERROR_ISKEYWORD_INTERNAL,
                format!("iskeyword: failed to build keyword list ({err})"),
            )
        })
}

fn text_scalar(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => Ok(text.as_str().to_owned()),
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Ok(array.data()[0]
                .text_ref()
                .expect("checked nonmissing scalar string")
                .as_str()
                .to_owned())
        }
        Value::CharArray(chars) if chars.rows <= 1 => {
            super::text::decode_utf16_with_error(chars.data(), || {
                error("iskeyword", &ERROR_ISKEYWORD_TEXT)
            })
        }
        _ => Err(error("iskeyword", &ERROR_ISKEYWORD_TEXT)),
    }
}

fn error(builtin: &str, descriptor: &'static BuiltinErrorDescriptor) -> RuntimeError {
    error_with_detail(builtin, descriptor, descriptor.message)
}

fn error_with_detail(
    builtin: &str,
    descriptor: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(builtin);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::StringArray;

    #[test]
    fn namelengthmax_returns_current_identifier_limit() {
        assert_eq!(
            namelengthmax_builtin(crate::builtins::common::test_support::builtin_arguments(
                Vec::new()
            ))
            .expect("namelengthmax"),
            Value::Num(2048.0)
        );
    }

    #[test]
    fn namelengthmax_rejects_arguments() {
        assert!(
            namelengthmax_builtin(crate::builtins::common::test_support::builtin_arguments(
                vec![Value::Num(1.0)]
            ))
            .is_err()
        );
    }

    #[test]
    fn iskeyword_lists_keywords_as_cell_column() {
        let Value::Cell(cell) = iskeyword_builtin(
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        )
        .expect("iskeyword") else {
            panic!("expected cell")
        };
        assert_eq!(cell.shape, vec![20, 1]);
        assert_eq!(
            cell.data().first(),
            Some(&Value::CharArray(CharArray::new_row("break")))
        );
        assert_eq!(
            cell.data().last(),
            Some(&Value::CharArray(CharArray::new_row("while")))
        );
    }

    #[test]
    fn iskeyword_accepts_string_scalar_and_char_vector() {
        assert_eq!(
            iskeyword_builtin(crate::builtins::common::test_support::builtin_arguments(
                vec![Value::String("for".to_string().into())]
            ))
            .expect("keyword"),
            Value::Bool(true)
        );
        assert_eq!(
            iskeyword_builtin(crate::builtins::common::test_support::builtin_arguments(
                vec![Value::CharArray(CharArray::new_row("plot"))]
            ))
            .expect("keyword"),
            Value::Bool(false)
        );
        assert_eq!(
            iskeyword_builtin(crate::builtins::common::test_support::builtin_arguments(
                vec![Value::StringArray(
                    StringArray::new(vec!["end".to_string()], vec![1, 1]).expect("string")
                )]
            ))
            .expect("keyword"),
            Value::Bool(true)
        );
    }

    #[test]
    fn iskeyword_rejects_non_scalar_text_and_other_values() {
        assert!(
            iskeyword_builtin(crate::builtins::common::test_support::builtin_arguments(
                vec![Value::StringArray(
                    StringArray::new(vec!["for".to_string(), "end".to_string()], vec![1, 2])
                        .expect("string array")
                )]
            ))
            .is_err()
        );
        assert!(
            iskeyword_builtin(crate::builtins::common::test_support::builtin_arguments(
                vec![Value::Num(1.0)]
            ))
            .is_err()
        );
    }

    #[test]
    fn identifier_errors_use_their_declared_public_descriptors() {
        let length = namelengthmax_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
        )
        .unwrap_err();
        assert_eq!(
            length.identifier(),
            ERROR_NAMELENGTHMAX_ARG_COUNT.identifier
        );
        let predicate = iskeyword_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
        )
        .unwrap_err();
        assert_eq!(predicate.identifier(), ERROR_ISKEYWORD_TEXT.identifier);
    }
}
