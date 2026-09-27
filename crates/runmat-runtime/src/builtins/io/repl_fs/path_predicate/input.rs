use std::path::PathBuf;

use runmat_builtins::BuiltinErrorDescriptor;
use runmat_value::{CharArray, Value};

use crate::builtins::common::fs::expand_user_path;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

pub(super) enum PathInputs {
    Scalar(PathBuf),
    Shaped {
        paths: Vec<PathBuf>,
        shape: Vec<usize>,
    },
}

pub(super) fn parse(
    args: &[Value],
    builtin: &'static str,
    arity_error: &'static BuiltinErrorDescriptor,
    path_error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<PathInputs> {
    let [input] = args else {
        return Err(descriptor_error(arity_error, builtin));
    };
    match input {
        Value::String(text) => Ok(PathInputs::Scalar(path(text, builtin)?)),
        Value::CharArray(chars) if character_is_row(chars) => {
            Ok(PathInputs::Scalar(path(&character_row(chars), builtin)?))
        }
        Value::StringArray(array) => Ok(PathInputs::Shaped {
            paths: array
                .data
                .iter()
                .map(|text| path(text, builtin))
                .collect::<BuiltinResult<_>>()?,
            shape: array.shape.clone(),
        }),
        Value::Cell(array) => Ok(PathInputs::Shaped {
            paths: array
                .iter_column_major()
                .map(|value| match value {
                    Value::CharArray(chars) if character_is_row(chars) => {
                        path(&character_row(chars), builtin)
                    }
                    _ => Err(descriptor_error(path_error, builtin)),
                })
                .collect::<BuiltinResult<_>>()?,
            shape: array.shape.clone(),
        }),
        _ => Err(descriptor_error(path_error, builtin)),
    }
}

fn path(text: &str, builtin: &'static str) -> BuiltinResult<PathBuf> {
    expand_user_path(text.trim(), builtin)
        .map(PathBuf::from)
        .map_err(|error| {
            build_runtime_error(format!("{builtin}: {error}"))
                .with_builtin(builtin)
                .build()
        })
}

fn character_row(chars: &CharArray) -> String {
    chars
        .data
        .iter()
        .take(chars.cols)
        .collect::<String>()
        .trim_end()
        .to_owned()
}

fn character_is_row(chars: &CharArray) -> bool {
    chars.rows == 1 && crate::builtins::common::shape::effective_rank(&chars.shape) <= 2
}

fn descriptor_error(
    descriptor: &'static BuiltinErrorDescriptor,
    builtin: &'static str,
) -> RuntimeError {
    let mut builder = build_runtime_error(descriptor.message).with_builtin(builtin);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
