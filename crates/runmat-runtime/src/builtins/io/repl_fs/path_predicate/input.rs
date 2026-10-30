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
    args: crate::call::FunctionArgumentView<'_>,
    builtin: &'static str,
    arity_error: &'static BuiltinErrorDescriptor,
    path_error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<PathInputs> {
    if args.len() != 1 {
        return Err(descriptor_error(arity_error, builtin));
    };
    let input = args.first().expect("validated single input");
    match input {
        Value::String(runmat_value::StringElement::Text(text)) => {
            Ok(PathInputs::Scalar(path(text, builtin)?))
        }
        Value::CharArray(chars) if character_is_row(chars) => Ok(PathInputs::Scalar(path(
            &character_row(chars, path_error, builtin)?,
            builtin,
        )?)),
        Value::StringArray(array) => Ok(PathInputs::Shaped {
            paths: array
                .data()
                .iter()
                .map(|text| {
                    path(
                        text.as_str()
                            .ok_or_else(|| descriptor_error(path_error, builtin))?,
                        builtin,
                    )
                })
                .collect::<BuiltinResult<_>>()?,
            shape: array.shape.as_slice().to_vec(),
        }),
        Value::Cell(array) => Ok(PathInputs::Shaped {
            paths: array
                .iter_column_major()
                .map(|value| match value {
                    Value::CharArray(chars) if character_is_row(chars) => {
                        path(&character_row(chars, path_error, builtin)?, builtin)
                    }
                    _ => Err(descriptor_error(path_error, builtin)),
                })
                .collect::<BuiltinResult<_>>()?,
            shape: array.shape.as_slice().to_vec(),
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

fn character_row(
    chars: &CharArray,
    error: &'static BuiltinErrorDescriptor,
    builtin: &'static str,
) -> BuiltinResult<String> {
    crate::builtins::io::utf16::decode_utf16_with_error(&chars.data()[..chars.cols], || {
        descriptor_error(error, builtin)
    })
    .map(|text| text.trim_end().to_owned())
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
