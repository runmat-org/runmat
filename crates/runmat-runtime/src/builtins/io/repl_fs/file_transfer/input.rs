use runmat_builtins::BuiltinErrorDescriptor;
use runmat_value::Value;

use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

pub(super) async fn gather(
    args: &crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
    builtin: &'static str,
) -> BuiltinResult<crate::call::BuiltinArguments> {
    args.try_map_preserving_roots_async(args.len(), runtime, async |index| {
        gather_if_needed_async(&args[index])
            .await
            .map_err(|error| map_gather_error(error, builtin))
    })
    .await
}

pub(super) fn text(
    value: &Value,
    error: &'static BuiltinErrorDescriptor,
    builtin: &'static str,
) -> BuiltinResult<String> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => Ok(text.as_str().to_owned()),
        Value::CharArray(array)
            if array.rows == 1
                && crate::builtins::common::shape::effective_rank(&array.shape) <= 2 =>
        {
            crate::builtins::io::utf16::decode_utf16_with_error(array.data(), || {
                descriptor_error(error, builtin)
            })
        }
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Ok(array.data()[0]
                .text_ref()
                .expect("checked nonmissing scalar string")
                .as_str()
                .to_owned())
        }
        _ => Err(descriptor_error(error, builtin)),
    }
}

pub(super) fn force_flag(
    value: &Value,
    error: &'static BuiltinErrorDescriptor,
    builtin: &'static str,
) -> BuiltinResult<bool> {
    if text(value, error, builtin)?.eq_ignore_ascii_case("f") {
        Ok(true)
    } else {
        Err(descriptor_error(error, builtin))
    }
}

pub(super) fn descriptor_error(
    error: &'static BuiltinErrorDescriptor,
    builtin: &'static str,
) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(builtin);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(super) fn message_identifier(error: &'static BuiltinErrorDescriptor) -> &'static str {
    error.identifier.unwrap_or("")
}

fn map_gather_error(error: RuntimeError, builtin: &'static str) -> RuntimeError {
    let identifier = error.identifier().map(str::to_string);
    let mut builder = build_runtime_error(format!("{builtin}: {}", error.message()))
        .with_builtin(builtin)
        .with_source(error);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transfer_text_admission_preserves_rows_and_rejects_pages() {
        for (builtin, descriptor) in [
            ("copyfile", &runmat_builtins::COPYFILE_ERROR_SOURCE_ARG),
            ("movefile", &runmat_builtins::MOVEFILE_ERROR_SOURCE_ARG),
        ] {
            for (shape, valid) in [(vec![1, 2, 1], true), (vec![1, 1, 2], false)] {
                let value = Value::CharArray(
                    runmat_value::CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], shape)
                        .unwrap(),
                );
                assert_eq!(text(&value, descriptor, builtin).is_ok(), valid);
            }
        }
    }
}
