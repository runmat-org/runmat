use runmat_value::Value;

use crate::BuiltinResult;

pub(super) enum Input {
    Current,
    Name(String),
    FolderPattern { folder: String, pattern: String },
}

impl Input {
    pub(super) fn parse(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<Self> {
        match args.len() {
            0 => Ok(Self::Current),
            1 => Ok(Self::Name(text(
                &args[0],
                &runmat_builtins::DIR_ERROR_NAME,
            )?)),
            2 => {
                let (folder, pattern) = (&args[0], &args[1]);
                crate::compatibility::ensure_builtin_extension_enabled(
                    &runmat_builtins::DIR_FOLDER_PATTERN_EXTENSION,
                    "dir",
                )?;
                Ok(Self::FolderPattern {
                    folder: text(folder, &runmat_builtins::DIR_ERROR_FOLDER)?,
                    pattern: text(pattern, &runmat_builtins::DIR_ERROR_PATTERN)?,
                })
            }
            _ => Err(super::error::contract(&runmat_builtins::DIR_ERROR_ARITY)),
        }
    }
}

fn text(
    value: &Value,
    error: &'static runmat_builtins::BuiltinErrorDescriptor,
) -> BuiltinResult<String> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => Ok(text.as_str().to_owned()),
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Ok(array.data()[0]
                .text_ref()
                .expect("checked nonmissing scalar string")
                .as_str()
                .to_owned())
        }
        Value::CharArray(chars)
            if chars.rows == 1
                && crate::builtins::common::shape::effective_rank(&chars.shape) <= 2 =>
        {
            crate::builtins::io::utf16::decode_utf16_with_error(chars.data(), || {
                super::error::contract(error)
            })
            .map(|text| text.trim_end().to_string())
        }
        _ => Err(super::error::contract(error)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_text_uses_effective_rank_without_flattening_pages() {
        for (shape, valid) in [
            (vec![1, 2, 1], true),
            (vec![1, 1, 2], false),
            (vec![2, 1], false),
        ] {
            let value = Value::CharArray(
                runmat_value::CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], shape)
                    .unwrap(),
            );
            assert_eq!(
                text(&value, &runmat_builtins::DIR_ERROR_NAME).is_ok(),
                valid
            );
        }
        let empty = Value::CharArray(
            runmat_value::CharArray::from_unicode_scalars_with_shape(vec![], vec![0, 0]).unwrap(),
        );
        assert!(text(&empty, &runmat_builtins::DIR_ERROR_NAME).is_err());
    }
}
