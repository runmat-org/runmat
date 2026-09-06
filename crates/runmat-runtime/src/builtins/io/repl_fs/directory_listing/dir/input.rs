use runmat_value::Value;

use crate::BuiltinResult;

pub(super) enum Input {
    Current,
    Name(String),
    FolderPattern { folder: String, pattern: String },
}

impl Input {
    pub(super) fn parse(args: &[Value]) -> BuiltinResult<Self> {
        match args {
            [] => Ok(Self::Current),
            [name] => Ok(Self::Name(text(name, &runmat_builtins::DIR_ERROR_NAME)?)),
            [folder, pattern] => {
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
        Value::String(text) => Ok(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        Value::CharArray(chars) if chars.rows == 1 => {
            Ok(chars.data.iter().collect::<String>().trim_end().to_string())
        }
        _ => Err(super::error::contract(error)),
    }
}
