use runmat_value::Value;

use crate::BuiltinResult;

pub(super) enum Input {
    Current,
    Name(String),
}

impl Input {
    pub(super) fn parse(args: &[Value]) -> BuiltinResult<Self> {
        match args {
            [] => Ok(Self::Current),
            [name] => Ok(Self::Name(text(name)?)),
            _ => Err(super::error::contract(&runmat_builtins::LS_ERROR_ARITY)),
        }
    }
}

fn text(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        Value::CharArray(chars) if chars.rows == 1 => {
            Ok(chars.data.iter().collect::<String>().trim_end().to_string())
        }
        _ => Err(super::error::contract(&runmat_builtins::LS_ERROR_NAME)),
    }
}
