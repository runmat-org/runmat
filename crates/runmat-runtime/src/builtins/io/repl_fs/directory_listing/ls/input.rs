use runmat_value::Value;

use crate::BuiltinResult;

pub(super) enum Input {
    Current,
    Name(String),
}

impl Input {
    pub(super) fn parse(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<Self> {
        match args.len() {
            0 => Ok(Self::Current),
            1 => Ok(Self::Name(text(&args[0])?)),
            _ => Err(super::error::contract(&runmat_builtins::LS_ERROR_ARITY)),
        }
    }
}

fn text(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.as_str().to_owned()),
        Value::StringArray(array) if array.data().len() == 1 => {
            Ok(array.data()[0].as_str().to_owned())
        }
        Value::CharArray(chars)
            if chars.rows == 1
                && crate::builtins::common::shape::effective_rank(&chars.shape) <= 2 =>
        {
            Ok(chars
                .data()
                .iter()
                .collect::<String>()
                .trim_end()
                .to_string())
        }
        _ => Err(super::error::contract(&runmat_builtins::LS_ERROR_NAME)),
    }
}
