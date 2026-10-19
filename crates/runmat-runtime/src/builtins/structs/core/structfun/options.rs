use runmat_value::{StringArray, Value};

use super::callback::Callable;
use super::error;

pub(super) struct Options {
    pub(super) uniform: bool,
    pub(super) handler: Option<Callable>,
}

impl Options {
    pub(super) fn parse(values: Vec<Value>) -> crate::BuiltinResult<Self> {
        if !values.len().is_multiple_of(2) {
            return Err(error::invalid(
                "structfun: optional arguments must be name-value pairs",
            ));
        }
        let mut options = Self {
            uniform: true,
            handler: None,
        };
        let mut values = values.into_iter();
        while let Some(name) = values.next() {
            let value = values.next().expect("validated name-value pairs");
            match option_name(name)?.as_str() {
                "uniformoutput" => options.uniform = boolean(value)?,
                "errorhandler" => options.handler = Some(Callable::parse(value)?),
                name => {
                    return Err(error::invalid(format!(
                        "structfun: unknown name-value argument '{name}'"
                    )))
                }
            }
        }
        Ok(options)
    }
}

fn option_name(value: Value) -> crate::BuiltinResult<String> {
    match value {
        Value::String(value) => Ok(value.trim().to_ascii_lowercase()),
        Value::CharArray(value) if value.rows == 1 => Ok(value
            .data
            .iter()
            .collect::<String>()
            .trim()
            .to_ascii_lowercase()),
        Value::StringArray(StringArray { data, .. }) if data.len() == 1 => {
            Ok(data[0].trim().to_ascii_lowercase())
        }
        _ => Err(error::invalid(
            "structfun: option names must be character rows or string scalars",
        )),
    }
}

fn boolean(value: Value) -> crate::BuiltinResult<bool> {
    match value {
        Value::Bool(value) => Ok(value),
        Value::Num(value) => Ok(value != 0.0),
        Value::Int(value) => Ok(!value.is_zero()),
        Value::String(value) => boolean_text(&value),
        Value::CharArray(value) if value.rows == 1 => {
            boolean_text(&value.data.iter().collect::<String>())
        }
        Value::StringArray(StringArray { data, .. }) if data.len() == 1 => boolean_text(&data[0]),
        _ => Err(boolean_error()),
    }
}

fn boolean_text(value: &str) -> crate::BuiltinResult<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "on" => Ok(true),
        "false" | "off" => Ok(false),
        _ => Err(boolean_error()),
    }
}

fn boolean_error() -> crate::RuntimeError {
    error::uniform("structfun: UniformOutput must be logical true or false")
}
