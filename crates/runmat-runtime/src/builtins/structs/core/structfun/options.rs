use runmat_value::Value;

use super::callback::Callable;
use super::error;

pub(super) struct Options {
    pub(super) uniform: bool,
    pub(super) handler: Option<Callable>,
}

impl Options {
    pub(super) fn parse(values: crate::call::BuiltinArguments) -> crate::BuiltinResult<Self> {
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
        Value::String(runmat_value::StringElement::Text(value)) => {
            Ok(value.trim().to_ascii_lowercase())
        }
        Value::CharArray(value) if value.rows == 1 => {
            crate::builtins::common::text::decode_utf16_with_error(value.data(), || {
                error::invalid("structfun: option name is not valid UTF-16 text")
            })
            .map(|text| text.trim().to_ascii_lowercase())
        }
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Ok(array.data()[0]
                .text_ref()
                .expect("checked nonmissing scalar string")
                .trim()
                .to_ascii_lowercase())
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
        Value::String(runmat_value::StringElement::Text(value)) => boolean_text(&value),
        Value::CharArray(value) if value.rows == 1 => {
            let text =
                crate::builtins::common::text::decode_utf16_with_error(value.data(), || {
                    error::uniform("structfun: UniformOutput is not valid UTF-16 text")
                })?;
            boolean_text(&text)
        }
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            boolean_text(
                array.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string"),
            )
        }
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
