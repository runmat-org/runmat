use crate::BuiltinResult;
use runmat_value::{NumericScalar, Value};

use super::callback::Callable;
use super::error;

pub(super) struct Invocation {
    pub(super) callable: Callable,
    pub(super) arguments: Vec<Value>,
    pub(super) uniform_output: bool,
    pub(super) error_handler: Option<Callable>,
}

impl Invocation {
    pub(super) fn parse(function: Value, mut arguments: Vec<Value>) -> BuiltinResult<Self> {
        let callable = Callable::parse(function)?;
        let mut uniform_output = true;
        let mut error_handler = None;
        while let Some(option) = trailing_option(&arguments) {
            let value = arguments
                .pop()
                .ok_or_else(|| error::invalid("cellfun: option requires a value"))?;
            arguments.pop();
            match option {
                OptionName::UniformOutput => uniform_output = parse_uniform_output(value)?,
                OptionName::ErrorHandler => error_handler = Some(Callable::parse(value)?),
                OptionName::Unknown(name) => {
                    return Err(error::invalid(format!(
                        "cellfun: unknown name-value argument '{name}'"
                    )))
                }
            }
        }
        Ok(Self {
            callable,
            arguments,
            uniform_output,
            error_handler,
        })
    }
}

enum OptionName {
    UniformOutput,
    ErrorHandler,
    Unknown(String),
}

fn trailing_option(arguments: &[Value]) -> Option<OptionName> {
    let index = arguments.len().checked_sub(2)?;
    let name = extract_string(arguments.get(index)?)?;
    Some(match name.trim().to_ascii_lowercase().as_str() {
        "uniformoutput" => OptionName::UniformOutput,
        "errorhandler" => OptionName::ErrorHandler,
        _ => OptionName::Unknown(name),
    })
}

pub(super) fn parse_uniform_output(value: Value) -> BuiltinResult<bool> {
    match value {
        Value::Bool(value) => Ok(value),
        Value::LogicalArray(value) if value.len() == 1 => Ok(value.data[0] != 0),
        Value::Num(0.0) => Ok(false),
        Value::Num(1.0) => Ok(true),
        Value::Tensor(value)
            if value.len() == 1 && value.numeric_dtype() == runmat_value::NumericDType::F64 =>
        {
            match value.numeric_value_at(0) {
                Some(NumericScalar::F64(0.0)) => Ok(false),
                Some(NumericScalar::F64(1.0)) => Ok(true),
                _ => Err(invalid_uniform_output()),
            }
        }
        value => Err(error::uniform(format!(
            "cellfun: UniformOutput must be a logical scalar or double scalar 0 or 1, got {value:?}"
        ))),
    }
}

fn invalid_uniform_output() -> crate::RuntimeError {
    error::uniform("cellfun: UniformOutput must be a logical scalar or double scalar 0 or 1")
}

pub(super) fn extract_string(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value.clone()),
        Value::CharArray(value) if value.rows == 1 => Some(value.data.iter().collect()),
        Value::StringArray(value) if value.data.len() == 1 => Some(value.data[0].clone()),
        _ => None,
    }
}
