use crate::BuiltinResult;
use runmat_builtins::{
    ARRAYFUN_ERROR_UNIFORM_OUTPUT_OPTION, ARRAYFUN_GPU_OPTIONS_EXTENSION,
    ARRAYFUN_TEXT_CALLABLE_EXTENSION,
};
use runmat_value::{NumericScalar, Value};

use super::callback::Callable;
use super::error::{arrayfun_error, arrayfun_error_with_detail, arrayfun_flow};
use super::BUILTIN_NAME;

pub(super) struct Invocation {
    pub callable: Callable,
    pub inputs: Vec<Value>,
    pub uniform_output: bool,
    pub error_handler: Option<Callable>,
}

impl Invocation {
    pub fn parse(function: Value, mut inputs: Vec<Value>) -> BuiltinResult<Self> {
        ensure_text_callable_policy(&function)?;
        let callable = Callable::from_function(function)?;
        let mut uniform_output = true;
        let mut uniform_output_explicit = false;
        let mut error_handler = None;

        while let Some(option) = trailing_option(&inputs) {
            let value = inputs.pop().ok_or_else(|| {
                arrayfun_flow("arrayfun: option name requires a corresponding value")
            })?;
            inputs.pop();
            match option {
                OptionName::UniformOutput => {
                    uniform_output = parse_uniform_output(value)?;
                    uniform_output_explicit = true;
                }
                OptionName::ErrorHandler => {
                    ensure_text_callable_policy(&value)?;
                    error_handler = Some(Callable::from_function(value)?);
                }
                OptionName::Unknown(name) => {
                    return Err(arrayfun_flow(format!(
                        "arrayfun: unknown name-value argument '{name}'"
                    )))
                }
            }
        }
        if inputs.is_empty() {
            return Err(arrayfun_flow("arrayfun: expected at least one input array"));
        }
        let has_gpu_input = inputs
            .iter()
            .any(|value| matches!(value, Value::GpuTensor(_)));
        if has_gpu_input && (uniform_output_explicit || error_handler.is_some()) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &ARRAYFUN_GPU_OPTIONS_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
        Ok(Self {
            callable,
            inputs,
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

fn trailing_option(values: &[Value]) -> Option<OptionName> {
    let candidate = values.len().checked_sub(2)?;
    let name = extract_string(values.get(candidate)?)?;
    Some(match name.trim().to_ascii_lowercase().as_str() {
        "uniformoutput" => OptionName::UniformOutput,
        "errorhandler" => OptionName::ErrorHandler,
        _ => OptionName::Unknown(name),
    })
}

fn ensure_text_callable_policy(value: &Value) -> BuiltinResult<()> {
    if matches!(
        value,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
    ) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ARRAYFUN_TEXT_CALLABLE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

pub(super) fn parse_uniform_output(value: Value) -> BuiltinResult<bool> {
    match value {
        Value::Bool(b) => Ok(b),
        Value::LogicalArray(logical) if logical.len() == 1 => Ok(logical.data[0] != 0),
        Value::Num(0.0) => Ok(false),
        Value::Num(1.0) => Ok(true),
        Value::Tensor(tensor)
            if tensor.len() == 1 && tensor.numeric_dtype() == runmat_value::NumericDType::F64 =>
        {
            match tensor.numeric_value_at(0) {
                Some(NumericScalar::F64(0.0)) => Ok(false),
                Some(NumericScalar::F64(1.0)) => Ok(true),
                _ => Err(arrayfun_error(&ARRAYFUN_ERROR_UNIFORM_OUTPUT_OPTION)),
            }
        }
        other => Err(arrayfun_error_with_detail(
            &ARRAYFUN_ERROR_UNIFORM_OUTPUT_OPTION,
            format!("got {other:?}"),
        )),
    }
}

pub(super) fn extract_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        _ => None,
    }
}
