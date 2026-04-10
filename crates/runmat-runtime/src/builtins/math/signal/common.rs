use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::{NumericDType, Tensor, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WindowSampling {
    Symmetric,
    Periodic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WindowOutputType {
    Double,
    Single,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WindowOptions {
    pub len: usize,
    pub sampling: WindowSampling,
    pub output_type: WindowOutputType,
}

pub(crate) fn signal_error(name: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
}

pub(crate) fn scalar_length_arg(value: Value, name: &str) -> BuiltinResult<usize> {
    let scalar = match value {
        Value::Num(n) => n,
        Value::Int(i) => i.to_f64(),
        Value::Bool(b) => usize::from(b) as f64,
        Value::Tensor(t) if t.data.len() == 1 => t.data[0],
        _ => {
            return Err(signal_error(
                name,
                format!("{name}: expected a nonnegative scalar integer length"),
            ));
        }
    };
    if !scalar.is_finite() || scalar < 0.0 {
        return Err(signal_error(
            name,
            format!("{name}: expected a nonnegative scalar integer length"),
        ));
    }
    Ok(scalar.round() as usize)
}

pub(crate) fn window_tensor(
    options: WindowOptions,
    name: &str,
    coeff: impl Fn(usize, usize) -> f64,
) -> BuiltinResult<Value> {
    let len = options.len;
    if len == 0 {
        return Tensor::new_with_dtype(Vec::new(), vec![0, 1], dtype_for(options.output_type))
            .map(Value::Tensor)
            .map_err(|e| signal_error(name, format!("{name}: {e}")));
    }
    if len == 1 {
        return Tensor::new_with_dtype(vec![1.0], vec![1, 1], dtype_for(options.output_type))
            .map(Value::Tensor)
            .map_err(|e| signal_error(name, format!("{name}: {e}")));
    }
    let effective_len = match options.sampling {
        WindowSampling::Symmetric => len,
        WindowSampling::Periodic => len + 1,
    };
    let mut data = (0..effective_len)
        .map(|idx| coeff(idx, effective_len))
        .collect::<Vec<_>>();
    if matches!(options.sampling, WindowSampling::Periodic) {
        data.pop();
    }
    Tensor::new_with_dtype(data, vec![len, 1], dtype_for(options.output_type))
        .map(Value::Tensor)
        .map_err(|e| signal_error(name, format!("{name}: {e}")))
}

pub(crate) fn parse_window_options(
    name: &str,
    len_value: Value,
    rest: &[Value],
    allow_type_name: bool,
) -> BuiltinResult<WindowOptions> {
    let len = scalar_length_arg(len_value, name)?;
    let mut sampling = WindowSampling::Symmetric;
    let mut output_type = WindowOutputType::Double;
    for arg in rest {
        let Some(keyword) = string_keyword(arg) else {
            return Err(signal_error(name, format!("{name}: unrecognized option")));
        };
        match keyword.as_str() {
            "symmetric" => sampling = WindowSampling::Symmetric,
            "periodic" => sampling = WindowSampling::Periodic,
            "double" if allow_type_name => output_type = WindowOutputType::Double,
            "single" if allow_type_name => output_type = WindowOutputType::Single,
            _ => {
                return Err(signal_error(
                    name,
                    format!("{name}: unrecognized option '{keyword}'"),
                ))
            }
        }
    }
    Ok(WindowOptions {
        len,
        sampling,
        output_type,
    })
}

pub(crate) fn dtype_for(output_type: WindowOutputType) -> NumericDType {
    match output_type {
        WindowOutputType::Double => NumericDType::F64,
        WindowOutputType::Single => NumericDType::F32,
    }
}

pub(crate) fn provider_precision_matches(output_type: WindowOutputType) -> bool {
    let Some(provider) = runmat_accelerate_api::provider() else {
        return false;
    };
    matches!(
        (provider.precision(), output_type),
        (
            runmat_accelerate_api::ProviderPrecision::F64,
            WindowOutputType::Double
        ) | (
            runmat_accelerate_api::ProviderPrecision::F32,
            WindowOutputType::Single
        )
    )
}

fn string_keyword(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.to_ascii_lowercase()),
        Value::CharArray(chars) => Some(chars.data.iter().collect::<String>().to_ascii_lowercase()),
        _ => None,
    }
}
