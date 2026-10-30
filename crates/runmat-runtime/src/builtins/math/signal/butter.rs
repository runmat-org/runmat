//! Butterworth IIR filter design builtin.

use runmat_builtins::{
    BuiltinErrorDescriptor, BUTTER_ERROR_ARG_COUNT, BUTTER_ERROR_INTERNAL,
    BUTTER_ERROR_INVALID_FREQUENCY, BUTTER_ERROR_INVALID_OPTION, BUTTER_ERROR_INVALID_ORDER,
    BUTTER_ERROR_TOO_MANY_OUTPUTS, BUTTER_GPU_INPUT_EXTENSION, BUTTER_INTEGER_CUTOFF_EXTENSION,
    BUTTER_INTEGER_ORDER_EXTENSION, BUTTER_LARGE_ORDER_EXTENSION, BUTTER_LOGICAL_CUTOFF_EXTENSION,
    BUTTER_LOGICAL_ORDER_EXTENSION, BUTTER_OPTION_ALIAS_EXTENSION, BUTTER_SINGLE_CUTOFF_EXTENSION,
    BUTTER_SINGLE_ORDER_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, Tensor, Value};

use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::builtins::math::signal::common::SignalCooperation;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "butter";
const EPS: f64 = 1.0e-12;
const MAX_DOCUMENTED_ORDER: usize = 500;
const MAX_ORDER: usize = 1024;

mod numerical;
use numerical::design_butterworth;

fn butter_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    butter_error_with_message(error.message, error)
}

fn butter_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    butter_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn butter_error_with_source(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
    source: RuntimeError,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn butter_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "butter",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::signal::butter"
)]
async fn butter_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    n: Value,
    wn: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let mut cooperation = SignalCooperation::new(runtime.as_ref(), BUILTIN_NAME);
    evaluate_with_cooperation(n, wn, rest.view(), &mut cooperation).await
}

async fn evaluate_with_cooperation(
    n: Value,
    wn: Value,
    rest: crate::call::FunctionArgumentView<'_>,
    cooperation: &mut SignalCooperation,
) -> crate::BuiltinResult<Value> {
    cooperation.completed(0).await?;
    let out_count = crate::output_count::current_output_count();
    let args = ButterArgs::parse(n, wn, rest).await?;
    let design = design_butterworth(&args, cooperation).await?;

    let output = if let Some(out_count) = out_count {
        match out_count {
            0 => Value::output_list(Vec::new()),
            1 => Value::output_list(vec![design.numerator_value()?]),
            2 => Value::output_list(vec![design.numerator_value()?, design.denominator_value()?]),
            3 => {
                let (z, p, k) = design.zpk_values()?;
                Value::output_list(vec![z, p, k])
            }
            4 => {
                let (a, b, c, d) = design.state_space_values(cooperation).await?;
                Value::output_list(vec![a, b, c, d])
            }
            _ => return Err(butter_error(&BUTTER_ERROR_TOO_MANY_OUTPUTS)),
        }
    } else {
        design.numerator_value()?
    };

    // Every result adapter may allocate and the state-space converter does
    // bounded synchronous work. Check the same invocation immediately before
    // any completed Value is handed back for publication.
    cooperation.completed(0).await?;
    Ok(output)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FilterKind {
    Lowpass,
    Highpass,
    Bandpass,
    Bandstop,
}

#[derive(Debug, Clone)]
struct ButterArgs {
    order: usize,
    cutoff: Vec<f64>,
    kind: FilterKind,
    analog: bool,
}

impl ButterArgs {
    async fn parse(
        n: Value,
        wn: Value,
        rest: crate::call::FunctionArgumentView<'_>,
    ) -> BuiltinResult<Self> {
        if rest.len() > 2 {
            return Err(butter_error(&BUTTER_ERROR_ARG_COUNT));
        }

        ensure_input_extensions(&n, &wn, rest)?;
        let order = parse_order(gather_value(&n).await?)?;
        if order > MAX_DOCUMENTED_ORDER {
            crate::compatibility::ensure_builtin_extension_enabled(
                &BUTTER_LARGE_ORDER_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
        let cutoff = parse_cutoff(gather_value(&wn).await?)?;
        let mut analog = false;
        let mut kind = None;

        for value in rest {
            let token = parse_option_token(gather_value(value).await?)?;
            if is_runmat_option_alias(&token) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &BUTTER_OPTION_ALIAS_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            match classify_option_token(&token)? {
                OptionToken::Analog => {
                    if analog {
                        return Err(butter_error_with_detail(
                            &BUTTER_ERROR_INVALID_OPTION,
                            "analog flag was provided more than once",
                        ));
                    }
                    analog = true;
                }
                OptionToken::Kind(parsed_kind) => {
                    if kind.is_some() {
                        return Err(butter_error_with_detail(
                            &BUTTER_ERROR_INVALID_OPTION,
                            "filter type was provided more than once",
                        ));
                    }
                    kind = Some(parsed_kind);
                }
            }
        }

        let kind = match kind {
            Some(kind) => kind,
            None if cutoff.len() == 1 => FilterKind::Lowpass,
            None if cutoff.len() == 2 => FilterKind::Bandpass,
            None => {
                return Err(butter_error_with_detail(
                    &BUTTER_ERROR_INVALID_FREQUENCY,
                    "Wn must be a scalar or a two-element vector",
                ));
            }
        };

        validate_cutoff(&cutoff, kind, analog)?;

        Ok(Self {
            order,
            cutoff,
            kind,
            analog,
        })
    }
}

fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn is_logical_value(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
}

fn is_single_value(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
        || matches!(value, Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_integer_type(handle).is_none()
                && !runmat_accelerate_api::handle_is_logical(handle)
                && runmat_accelerate_api::handle_precision(handle)
                    == Some(runmat_accelerate_api::ProviderPrecision::F32))
}

fn ensure_input_extensions(
    n: &Value,
    wn: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    for (value, integer, logical, single) in [
        (
            n,
            &BUTTER_INTEGER_ORDER_EXTENSION,
            &BUTTER_LOGICAL_ORDER_EXTENSION,
            &BUTTER_SINGLE_ORDER_EXTENSION,
        ),
        (
            wn,
            &BUTTER_INTEGER_CUTOFF_EXTENSION,
            &BUTTER_LOGICAL_CUTOFF_EXTENSION,
            &BUTTER_SINGLE_CUTOFF_EXTENSION,
        ),
    ] {
        if is_typed_integer_value(value) {
            crate::compatibility::ensure_builtin_extension_enabled(integer, BUILTIN_NAME)?;
        }
        if is_logical_value(value) {
            crate::compatibility::ensure_builtin_extension_enabled(logical, BUILTIN_NAME)?;
        }
        if is_single_value(value) {
            crate::compatibility::ensure_builtin_extension_enabled(single, BUILTIN_NAME)?;
        }
    }
    if std::iter::once(n)
        .chain(std::iter::once(wn))
        .chain(rest.iter())
        .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &BUTTER_GPU_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum OptionToken {
    Analog,
    Kind(FilterKind),
}

async fn gather_value(value: &Value) -> BuiltinResult<Value> {
    gpu_helpers::gather_value_async(value).await.map_err(|err| {
        butter_error_with_source(
            &BUTTER_ERROR_INTERNAL,
            "failed to gather GPU argument",
            map_control_flow_with_builtin(err, BUILTIN_NAME),
        )
    })
}

fn parse_order(value: Value) -> BuiltinResult<usize> {
    if let Some(value) = tensor::scalar_integer_value(&value) {
        let order = value.try_to_usize().ok_or_else(|| {
            butter_error_with_detail(
                &BUTTER_ERROR_INVALID_ORDER,
                "order is outside the supported host range",
            )
        })?;
        return validate_integer_order(order);
    }
    let raw = match value {
        Value::Num(value) => value,
        Value::Bool(value) => {
            if value {
                1.0
            } else {
                0.0
            }
        }
        Value::Tensor(tensor) if tensor::is_scalar_tensor(&tensor) => {
            tensor::tensor_value_f64(&tensor, 0)
        }
        Value::LogicalArray(logical) if logical.data.len() == 1 => {
            if logical.data[0] != 0 {
                1.0
            } else {
                0.0
            }
        }
        other => {
            return Err(butter_error_with_detail(
                &BUTTER_ERROR_INVALID_ORDER,
                format!("received {other:?}"),
            ));
        }
    };

    if !raw.is_finite() {
        return Err(butter_error_with_detail(
            &BUTTER_ERROR_INVALID_ORDER,
            "order must be finite",
        ));
    }
    let rounded = raw.round();
    if (rounded - raw).abs() > EPS || rounded < 1.0 {
        return Err(butter_error_with_detail(
            &BUTTER_ERROR_INVALID_ORDER,
            "order must be an integer greater than or equal to 1",
        ));
    }
    if rounded > MAX_ORDER as f64 {
        return Err(butter_error_with_detail(
            &BUTTER_ERROR_INVALID_ORDER,
            format!("order exceeds RunMat's resource limit of {MAX_ORDER}"),
        ));
    }
    Ok(rounded as usize)
}

fn validate_integer_order(order: usize) -> BuiltinResult<usize> {
    if order == 0 {
        return Err(butter_error_with_detail(
            &BUTTER_ERROR_INVALID_ORDER,
            "order must be an integer greater than or equal to 1",
        ));
    }
    if order > MAX_ORDER {
        return Err(butter_error_with_detail(
            &BUTTER_ERROR_INVALID_ORDER,
            format!("order exceeds RunMat's resource limit of {MAX_ORDER}"),
        ));
    }
    Ok(order)
}

fn parse_cutoff(value: Value) -> BuiltinResult<Vec<f64>> {
    let values = match value {
        Value::Num(value) => vec![value],
        Value::Int(value) => vec![value.to_f64()],
        Value::Bool(value) => vec![if value { 1.0 } else { 0.0 }],
        Value::Tensor(tensor) => {
            ensure_vector_shape("Wn", tensor.shape())?;
            tensor::tensor_into_values_f64(tensor)
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical).map_err(|detail| {
                butter_error_with_detail(&BUTTER_ERROR_INVALID_FREQUENCY, detail)
            })?;
            ensure_vector_shape("Wn", tensor.shape())?;
            tensor::tensor_into_values_f64(tensor)
        }
        other => {
            return Err(butter_error_with_detail(
                &BUTTER_ERROR_INVALID_FREQUENCY,
                format!("received {other:?}"),
            ));
        }
    };

    if values.is_empty() || values.len() > 2 {
        return Err(butter_error_with_detail(
            &BUTTER_ERROR_INVALID_FREQUENCY,
            "Wn must be a scalar or a two-element vector",
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(butter_error_with_detail(
            &BUTTER_ERROR_INVALID_FREQUENCY,
            "cutoff frequencies must be finite",
        ));
    }
    Ok(values)
}

fn parse_option_token(value: Value) -> BuiltinResult<String> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => Ok(text.into_string_unbounded()),
        Value::CharArray(chars) => {
            runmat_value::decode_utf16_code_units(chars.data()).map_err(|source| {
                let mut error = butter_error(&BUTTER_ERROR_INVALID_OPTION);
                error.source = Some(Box::new(source));
                error
            })
        }
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Ok(array.data()[0]
                .text_ref()
                .expect("checked nonmissing scalar string")
                .as_str()
                .to_owned())
        }
        other => Err(butter_error_with_detail(
            &BUTTER_ERROR_INVALID_OPTION,
            format!("expected a string option, received {other:?}"),
        )),
    }
}

fn classify_option_token(token: &str) -> BuiltinResult<OptionToken> {
    match token.trim().to_ascii_lowercase().as_str() {
        "s" => Ok(OptionToken::Analog),
        "low" | "lowpass" => Ok(OptionToken::Kind(FilterKind::Lowpass)),
        "high" | "highpass" => Ok(OptionToken::Kind(FilterKind::Highpass)),
        "bandpass" | "pass" => Ok(OptionToken::Kind(FilterKind::Bandpass)),
        "stop" | "bandstop" | "bandreject" | "notch" => Ok(OptionToken::Kind(FilterKind::Bandstop)),
        other => Err(butter_error_with_detail(
            &BUTTER_ERROR_INVALID_OPTION,
            format!("unrecognized option '{other}'"),
        )),
    }
}

fn is_runmat_option_alias(token: &str) -> bool {
    matches!(
        token.trim().to_ascii_lowercase().as_str(),
        "lowpass" | "highpass" | "pass" | "bandstop" | "bandreject" | "notch"
    )
}

fn validate_cutoff(cutoff: &[f64], kind: FilterKind, analog: bool) -> BuiltinResult<()> {
    let expected_len = match kind {
        FilterKind::Lowpass | FilterKind::Highpass => 1,
        FilterKind::Bandpass | FilterKind::Bandstop => 2,
    };
    if cutoff.len() != expected_len {
        return Err(butter_error_with_detail(
            &BUTTER_ERROR_INVALID_FREQUENCY,
            match kind {
                FilterKind::Lowpass | FilterKind::Highpass => {
                    "lowpass and highpass designs require scalar Wn"
                }
                FilterKind::Bandpass | FilterKind::Bandstop => {
                    "bandpass and bandstop designs require two-element Wn"
                }
            },
        ));
    }

    if cutoff.iter().any(|value| *value <= 0.0) {
        return Err(butter_error_with_detail(
            &BUTTER_ERROR_INVALID_FREQUENCY,
            "cutoff frequencies must be greater than zero",
        ));
    }
    if !analog && cutoff.iter().any(|value| *value >= 1.0) {
        return Err(butter_error_with_detail(
            &BUTTER_ERROR_INVALID_FREQUENCY,
            "digital cutoff frequencies must be less than 1.0, where 1.0 is Nyquist",
        ));
    }
    if cutoff.len() == 2 && cutoff[0] >= cutoff[1] {
        return Err(butter_error_with_detail(
            &BUTTER_ERROR_INVALID_FREQUENCY,
            "two-element Wn must be strictly increasing",
        ));
    }
    Ok(())
}

fn ensure_vector_shape(label: &str, shape: &[usize]) -> BuiltinResult<()> {
    if shape.is_empty() {
        return Ok(());
    }
    let non_singletons = shape.iter().filter(|&&dim| dim != 1).count();
    if non_singletons > 1 {
        return Err(butter_error_with_detail(
            &BUTTER_ERROR_INVALID_FREQUENCY,
            format!("{label} must be a vector"),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
