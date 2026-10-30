//! MATLAB-compatible `pulstran` builtin for sampled pulse trains.

use runmat_builtins::{
    BuiltinErrorDescriptor, PULSTRAN_ERROR_ARG_COUNT, PULSTRAN_ERROR_INTERNAL,
    PULSTRAN_ERROR_INVALID_DELAY, PULSTRAN_ERROR_INVALID_INPUT, PULSTRAN_ERROR_INVALID_PARAMETER,
    PULSTRAN_ERROR_INVALID_PULSE, PULSTRAN_EXPLICIT_GPU_EXTENSION, PULSTRAN_INTEGER_D_EXTENSION,
    PULSTRAN_INTEGER_PARAMETER_EXTENSION, PULSTRAN_INTEGER_P_EXTENSION,
    PULSTRAN_INTEGER_T_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

use crate::builtins::common::tensor::{scalar_f64_from_host_value, tensor_into_value};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::signal::gauspuls::{
    default_params as gauspuls_default_params, gauspuls_scalar, validate_params, GauspulsParams,
};
use crate::builtins::math::signal::rectpuls::{
    rectpuls_scalar, validate_width as validate_rect_width,
};
use crate::builtins::math::signal::tripuls::{
    tripuls_scalar, validate_skew as validate_tripuls_skew,
    validate_width as validate_tripuls_width,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "pulstran";

#[derive(Clone, Copy, Debug)]
struct PulseInstance {
    delay: f64,
    amplitude: f64,
}

#[derive(Clone, Debug)]
struct PulseSource {
    waveform: PulseWaveform,
    arguments: crate::call::BuiltinArguments,
}

#[derive(Clone, Debug)]
enum PulseWaveform {
    Rect { width: f64 },
    Tri { width: f64, skew: f64 },
    Gaus { params: GauspulsParams },
    Callback(Value),
    Prototype { samples: Vec<f64>, fs: f64 },
}

enum PulseSelection {
    Prototype,
    Rect,
    Tri,
    Gaus,
    NamedCallback(String),
    HandleCallback,
    Invalid(Box<RuntimeError>),
}

impl PulseSelection {
    fn uses_numeric_controls(&self) -> bool {
        matches!(self, Self::Prototype | Self::Rect | Self::Tri | Self::Gaus)
    }
}

fn pulstran_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    pulstran_error_with_message(error.message, error)
}

fn pulstran_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    pulstran_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn pulstran_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn pulstran_error_with_source(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
    source: RuntimeError,
) -> RuntimeError {
    let retry = source.gpu_gather_retry;
    let span = source.span;
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME)
        .with_source(source)
        .with_gpu_gather_retry(retry);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    let mut owner = builder.build();
    owner.span = span;
    owner
}

#[runtime_builtin(
    name = "pulstran",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::signal::pulstran"
)]
async fn pulstran_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    t: Value,
    d: Value,
    pulse: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let selection = decode_pulse_selection(&pulse);
    ensure_pulstran_extensions(&t, &d, &pulse, rest.view(), &selection).await?;
    let gpu_operands = pulstran_gpu_operands(&t, &d, &pulse, rest.view(), &selection);
    let gpu_source =
        gpu_helpers::select_resident_output_source(gpu_operands.iter().cloned(), BUILTIN_NAME)?;
    let t = real_tensor_arg(t, &PULSTRAN_ERROR_INVALID_INPUT).await?;
    let delays = parse_delays(real_tensor_arg(d, &PULSTRAN_ERROR_INVALID_DELAY).await?)?;
    let source = parse_pulse_source(selection, pulse, rest).await?;
    let y = evaluate_pulse_train(runtime.as_ref(), &t, &delays, &source).await?;
    let Some(source) = gpu_source else {
        return Ok(tensor_into_value(y));
    };
    let borrowed = gpu_operands.iter().collect::<Vec<_>>();
    let restored = gpu_helpers::restore_class_preserving_value_with_operands(
        &source,
        Value::Tensor(y),
        BUILTIN_NAME,
        gpu_helpers::RestoreOperandContext::new(&borrowed, &borrowed),
    )?;
    if gpu_operands
        .iter()
        .any(runmat_accelerate_api::handle_is_explicit)
        && !matches!(restored, Value::GpuTensor(_))
    {
        return Err(pulstran_error_with_detail(
            &PULSTRAN_ERROR_INTERNAL,
            "provider cannot preserve explicit gpuArray output residency",
        ));
    }
    Ok(restored)
}

async fn ensure_pulstran_extensions(
    t: &Value,
    d: &Value,
    pulse: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
    selection: &PulseSelection,
) -> BuiltinResult<()> {
    reject_nonreal_operand(t, &PULSTRAN_ERROR_INVALID_INPUT)?;
    reject_nonreal_operand(d, &PULSTRAN_ERROR_INVALID_DELAY)?;
    if matches!(selection, PulseSelection::Prototype) {
        reject_nonreal_operand(pulse, &PULSTRAN_ERROR_INVALID_PULSE)?;
    }
    if selection.uses_numeric_controls() {
        for value in rest {
            reject_nonreal_operand(value, &PULSTRAN_ERROR_INVALID_PARAMETER)?;
        }
    }
    preflight_computation_shapes(d, pulse, rest, selection)?;
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        t,
        &PULSTRAN_INTEGER_T_EXTENSION,
        BUILTIN_NAME,
        "T",
    )
    .await?;
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        d,
        &PULSTRAN_INTEGER_D_EXTENSION,
        BUILTIN_NAME,
        "D",
    )
    .await?;
    let sampled = matches!(selection, PulseSelection::Prototype);
    if sampled && crate::builtins::common::validation::value_has_native_integer_class(pulse) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &PULSTRAN_INTEGER_P_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if selection.uses_numeric_controls()
        && rest
            .iter()
            .any(|value| crate::builtins::common::validation::value_has_native_integer_class(value))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &PULSTRAN_INTEGER_PARAMETER_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if pulstran_resident_values(t, d, pulse, rest, selection).any(|value| {
        matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle))
    }) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &PULSTRAN_EXPLICIT_GPU_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

/// Known-invalid role geometry must win before any earlier resident operand's
/// exactness scan or download. This mirrors the eventual host decoders without
/// reading payloads or reclassifying callback extra arguments as controls.
fn preflight_computation_shapes(
    d: &Value,
    pulse: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
    selection: &PulseSelection,
) -> BuiltinResult<()> {
    if let Some(shape) = computation_shape(d) {
        let empty = shape.contains(&0);
        let valid = empty
            || (shape.len() <= 2 && {
                let len = shape
                    .iter()
                    .try_fold(1usize, |count, &dim| count.checked_mul(dim));
                let rows = shape.first().copied().unwrap_or(1);
                let cols = shape.get(1).copied().unwrap_or(1);
                rows == 1 || cols == 1 || len == Some(1) || cols == 2
            });
        if !valid {
            return Err(pulstran_error(&PULSTRAN_ERROR_INVALID_DELAY));
        }
    }
    if matches!(selection, PulseSelection::Prototype)
        && computation_shape(pulse).is_some_and(|shape| !is_vector_shape(shape))
    {
        return Err(pulstran_error_with_detail(
            &PULSTRAN_ERROR_INVALID_PULSE,
            "sampled prototype must be a vector",
        ));
    }
    let controls = match selection {
        PulseSelection::Prototype | PulseSelection::Rect => 1,
        PulseSelection::Tri => 2,
        PulseSelection::Gaus => 3,
        _ => 0,
    };
    for value in rest.iter().take(controls) {
        if computation_shape(value).is_some_and(|shape| {
            shape
                .iter()
                .try_fold(1usize, |count, &dim| count.checked_mul(dim))
                != Some(1)
        }) {
            return Err(pulstran_error_with_detail(
                &PULSTRAN_ERROR_INVALID_PARAMETER,
                "numeric pulse control must be scalar",
            ));
        }
    }
    Ok(())
}

fn computation_shape(value: &Value) -> Option<&[usize]> {
    match value {
        Value::Tensor(tensor) => Some(tensor.shape()),
        Value::LogicalArray(array) => Some(&array.shape),
        Value::GpuTensor(handle) => Some(handle.shape()),
        _ => None,
    }
}

fn decode_pulse_selection(pulse: &Value) -> PulseSelection {
    if is_numeric_or_gpu(pulse) {
        return PulseSelection::Prototype;
    }
    let name = match text_scalar(pulse) {
        Ok(text) => text,
        Err(error) => return PulseSelection::Invalid(Box::new(error)),
    }
    .or_else(|| match pulse {
        Value::FunctionHandle(name) | Value::ExternalFunctionHandle(name) => {
            Some(name.name().to_owned())
        }
        _ => None,
    });
    if let Some(name) = name {
        let name = match normalize_pulse_name(&name) {
            Ok(name) => name,
            Err(error) => return PulseSelection::Invalid(Box::new(error)),
        };
        return match name.to_ascii_lowercase().as_str() {
            "rectpuls" => PulseSelection::Rect,
            "tripuls" => PulseSelection::Tri,
            "gauspuls" => PulseSelection::Gaus,
            _ => PulseSelection::NamedCallback(name),
        };
    }
    match pulse {
        Value::BoundFunctionHandle(_) | Value::MethodFunctionHandle(_) => {
            PulseSelection::HandleCallback
        }
        _ => PulseSelection::Invalid(Box::new(pulstran_error(&PULSTRAN_ERROR_INVALID_PULSE))),
    }
}

fn pulstran_resident_values<'a>(
    t: &'a Value,
    d: &'a Value,
    pulse: &'a Value,
    rest: crate::call::FunctionArgumentView<'a>,
    selection: &PulseSelection,
) -> impl Iterator<Item = &'a Value> {
    let uses_controls = selection.uses_numeric_controls();
    std::iter::once(t)
        .chain(std::iter::once(d))
        .chain(std::iter::once(pulse).filter(|value| is_numeric_or_gpu(value)))
        .chain(rest.into_iter().filter(move |_| uses_controls))
}

fn pulstran_gpu_operands(
    t: &Value,
    d: &Value,
    pulse: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
    selection: &PulseSelection,
) -> Vec<runmat_accelerate_api::GpuTensorHandle> {
    pulstran_resident_values(t, d, pulse, rest, selection)
        .filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        })
        .collect()
}

fn reject_nonreal_operand(
    value: &Value,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<()> {
    let complex = matches!(value, Value::Complex(_, _) | Value::ComplexTensor(_))
        || matches!(value, Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_storage(handle)
                == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved);
    if complex {
        return Err(pulstran_error_with_detail(
            error,
            "real numeric data required; complex storage is not supported",
        ));
    }
    Ok(())
}

async fn real_tensor_arg(
    value: Value,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<Tensor> {
    reject_nonreal_operand(&value, error)?;
    match value {
        Value::GpuTensor(handle) => {
            gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(|source| {
                    pulstran_error_with_source(
                        &PULSTRAN_ERROR_INTERNAL,
                        "gpu gather failed",
                        source,
                    )
                })
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(pulstran_error(error)),
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
            Err(pulstran_error(error))
        }
        other => tensor::value_into_tensor_for(BUILTIN_NAME, other)
            .map_err(|err| pulstran_error_with_detail(error, err)),
    }
}

fn parse_delays(tensor: Tensor) -> BuiltinResult<Vec<PulseInstance>> {
    if tensor.is_empty() {
        return Ok(Vec::new());
    }
    let values = tensor::tensor_values_f64(&tensor);
    let shape = tensor.shape();
    if shape.len() <= 2 {
        let rows = shape.first().copied().unwrap_or(values.len());
        let cols = shape.get(1).copied().unwrap_or(1);
        let is_vector = rows == 1 || cols == 1 || values.len() == 1;
        if !is_vector && cols == 2 {
            return Ok((0..rows)
                .map(|row| PulseInstance {
                    delay: values[row],
                    amplitude: values[row + rows],
                })
                .collect());
        }
        if is_vector {
            return Ok(values
                .into_iter()
                .map(|delay| PulseInstance {
                    delay,
                    amplitude: 1.0,
                })
                .collect());
        }
    }
    Err(pulstran_error(&PULSTRAN_ERROR_INVALID_DELAY))
}

async fn parse_pulse_source(
    selection: PulseSelection,
    pulse: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<PulseSource> {
    let arguments = rest.fork();
    let waveform = match selection {
        PulseSelection::Prototype => parse_sampled_prototype(pulse, rest).await,
        PulseSelection::Rect => Ok(PulseWaveform::Rect {
            width: parse_rect_width(rest.view()).await?,
        }),
        PulseSelection::Tri => {
            let (width, skew) = parse_tripuls_options(rest.view()).await?;
            Ok(PulseWaveform::Tri { width, skew })
        }
        PulseSelection::Gaus => Ok(PulseWaveform::Gaus {
            params: parse_gauspuls_params(rest.view()).await?,
        }),
        PulseSelection::NamedCallback(name) => {
            Ok(PulseWaveform::Callback(Value::function_handle(name)))
        }
        PulseSelection::HandleCallback => Ok(PulseWaveform::Callback(pulse)),
        PulseSelection::Invalid(error) => Err(*error),
    }?;
    Ok(PulseSource {
        waveform,
        arguments,
    })
}

async fn parse_sampled_prototype(
    pulse: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<PulseWaveform> {
    if rest.len() > 1 {
        return Err(pulstran_error_with_detail(
            &PULSTRAN_ERROR_ARG_COUNT,
            format!("got {}", rest.len() + 3),
        ));
    }
    reject_nonreal_operand(&pulse, &PULSTRAN_ERROR_INVALID_PULSE)?;
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        &pulse,
        &PULSTRAN_INTEGER_P_EXTENSION,
        BUILTIN_NAME,
        "prototype",
    )
    .await?;
    let prototype = real_tensor_arg(pulse, &PULSTRAN_ERROR_INVALID_PULSE).await?;
    if !is_vector_shape(prototype.shape()) {
        return Err(pulstran_error_with_detail(
            &PULSTRAN_ERROR_INVALID_PULSE,
            "sampled prototype must be a vector",
        ));
    }
    let fs = match rest.first() {
        Some(value) => {
            reject_nonreal_operand(value, &PULSTRAN_ERROR_INVALID_PARAMETER)?;
            crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
                value,
                &PULSTRAN_INTEGER_PARAMETER_EXTENSION,
                BUILTIN_NAME,
                "FS",
            )
            .await?;
            let raw = scalar_control(value, "FS")
                .await?
                .ok_or_else(|| pulstran_error(&PULSTRAN_ERROR_INVALID_PARAMETER))?;
            if !raw.is_finite() || raw <= 0.0 {
                return Err(pulstran_error_with_detail(
                    &PULSTRAN_ERROR_INVALID_PARAMETER,
                    format!("sample rate must be positive and finite, got {raw}"),
                ));
            }
            raw
        }
        None => 1.0,
    };
    Ok(PulseWaveform::Prototype {
        samples: tensor::tensor_into_values_f64(prototype),
        fs,
    })
}

fn is_vector_shape(shape: &[usize]) -> bool {
    match shape {
        [] => true,
        [_] => true,
        [rows, cols] => *rows == 1 || *cols == 1,
        _ => false,
    }
}

fn is_numeric_or_gpu(value: &Value) -> bool {
    matches!(
        value,
        Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::Tensor(_)
            | Value::LogicalArray(_)
            | Value::GpuTensor(_)
    )
}

fn text_scalar(value: &Value) -> crate::BuiltinResult<Option<String>> {
    Ok(match value {
        Value::String(runmat_value::StringElement::Text(text)) => Some(text.as_str().to_owned()),
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Some(
                array.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string")
                    .as_str()
                    .to_owned(),
            )
        }
        Value::CharArray(array) if array.rows == 1 => Some(
            runmat_value::decode_utf16_code_units(array.data()).map_err(|source| {
                let mut error = pulstran_error(&PULSTRAN_ERROR_INVALID_PULSE);
                error.source = Some(Box::new(source));
                error
            })?,
        ),
        _ => None,
    })
}

fn normalize_pulse_name(name: &str) -> BuiltinResult<String> {
    let trimmed = name.trim();
    let trimmed = trimmed.strip_prefix('@').unwrap_or(trimmed).trim();
    if trimmed.is_empty() {
        Err(pulstran_error_with_detail(
            &PULSTRAN_ERROR_INVALID_PULSE,
            "pulse function name must not be empty",
        ))
    } else {
        Ok(trimmed.to_string())
    }
}

async fn parse_rect_width(rest: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<f64> {
    match rest.len() {
        0 => Ok(1.0),
        1 => {
            let raw = scalar_arg(&rest[0], "width").await?;
            validate_rect_width(raw)
                .map_err(|err| pulstran_error_with_detail(&PULSTRAN_ERROR_INVALID_PARAMETER, err))
        }
        _ => Err(pulstran_error_with_detail(
            &PULSTRAN_ERROR_ARG_COUNT,
            format!("rectpuls got {}", rest.len() + 1),
        )),
    }
}

async fn parse_tripuls_options(
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<(f64, f64)> {
    if rest.len() > 2 {
        return Err(pulstran_error_with_detail(
            &PULSTRAN_ERROR_ARG_COUNT,
            format!("tripuls got {}", rest.len() + 1),
        ));
    }
    let width = match rest.first() {
        Some(value) => validate_tripuls_width(scalar_arg(value, "width").await?)
            .map_err(|err| pulstran_error_with_detail(&PULSTRAN_ERROR_INVALID_PARAMETER, err))?,
        None => 1.0,
    };
    let skew = match rest.get(1) {
        Some(value) => validate_tripuls_skew(scalar_arg(value, "skew").await?)
            .map_err(|err| pulstran_error_with_detail(&PULSTRAN_ERROR_INVALID_PARAMETER, err))?,
        None => 0.0,
    };
    Ok((width, skew))
}

async fn parse_gauspuls_params(
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<GauspulsParams> {
    if rest.len() > 3 {
        return Err(pulstran_error_with_detail(
            &PULSTRAN_ERROR_ARG_COUNT,
            format!("gauspuls got {}", rest.len() + 1),
        ));
    }
    let mut params = gauspuls_default_params();
    if let Some(value) = rest.first() {
        params.fc = scalar_arg(value, "FC").await?;
    }
    if let Some(value) = rest.get(1) {
        params.bw = scalar_arg(value, "BW").await?;
    }
    if let Some(value) = rest.get(2) {
        params.bwr = scalar_arg(value, "BWR").await?;
    }
    validate_params(params)
        .map_err(|err| pulstran_error_with_detail(&PULSTRAN_ERROR_INVALID_PARAMETER, err))
}

async fn scalar_arg(value: &Value, label: &str) -> BuiltinResult<f64> {
    reject_nonreal_operand(value, &PULSTRAN_ERROR_INVALID_PARAMETER)?;
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        value,
        &PULSTRAN_INTEGER_PARAMETER_EXTENSION,
        BUILTIN_NAME,
        label,
    )
    .await?;
    scalar_control(value, label).await?.ok_or_else(|| {
        pulstran_error_with_detail(
            &PULSTRAN_ERROR_INVALID_PARAMETER,
            format!("{label}: expected scalar"),
        )
    })
}

/// Decode a numeric control without flattening provider/cancellation failures
/// into the string-only common scalar convenience API.
async fn scalar_control(value: &Value, label: &str) -> BuiltinResult<Option<f64>> {
    reject_nonreal_operand(value, &PULSTRAN_ERROR_INVALID_PARAMETER)?;
    if let Value::GpuTensor(handle) = value {
        let len = handle
            .shape()
            .iter()
            .try_fold(1usize, |count, &dimension| count.checked_mul(dimension));
        if !handle.shape().is_empty() && len != Some(1) {
            return Err(pulstran_error_with_detail(
                &PULSTRAN_ERROR_INVALID_PARAMETER,
                format!(
                    "{label}: expected scalar gpuArray, got array of size {}",
                    len.map_or_else(|| "overflow".to_string(), |count| count.to_string())
                ),
            ));
        }
        let gathered = crate::gather_if_needed_async(value)
            .await
            .map_err(|source| {
                pulstran_error_with_source(
                    &PULSTRAN_ERROR_INVALID_PARAMETER,
                    format!("{label}: scalar gather failed"),
                    source,
                )
            })?;
        scalar_f64_from_host_value(&gathered).map_err(|detail| {
            pulstran_error_with_detail(
                &PULSTRAN_ERROR_INVALID_PARAMETER,
                format!("{label}: {detail}"),
            )
        })
    } else {
        scalar_f64_from_host_value(value).map_err(|detail| {
            pulstran_error_with_detail(
                &PULSTRAN_ERROR_INVALID_PARAMETER,
                format!("{label}: {detail}"),
            )
        })
    }
}

mod evaluation;
use evaluation::evaluate_pulse_train;

#[cfg(test)]
mod tests;
