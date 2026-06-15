//! MATLAB-compatible `pulstran` builtin for sampled pulse trains.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::tensor::{scalar_f64_from_value_async, tensor_into_value};
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
use crate::builtins::math::signal::type_resolvers::pulse_train_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "pulstran";

const PULSTRAN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Pulse train sampled at T.",
}];

const PULSTRAN_INPUTS_FUNCTION: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "T",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample times.",
    },
    BuiltinParamDescriptor {
        name: "D",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Pulse delays, or an N-by-2 delay/amplitude matrix.",
    },
    BuiltinParamDescriptor {
        name: "FUN",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Pulse function name or function handle.",
    },
    BuiltinParamDescriptor {
        name: "P",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional pulse function parameters.",
    },
];

const PULSTRAN_INPUTS_PROTO: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "T",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample times.",
    },
    BuiltinParamDescriptor {
        name: "D",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Pulse delays, or an N-by-2 delay/amplitude matrix.",
    },
    BuiltinParamDescriptor {
        name: "P",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sampled prototype pulse.",
    },
    BuiltinParamDescriptor {
        name: "FS",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("1"),
        description: "Prototype sample rate.",
    },
];

const PULSTRAN_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = pulstran(T, D, FUN, P1, ...)",
        inputs: &PULSTRAN_INPUTS_FUNCTION,
        outputs: &PULSTRAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = pulstran(T, D, P, FS)",
        inputs: &PULSTRAN_INPUTS_PROTO,
        outputs: &PULSTRAN_OUTPUT,
    },
];

const PULSTRAN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PULSTRAN.INVALID_INPUT",
    identifier: Some("RunMat:pulstran:InvalidInput"),
    when: "T, D, or sampled prototype inputs are not real numeric arrays.",
    message: "pulstran: expected real numeric input",
};

const PULSTRAN_ERROR_INVALID_DELAY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PULSTRAN.INVALID_DELAY",
    identifier: Some("RunMat:pulstran:InvalidDelay"),
    when: "Delay input is not a vector or N-by-2 delay/amplitude matrix.",
    message: "pulstran: D must be a delay vector or N-by-2 delay/amplitude matrix",
};

const PULSTRAN_ERROR_INVALID_PULSE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PULSTRAN.INVALID_PULSE",
    identifier: Some("RunMat:pulstran:InvalidPulse"),
    when: "Pulse function or sampled prototype is malformed.",
    message: "pulstran: invalid pulse specification",
};

const PULSTRAN_ERROR_INVALID_PARAMETER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PULSTRAN.INVALID_PARAMETER",
    identifier: Some("RunMat:pulstran:InvalidParameter"),
    when: "Pulse function parameters or sampled prototype rate are malformed.",
    message: "pulstran: invalid pulse parameter",
};

const PULSTRAN_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PULSTRAN.ARG_COUNT",
    identifier: Some("RunMat:pulstran:ArgCount"),
    when: "Required arguments are missing or too many prototype arguments are provided.",
    message: "pulstran: expected pulstran(T, D, FUN, ...) or pulstran(T, D, P, FS)",
};

const PULSTRAN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PULSTRAN.INTERNAL",
    identifier: Some("RunMat:pulstran:InternalError"),
    when: "Internal tensor construction, callback result materialization, or GPU gather fails.",
    message: "pulstran: internal error",
};

const PULSTRAN_ERRORS: [BuiltinErrorDescriptor; 6] = [
    PULSTRAN_ERROR_INVALID_INPUT,
    PULSTRAN_ERROR_INVALID_DELAY,
    PULSTRAN_ERROR_INVALID_PULSE,
    PULSTRAN_ERROR_INVALID_PARAMETER,
    PULSTRAN_ERROR_ARG_COUNT,
    PULSTRAN_ERROR_INTERNAL,
];

pub const PULSTRAN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PULSTRAN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &PULSTRAN_ERRORS,
};

#[derive(Clone, Copy, Debug)]
struct PulseInstance {
    delay: f64,
    amplitude: f64,
}

#[derive(Clone, Debug)]
enum PulseSource {
    Rect { width: f64 },
    Tri { width: f64, skew: f64 },
    Gaus { params: GauspulsParams },
    Callback { handle: Value, args: Vec<Value> },
    Prototype { samples: Vec<f64>, fs: f64 },
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
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "pulstran",
    category = "math/signal",
    summary = "Generate pulse trains from pulse functions or sampled prototypes.",
    keywords = "pulstran,pulse train,rectpuls,tripuls,gauspuls,signal processing",
    type_resolver(pulse_train_type),
    descriptor(crate::builtins::math::signal::pulstran::PULSTRAN_DESCRIPTOR),
    builtin_path = "crate::builtins::math::signal::pulstran"
)]
async fn pulstran_builtin(
    t: Value,
    d: Value,
    pulse: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let t = real_tensor_arg(t, &PULSTRAN_ERROR_INVALID_INPUT).await?;
    let delays = parse_delays(real_tensor_arg(d, &PULSTRAN_ERROR_INVALID_DELAY).await?)?;
    let source = parse_pulse_source(pulse, rest).await?;
    let y = evaluate_pulse_train(&t, &delays, &source).await?;
    Ok(tensor_into_value(y))
}

async fn real_tensor_arg(
    value: Value,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<Tensor> {
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
    if tensor.data.is_empty() {
        return Ok(Vec::new());
    }
    let shape = tensor.shape.as_slice();
    if shape.len() <= 2 {
        let rows = shape.first().copied().unwrap_or(tensor.data.len());
        let cols = shape.get(1).copied().unwrap_or(1);
        let is_vector = rows == 1 || cols == 1 || tensor.data.len() == 1;
        if !is_vector && cols == 2 {
            return Ok((0..rows)
                .map(|row| PulseInstance {
                    delay: tensor.data[row],
                    amplitude: tensor.data[row + rows],
                })
                .collect());
        }
        if is_vector {
            return Ok(tensor
                .data
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

async fn parse_pulse_source(pulse: Value, rest: Vec<Value>) -> BuiltinResult<PulseSource> {
    if is_numeric_or_gpu(&pulse) {
        return parse_sampled_prototype(pulse, rest).await;
    }
    if let Some(name) = text_scalar(&pulse) {
        let handle_name = normalize_pulse_name(&name)?;
        return parse_named_pulse_or_callback(handle_name, rest).await;
    }
    match pulse {
        Value::FunctionHandle(name) | Value::ExternalFunctionHandle(name) => {
            let handle_name = normalize_pulse_name(&name)?;
            parse_named_pulse_or_callback(handle_name, rest).await
        }
        Value::BoundFunctionHandle { .. } | Value::MethodFunctionHandle(_) => {
            Ok(PulseSource::Callback {
                handle: pulse,
                args: rest,
            })
        }
        _ => Err(pulstran_error(&PULSTRAN_ERROR_INVALID_PULSE)),
    }
}

async fn parse_named_pulse_or_callback(
    name: String,
    rest: Vec<Value>,
) -> BuiltinResult<PulseSource> {
    match name.to_ascii_lowercase().as_str() {
        "rectpuls" => Ok(PulseSource::Rect {
            width: parse_rect_width(&rest).await?,
        }),
        "tripuls" => {
            let (width, skew) = parse_tripuls_options(&rest).await?;
            Ok(PulseSource::Tri { width, skew })
        }
        "gauspuls" => Ok(PulseSource::Gaus {
            params: parse_gauspuls_params(&rest).await?,
        }),
        _ => Ok(PulseSource::Callback {
            handle: Value::FunctionHandle(name),
            args: rest,
        }),
    }
}

async fn parse_sampled_prototype(pulse: Value, rest: Vec<Value>) -> BuiltinResult<PulseSource> {
    if rest.len() > 1 {
        return Err(pulstran_error_with_detail(
            &PULSTRAN_ERROR_ARG_COUNT,
            format!("got {}", rest.len() + 3),
        ));
    }
    let prototype = real_tensor_arg(pulse, &PULSTRAN_ERROR_INVALID_PULSE).await?;
    if !is_vector_shape(&prototype.shape) {
        return Err(pulstran_error_with_detail(
            &PULSTRAN_ERROR_INVALID_PULSE,
            "sampled prototype must be a vector",
        ));
    }
    let fs = match rest.first() {
        Some(value) => {
            let raw = scalar_f64_from_value_async(value)
                .await
                .map_err(|err| pulstran_error_with_detail(&PULSTRAN_ERROR_INVALID_PARAMETER, err))?
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
    Ok(PulseSource::Prototype {
        samples: prototype.data,
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

fn text_scalar(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Some(array.data[0].clone()),
        Value::CharArray(array) if array.rows == 1 => Some(array.data.iter().collect()),
        _ => None,
    }
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

async fn parse_rect_width(rest: &[Value]) -> BuiltinResult<f64> {
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

async fn parse_tripuls_options(rest: &[Value]) -> BuiltinResult<(f64, f64)> {
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

async fn parse_gauspuls_params(rest: &[Value]) -> BuiltinResult<GauspulsParams> {
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
    scalar_f64_from_value_async(value)
        .await
        .map_err(|err| {
            pulstran_error_with_detail(&PULSTRAN_ERROR_INVALID_PARAMETER, format!("{label}: {err}"))
        })?
        .ok_or_else(|| {
            pulstran_error_with_detail(
                &PULSTRAN_ERROR_INVALID_PARAMETER,
                format!("{label}: expected scalar"),
            )
        })
}

async fn evaluate_pulse_train(
    t: &Tensor,
    delays: &[PulseInstance],
    source: &PulseSource,
) -> BuiltinResult<Tensor> {
    let mut out = vec![0.0; t.data.len()];
    for pulse in delays {
        match source {
            PulseSource::Rect { width } => {
                for (idx, &time) in t.data.iter().enumerate() {
                    out[idx] += pulse.amplitude * rectpuls_scalar(time - pulse.delay, *width);
                }
            }
            PulseSource::Tri { width, skew } => {
                for (idx, &time) in t.data.iter().enumerate() {
                    out[idx] += pulse.amplitude * tripuls_scalar(time - pulse.delay, *width, *skew);
                }
            }
            PulseSource::Gaus { params } => {
                for (idx, &time) in t.data.iter().enumerate() {
                    out[idx] += pulse.amplitude * gauspuls_scalar(time - pulse.delay, *params);
                }
            }
            PulseSource::Prototype { samples, fs } => {
                for (idx, &time) in t.data.iter().enumerate() {
                    out[idx] += pulse.amplitude * prototype_value(samples, *fs, time - pulse.delay);
                }
            }
            PulseSource::Callback { handle, args } => {
                let shifted = shifted_time_value(t, pulse.delay)?;
                let mut call_args = Vec::with_capacity(args.len() + 1);
                call_args.push(shifted);
                call_args.extend(args.iter().cloned());
                let value = crate::call_feval_async_with_outputs(handle.clone(), &call_args, 1)
                    .await
                    .map_err(|err| {
                        pulstran_error_with_source(
                            &PULSTRAN_ERROR_INVALID_PULSE,
                            "pulse function call failed",
                            err,
                        )
                    })?;
                let samples = callback_samples(value, t.data.len()).await?;
                for (idx, sample) in samples.into_iter().enumerate() {
                    out[idx] += pulse.amplitude * sample;
                }
            }
        }
    }
    Tensor::new(out, t.shape.clone())
        .map_err(|err| pulstran_error_with_detail(&PULSTRAN_ERROR_INTERNAL, &err))
}

fn shifted_time_value(t: &Tensor, delay: f64) -> BuiltinResult<Value> {
    let data = t.data.iter().map(|value| value - delay).collect::<Vec<_>>();
    Tensor::new(data, t.shape.clone())
        .map(tensor_into_value)
        .map_err(|err| pulstran_error_with_detail(&PULSTRAN_ERROR_INTERNAL, &err))
}

async fn callback_samples(value: Value, expected_len: usize) -> BuiltinResult<Vec<f64>> {
    let tensor = real_tensor_arg(value, &PULSTRAN_ERROR_INVALID_PULSE).await?;
    if tensor.data.len() == expected_len {
        Ok(tensor.data)
    } else if tensor.data.len() == 1 {
        Ok(vec![tensor.data[0]; expected_len])
    } else {
        Err(pulstran_error_with_detail(
            &PULSTRAN_ERROR_INVALID_PULSE,
            format!(
                "pulse function returned {} samples for {expected_len} input samples",
                tensor.data.len()
            ),
        ))
    }
}

fn prototype_value(samples: &[f64], fs: f64, t: f64) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    if !t.is_finite() || samples.is_empty() {
        return 0.0;
    }
    let x = t * fs;
    if x < 0.0 || x > (samples.len() - 1) as f64 {
        return 0.0;
    }
    if samples.len() == 1 {
        return if x == 0.0 { samples[0] } else { 0.0 };
    }
    let lower = x.floor() as usize;
    if lower + 1 >= samples.len() {
        return samples[lower];
    }
    let frac = x - lower as f64;
    samples[lower] * (1.0 - frac) + samples[lower + 1] * frac
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{builtin_function_by_name, CharArray, StringArray};

    fn call(t: Value, d: Value, pulse: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(pulstran_builtin(t, d, pulse, rest))
    }

    fn expect_tensor(value: Value) -> Tensor {
        match value {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn pulstran_rectpuls_named_char_reproduces_pulse_train() {
        let t = Tensor::new(vec![-0.5, 0.0, 0.5, 1.0], vec![1, 4]).unwrap();
        let d = Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap();
        let pulse = Value::CharArray(CharArray::new_row("rectpuls"));
        let out = expect_tensor(
            call(
                Value::Tensor(t),
                Value::Tensor(d),
                pulse,
                vec![Value::Num(0.25)],
            )
            .expect("pulstran"),
        );
        assert_eq!(out.shape, vec![1, 4]);
        assert_eq!(out.data, vec![0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn pulstran_accepts_delay_amplitude_matrix() {
        let t = Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap();
        let d = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
        let pulse =
            Value::StringArray(StringArray::new(vec!["rectpuls".to_string()], vec![1, 1]).unwrap());
        let out = expect_tensor(
            call(
                Value::Tensor(t),
                Value::Tensor(d),
                pulse,
                vec![Value::Num(0.25)],
            )
            .expect("pulstran"),
        );
        assert_eq!(out.data, vec![2.0, 3.0]);
    }

    #[test]
    fn pulstran_sampled_prototype_interpolates_at_sample_rate() {
        let t = Tensor::new(vec![0.0, 0.5, 1.0, 1.5, 2.0], vec![1, 5]).unwrap();
        let d = Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap();
        let prototype = Tensor::new(vec![0.0, 1.0, 0.0], vec![1, 3]).unwrap();
        let out = expect_tensor(
            call(
                Value::Tensor(t),
                Value::Tensor(d),
                Value::Tensor(prototype),
                vec![Value::Num(2.0)],
            )
            .expect("pulstran"),
        );
        assert_eq!(out.data, vec![0.0, 1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn pulstran_rejects_invalid_delay_shape_and_bad_sample_rate() {
        let t = Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap());
        let bad_d = Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2, 1]).unwrap());
        let err = call(
            t.clone(),
            bad_d,
            Value::CharArray(CharArray::new_row("rectpuls")),
            Vec::new(),
        )
        .expect_err("delay shape");
        assert_eq!(err.identifier(), PULSTRAN_ERROR_INVALID_DELAY.identifier);

        let d = Value::Num(0.0);
        let prototype = Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap());
        let err = call(t, d, prototype, vec![Value::Num(0.0)]).expect_err("sample rate");
        assert_eq!(
            err.identifier(),
            PULSTRAN_ERROR_INVALID_PARAMETER.identifier
        );
    }

    #[test]
    fn pulstran_is_registered() {
        assert!(builtin_function_by_name("pulstran").is_some());
    }
}
