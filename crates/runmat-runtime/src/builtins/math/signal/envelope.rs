//! MATLAB-compatible `envelope` builtin for signal upper/lower envelopes.

use num_complex::Complex;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Value,
};
use runmat_macros::runtime_builtin;
use rustfft::FftPlanner;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::interpolation::pp::{
    build_spline_pp, evaluate_pp, Extrapolation, NumericSeries, QueryPoints,
};
use crate::builtins::math::signal::common::{keyword, parse_nonnegative_integer};
use crate::builtins::math::signal::type_resolvers::envelope_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "envelope";
const EPS: f64 = 1.0e-12;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::signal::envelope")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: BUILTIN_NAME,
    op_kind: GpuOpKind::Custom("signal-envelope"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Computes analytic, RMS, and peak envelopes on the CPU reference path after gathering GPU inputs.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::signal::envelope")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: BUILTIN_NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "Envelope extraction is a windowed/FFT interpolation boundary and terminates fusion plans.",
};

const ENVELOPE_OUTPUT_UPPER: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "yupper",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Upper signal envelope.",
}];

const ENVELOPE_OUTPUT_BOTH: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "yupper",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Upper signal envelope.",
    },
    BuiltinParamDescriptor {
        name: "ylower",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Lower signal envelope.",
    },
];

const ENVELOPE_INPUTS_DEFAULT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Finite real-valued signal vector or matrix, processed column-wise.",
}];

const ENVELOPE_INPUTS_METHOD: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Finite real-valued signal vector or matrix, processed column-wise.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description:
            "Filter/window length for analytic or RMS mode, or peak separation for peak mode.",
    },
    BuiltinParamDescriptor {
        name: "method",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"analytic\""),
        description: "Envelope method: `analytic`, `rms`, or `peak`.",
    },
];

const ENVELOPE_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "yupper = envelope(x)",
        inputs: &ENVELOPE_INPUTS_DEFAULT,
        outputs: &ENVELOPE_OUTPUT_UPPER,
    },
    BuiltinSignatureDescriptor {
        label: "[yupper, ylower] = envelope(x)",
        inputs: &ENVELOPE_INPUTS_DEFAULT,
        outputs: &ENVELOPE_OUTPUT_BOTH,
    },
    BuiltinSignatureDescriptor {
        label: "yupper = envelope(x, n, method)",
        inputs: &ENVELOPE_INPUTS_METHOD,
        outputs: &ENVELOPE_OUTPUT_UPPER,
    },
    BuiltinSignatureDescriptor {
        label: "[yupper, ylower] = envelope(x, n, method)",
        inputs: &ENVELOPE_INPUTS_METHOD,
        outputs: &ENVELOPE_OUTPUT_BOTH,
    },
];

const ENVELOPE_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ENVELOPE.ARG_COUNT",
    identifier: Some("RunMat:envelope:ArgCount"),
    when: "The argument count is outside supported forms.",
    message: "envelope: expected envelope(x), envelope(x, n), or envelope(x, n, method)",
};

const ENVELOPE_ERROR_INVALID_SIGNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ENVELOPE.INVALID_SIGNAL",
    identifier: Some("RunMat:envelope:InvalidSignal"),
    when: "Input is not a finite real numeric vector or matrix.",
    message: "envelope: x must be a finite real numeric vector or matrix",
};

const ENVELOPE_ERROR_INVALID_LENGTH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ENVELOPE.INVALID_LENGTH",
    identifier: Some("RunMat:envelope:InvalidLength"),
    when: "Filter/window length or peak separation is not a positive integer.",
    message: "envelope: n must be a positive integer scalar",
};

const ENVELOPE_ERROR_INVALID_METHOD: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ENVELOPE.INVALID_METHOD",
    identifier: Some("RunMat:envelope:InvalidMethod"),
    when: "Method is not analytic, rms, or peak.",
    message: "envelope: method must be 'analytic', 'rms', or 'peak'",
};

const ENVELOPE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ENVELOPE.INTERNAL",
    identifier: Some("RunMat:envelope:Internal"),
    when: "Envelope computation or output tensor assembly fails internally.",
    message: "envelope: internal error",
};

const ENVELOPE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ENVELOPE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:envelope:TooManyOutputs"),
    when: "More than two output arguments are requested.",
    message: "envelope: expected at most two output arguments",
};

const ENVELOPE_ERRORS: [BuiltinErrorDescriptor; 6] = [
    ENVELOPE_ERROR_ARG_COUNT,
    ENVELOPE_ERROR_INVALID_SIGNAL,
    ENVELOPE_ERROR_INVALID_LENGTH,
    ENVELOPE_ERROR_INVALID_METHOD,
    ENVELOPE_ERROR_INTERNAL,
    ENVELOPE_ERROR_TOO_MANY_OUTPUTS,
];

pub const ENVELOPE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ENVELOPE_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ENVELOPE_ERRORS,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EnvelopeMethod {
    Analytic,
    Rms,
    Peak,
}

#[derive(Clone, Debug)]
struct EnvelopeOptions {
    method: EnvelopeMethod,
    parameter: Option<usize>,
}

#[derive(Clone, Debug)]
struct SignalInput {
    data: Vec<f64>,
    shape: Vec<usize>,
    rows: usize,
    cols: usize,
    vector: bool,
}

#[derive(Clone, Debug)]
struct EnvelopeEvaluation {
    input: SignalInput,
    upper: Vec<f64>,
    lower: Vec<f64>,
}

fn envelope_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    envelope_error_with_message(error.message, error)
}

fn envelope_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    envelope_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn envelope_error_with_message(
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
    name = "envelope",
    category = "math/signal",
    summary = "Compute upper and lower signal envelopes.",
    keywords = "envelope,analytic signal,hilbert,rms,peak,signal processing",
    accel = "sink",
    sink = true,
    suppress_auto_output = true,
    type_resolver(envelope_type),
    descriptor(crate::builtins::math::signal::envelope::ENVELOPE_DESCRIPTOR),
    builtin_path = "crate::builtins::math::signal::envelope"
)]
async fn envelope_builtin(x: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let input = value_to_signal_input(x).await?;
    let options = parse_options(&rest)?;
    let eval = evaluate(input, &options)?;

    if crate::output_context::requested_output_count() == Some(0)
        && crate::output_count::current_output_count().is_none()
    {
        render_envelope_plot(&eval).await?;
        return Ok(Value::OutputList(Vec::new()));
    }

    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            render_envelope_plot(&eval).await?;
            return Ok(Value::OutputList(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::OutputList(vec![eval.upper_value()?]));
        }
        if out_count > 2 {
            return Err(envelope_error(&ENVELOPE_ERROR_TOO_MANY_OUTPUTS));
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![eval.upper_value()?, eval.lower_value()?],
        ));
    }

    eval.upper_value()
}

async fn value_to_signal_input(value: Value) -> BuiltinResult<SignalInput> {
    match value {
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(|err| {
                    envelope_error_with_detail(&ENVELOPE_ERROR_INVALID_SIGNAL, err.message())
                })?;
            tensor_to_signal_input(tensor)
        }
        Value::Tensor(tensor) => tensor_to_signal_input(tensor),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|err| envelope_error_with_detail(&ENVELOPE_ERROR_INVALID_SIGNAL, err))?;
            tensor_to_signal_input(tensor)
        }
        Value::Num(n) => scalar_signal_input(n),
        Value::Int(i) => scalar_signal_input(i.to_f64()),
        Value::Bool(b) => scalar_signal_input(f64::from(u8::from(b))),
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err(envelope_error(&ENVELOPE_ERROR_INVALID_SIGNAL))
        }
        _ => Err(envelope_error(&ENVELOPE_ERROR_INVALID_SIGNAL)),
    }
}

fn scalar_signal_input(value: f64) -> BuiltinResult<SignalInput> {
    if !value.is_finite() {
        return Err(envelope_error(&ENVELOPE_ERROR_INVALID_SIGNAL));
    }
    Ok(SignalInput {
        data: vec![value],
        shape: vec![1, 1],
        rows: 1,
        cols: 1,
        vector: true,
    })
}

fn tensor_to_signal_input(tensor: Tensor) -> BuiltinResult<SignalInput> {
    if tensor.data.is_empty() || tensor.data.iter().any(|value| !value.is_finite()) {
        return Err(envelope_error(&ENVELOPE_ERROR_INVALID_SIGNAL));
    }
    let shape = tensor::default_shape_for(&tensor.shape, tensor.data.len());
    if shape.len() > 2 || shape.iter().filter(|&&dim| dim > 1).count() > 2 {
        return Err(envelope_error(&ENVELOPE_ERROR_INVALID_SIGNAL));
    }
    let rows = shape.first().copied().unwrap_or(tensor.data.len());
    let cols = shape.get(1).copied().unwrap_or(1);
    if rows == 0 || cols == 0 || rows.checked_mul(cols) != Some(tensor.data.len()) {
        return Err(envelope_error(&ENVELOPE_ERROR_INVALID_SIGNAL));
    }
    Ok(SignalInput {
        data: tensor.data,
        shape,
        rows,
        cols,
        vector: rows == 1 || cols == 1,
    })
}

fn parse_options(args: &[Value]) -> BuiltinResult<EnvelopeOptions> {
    match args.len() {
        0 => Ok(EnvelopeOptions {
            method: EnvelopeMethod::Analytic,
            parameter: None,
        }),
        1 | 2 => {
            let parameter = parse_positive_integer("n", &args[0])?;
            let method = if args.len() == 2 {
                parse_method(&args[1])?
            } else {
                EnvelopeMethod::Analytic
            };
            Ok(EnvelopeOptions {
                method,
                parameter: Some(parameter),
            })
        }
        _ => Err(envelope_error(&ENVELOPE_ERROR_ARG_COUNT)),
    }
}

fn parse_positive_integer(label: &str, value: &Value) -> BuiltinResult<usize> {
    let parsed = parse_nonnegative_integer(BUILTIN_NAME, label, value)
        .map_err(|_| envelope_error(&ENVELOPE_ERROR_INVALID_LENGTH))?;
    if parsed == 0 {
        return Err(envelope_error(&ENVELOPE_ERROR_INVALID_LENGTH));
    }
    Ok(parsed)
}

fn parse_method(value: &Value) -> BuiltinResult<EnvelopeMethod> {
    match keyword(value).as_deref() {
        Some("analytic") => Ok(EnvelopeMethod::Analytic),
        Some("rms") => Ok(EnvelopeMethod::Rms),
        Some("peak") => Ok(EnvelopeMethod::Peak),
        _ => Err(envelope_error(&ENVELOPE_ERROR_INVALID_METHOD)),
    }
}

fn evaluate(input: SignalInput, options: &EnvelopeOptions) -> BuiltinResult<EnvelopeEvaluation> {
    let mut upper = vec![0.0; input.data.len()];
    let mut lower = vec![0.0; input.data.len()];
    for channel in input.channels() {
        let (up, lo) = match options.method {
            EnvelopeMethod::Analytic => analytic_envelope(channel.values, options.parameter)?,
            EnvelopeMethod::Rms => rms_envelope(channel.values, options.parameter.unwrap_or(1)),
            EnvelopeMethod::Peak => peak_envelope(channel.values, options.parameter.unwrap_or(1))?,
        };
        for (offset, (up_value, lo_value)) in up.into_iter().zip(lo).enumerate() {
            upper[channel.start + offset] = up_value;
            lower[channel.start + offset] = lo_value;
        }
    }
    Ok(EnvelopeEvaluation {
        input,
        upper,
        lower,
    })
}

struct Channel<'a> {
    values: &'a [f64],
    start: usize,
}

impl SignalInput {
    fn channels(&self) -> Vec<Channel<'_>> {
        if self.vector {
            return vec![Channel {
                values: &self.data,
                start: 0,
            }];
        }
        (0..self.cols)
            .map(|col| {
                let start = col * self.rows;
                Channel {
                    values: &self.data[start..start + self.rows],
                    start,
                }
            })
            .collect()
    }
}

impl EnvelopeEvaluation {
    fn upper_value(&self) -> BuiltinResult<Value> {
        value_from_data(self.upper.clone(), self.input.shape.clone())
    }

    fn lower_value(&self) -> BuiltinResult<Value> {
        value_from_data(self.lower.clone(), self.input.shape.clone())
    }

    fn input_value(&self) -> BuiltinResult<Value> {
        value_from_data(self.input.data.clone(), self.input.shape.clone())
    }
}

fn value_from_data(data: Vec<f64>, shape: Vec<usize>) -> BuiltinResult<Value> {
    if data.len() == 1 {
        return Ok(Value::Num(data[0]));
    }
    Tensor::new(data, shape)
        .map(Value::Tensor)
        .map_err(|err| envelope_error_with_detail(&ENVELOPE_ERROR_INTERNAL, err))
}

fn analytic_envelope(
    values: &[f64],
    filter_len: Option<usize>,
) -> BuiltinResult<(Vec<f64>, Vec<f64>)> {
    if values.len() == 1 {
        return Ok((vec![values[0]], vec![values[0]]));
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let centered = values.iter().map(|value| value - mean).collect::<Vec<_>>();
    let magnitude = if let Some(filter_len) = filter_len {
        analytic_fir_magnitude(&centered, filter_len)
    } else {
        let analytic = analytic_signal(&centered);
        analytic.iter().map(|z| z.norm()).collect::<Vec<_>>()
    };
    let upper = magnitude
        .iter()
        .map(|value| mean + value)
        .collect::<Vec<_>>();
    let lower = magnitude
        .iter()
        .map(|value| mean - value)
        .collect::<Vec<_>>();
    Ok((upper, lower))
}

fn analytic_fir_magnitude(values: &[f64], filter_len: usize) -> Vec<f64> {
    if filter_len <= 1 {
        return values.iter().map(|value| value.abs()).collect();
    }
    let kernel = hilbert_fir_kernel(filter_len);
    let quadrature = centered_convolution(values, &kernel);
    values
        .iter()
        .zip(quadrature)
        .map(|(&re, im)| re.hypot(im))
        .collect()
}

fn hilbert_fir_kernel(filter_len: usize) -> Vec<f64> {
    let center = (filter_len as f64 - 1.0) / 2.0;
    let denominator = modified_bessel_i0(8.0);
    (0..filter_len)
        .map(|idx| {
            let k = idx as f64 - center;
            let ideal = if k.abs() <= EPS {
                0.0
            } else {
                let rounded = k.round();
                if (rounded - k).abs() <= EPS && (rounded as i64).rem_euclid(2) == 0 {
                    0.0
                } else {
                    2.0 / (std::f64::consts::PI * k)
                }
            };
            ideal * kaiser_window(idx, filter_len, 8.0, denominator)
        })
        .collect()
}

fn kaiser_window(idx: usize, len: usize, beta: f64, denominator: f64) -> f64 {
    if len <= 1 {
        return 1.0;
    }
    let ratio = 2.0 * idx as f64 / (len - 1) as f64 - 1.0;
    let argument = beta * (1.0 - ratio * ratio).max(0.0).sqrt();
    modified_bessel_i0(argument) / denominator
}

fn modified_bessel_i0(x: f64) -> f64 {
    let y = x * x / 4.0;
    let mut term = 1.0;
    let mut sum = 1.0;
    for k in 1..=32 {
        term *= y / ((k * k) as f64);
        sum += term;
        if term.abs() <= sum.abs() * 1.0e-15 {
            break;
        }
    }
    sum
}

fn centered_convolution(values: &[f64], kernel: &[f64]) -> Vec<f64> {
    let center = kernel.len() / 2;
    let mut out = vec![0.0; values.len()];
    for (idx, slot) in out.iter_mut().enumerate() {
        let mut acc = 0.0;
        for (tap, &weight) in kernel.iter().enumerate() {
            let source = idx as isize + tap as isize - center as isize;
            if (0..values.len() as isize).contains(&source) {
                acc += values[source as usize] * weight;
            }
        }
        *slot = acc;
    }
    out
}

fn analytic_signal(values: &[f64]) -> Vec<Complex<f64>> {
    let n = values.len();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);
    let mut buffer = values
        .iter()
        .map(|&value| Complex::new(value, 0.0))
        .collect::<Vec<_>>();
    fft.process(&mut buffer);
    for (freq, value) in buffer.iter_mut().enumerate() {
        *value *= analytic_multiplier(freq, n);
    }
    ifft.process(&mut buffer);
    let scale = 1.0 / n as f64;
    for value in &mut buffer {
        *value *= scale;
    }
    buffer
}

fn analytic_multiplier(freq: usize, len: usize) -> f64 {
    if freq == 0 {
        return 1.0;
    }
    if len.is_multiple_of(2) {
        if freq < len / 2 {
            2.0
        } else if freq == len / 2 {
            1.0
        } else {
            0.0
        }
    } else if freq <= len / 2 {
        2.0
    } else {
        0.0
    }
}

fn rms_envelope(values: &[f64], window_len: usize) -> (Vec<f64>, Vec<f64>) {
    let n = values.len();
    let half_before = (window_len - 1) / 2;
    let half_after = window_len / 2;
    let mut prefix = Vec::with_capacity(n + 1);
    prefix.push(0.0);
    for value in values {
        prefix.push(prefix.last().copied().unwrap_or(0.0) + value * value);
    }
    let mut upper = Vec::with_capacity(n);
    for idx in 0..n {
        let start = idx.saturating_sub(half_before);
        let end = (idx + half_after + 1).min(n);
        let count = (end - start).max(1) as f64;
        upper.push(((prefix[end] - prefix[start]) / count).sqrt());
    }
    let lower = upper.iter().map(|value| -*value).collect();
    (upper, lower)
}

fn peak_envelope(values: &[f64], separation: usize) -> BuiltinResult<(Vec<f64>, Vec<f64>)> {
    if values.len() == 1 {
        return Ok((vec![values[0]], vec![values[0]]));
    }
    let upper_points = extrema_points(values, separation, true);
    let lower_points = extrema_points(values, separation, false);
    let query = (0..values.len()).map(|idx| idx as f64).collect::<Vec<_>>();
    let upper = spline_interpolate(&upper_points, &query)?;
    let lower = spline_interpolate(&lower_points, &query)?;
    Ok((upper, lower))
}

fn extrema_points(values: &[f64], separation: usize, maxima: bool) -> Vec<(usize, f64)> {
    let mut candidates = Vec::new();
    for idx in 1..values.len() - 1 {
        let prev = values[idx - 1];
        let curr = values[idx];
        let next = values[idx + 1];
        let is_extremum = if maxima {
            (curr >= prev && curr > next) || (curr > prev && curr >= next)
        } else {
            (curr <= prev && curr < next) || (curr < prev && curr <= next)
        };
        if is_extremum {
            candidates.push((idx, curr));
        }
    }
    candidates.sort_by(|a, b| {
        let lhs = if maxima { a.1 } else { -a.1 };
        let rhs = if maxima { b.1 } else { -b.1 };
        rhs.partial_cmp(&lhs).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut selected = Vec::new();
    for candidate in candidates {
        if selected
            .iter()
            .all(|(idx, _): &(usize, f64)| idx.abs_diff(candidate.0) >= separation)
        {
            selected.push(candidate);
        }
    }
    selected.push((0, values[0]));
    selected.push((values.len() - 1, values[values.len() - 1]));
    selected.sort_by_key(|(idx, _)| *idx);
    selected.dedup_by_key(|(idx, _)| *idx);
    if selected.len() == 1 {
        selected.push((values.len() - 1, values[values.len() - 1]));
    }
    selected
}

fn spline_interpolate(points: &[(usize, f64)], query: &[f64]) -> BuiltinResult<Vec<f64>> {
    if points.len() < 2 {
        return Err(envelope_error(&ENVELOPE_ERROR_INTERNAL));
    }
    let series = NumericSeries {
        x: points
            .iter()
            .map(|(idx, _)| *idx as f64)
            .collect::<Vec<_>>(),
        y: points.iter().map(|(_, value)| *value).collect::<Vec<_>>(),
        series: 1,
        trailing_shape: Vec::new(),
    };
    let query = QueryPoints {
        values: query.to_vec(),
        shape: vec![1, query.len()],
    };
    let pp = build_spline_pp(&series, BUILTIN_NAME)
        .map_err(|err| envelope_error_with_detail(&ENVELOPE_ERROR_INTERNAL, err.message()))?;
    let value = evaluate_pp(&pp, &query, &Extrapolation::Extrapolate, BUILTIN_NAME)
        .map_err(|err| envelope_error_with_detail(&ENVELOPE_ERROR_INTERNAL, err.message()))?;
    match value {
        Value::Tensor(tensor) => Ok(tensor.data),
        Value::Num(value) => Ok(vec![value]),
        _ => Err(envelope_error(&ENVELOPE_ERROR_INTERNAL)),
    }
}

async fn render_envelope_plot(eval: &EnvelopeEvaluation) -> BuiltinResult<()> {
    let mut args = Vec::new();
    let x = if eval.input.vector {
        Tensor::new(
            (1..=eval.input.data.len()).map(|idx| idx as f64).collect(),
            vec![eval.input.data.len(), 1],
        )
        .map(Value::Tensor)
        .map_err(|err| envelope_error_with_detail(&ENVELOPE_ERROR_INTERNAL, err))?
    } else {
        Tensor::new(
            (1..=eval.input.rows).map(|idx| idx as f64).collect(),
            vec![eval.input.rows, 1],
        )
        .map(Value::Tensor)
        .map_err(|err| envelope_error_with_detail(&ENVELOPE_ERROR_INTERNAL, err))?
    };

    if eval.input.vector {
        args.extend([
            x.clone(),
            eval.input_value()?,
            x.clone(),
            eval.upper_value()?,
            x,
            eval.lower_value()?,
        ]);
    } else {
        for col in 0..eval.input.cols {
            let start = col * eval.input.rows;
            let end = start + eval.input.rows;
            args.push(x.clone());
            args.push(column_value(&eval.input.data[start..end])?);
            args.push(x.clone());
            args.push(column_value(&eval.upper[start..end])?);
            args.push(x.clone());
            args.push(column_value(&eval.lower[start..end])?);
        }
    }

    let _handle = crate::builtins::plotting::plot::plot_builtin(args)
        .await
        .map_err(|err| envelope_error_with_detail(&ENVELOPE_ERROR_INTERNAL, err.message()))?;
    Ok(())
}

fn column_value(values: &[f64]) -> BuiltinResult<Value> {
    Tensor::new(values.to_vec(), vec![values.len(), 1])
        .map(Value::Tensor)
        .map_err(|err| envelope_error_with_detail(&ENVELOPE_ERROR_INTERNAL, err))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    fn row(values: Vec<f64>) -> Value {
        Value::Tensor(Tensor::new(values.clone(), vec![1, values.len()]).expect("tensor"))
    }

    fn col(values: Vec<f64>) -> Value {
        Value::Tensor(Tensor::new(values.clone(), vec![values.len(), 1]).expect("tensor"))
    }

    fn call(value: Value, rest: Vec<Value>, outputs: Option<usize>) -> BuiltinResult<Value> {
        let _guard = outputs.map(|count| crate::output_count::push_output_count(Some(count)));
        block_on(envelope_builtin(value, rest))
    }

    fn tensor_data(value: Value) -> Vec<f64> {
        match value {
            Value::Tensor(tensor) => tensor.data,
            Value::Num(value) => vec![value],
            other => panic!("expected numeric output, got {other:?}"),
        }
    }

    fn output_list(value: Value) -> Vec<Value> {
        let Value::OutputList(values) = value else {
            panic!("expected output list, got {value:?}");
        };
        values
    }

    #[test]
    fn analytic_envelope_tracks_am_modulation() {
        let n = 128usize;
        let values = (0..n)
            .map(|idx| {
                let t = idx as f64 / n as f64;
                let carrier = (2.0 * std::f64::consts::PI * 8.0 * t).sin();
                let amp = 1.25 + 0.5 * (2.0 * std::f64::consts::PI * t).sin();
                amp * carrier
            })
            .collect::<Vec<_>>();

        let out = call(row(values), Vec::new(), None).expect("envelope");
        let upper = tensor_data(out);
        for (idx, actual) in upper.iter().enumerate().skip(4).take(n - 8) {
            let t = idx as f64 / n as f64;
            let expected = 1.25 + 0.5 * (2.0 * std::f64::consts::PI * t).sin();
            assert!(
                (actual - expected).abs() < 0.15,
                "idx={idx} actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    fn analytic_envelope_adds_mean_back_to_bounds() {
        let values = vec![3.0, 4.0, 3.0, 2.0];
        let outputs = output_list(call(row(values), Vec::new(), Some(2)).expect("envelope"));
        let upper = tensor_data(outputs[0].clone());
        let lower = tensor_data(outputs[1].clone());
        assert_eq!(upper.len(), 4);
        assert_eq!(lower.len(), 4);
        assert!(upper.iter().all(|value| *value >= 3.0));
        assert!(lower.iter().all(|value| *value <= 3.0));
    }

    #[test]
    fn rms_envelope_uses_centered_window() {
        let outputs = output_list(
            call(
                row(vec![0.0, 3.0, 4.0, 0.0]),
                vec![Value::Num(3.0), Value::String("rms".to_string())],
                Some(2),
            )
            .expect("envelope"),
        );
        let upper = tensor_data(outputs[0].clone());
        let lower = tensor_data(outputs[1].clone());
        assert!((upper[1] - (25.0f64 / 3.0).sqrt()).abs() < 1.0e-12);
        assert_eq!(lower[1], -upper[1]);
    }

    #[test]
    fn peak_envelope_interpolates_extrema() {
        let outputs = output_list(
            call(
                row(vec![0.0, 2.0, 0.0, -1.0, 0.0, 3.0, 0.0]),
                vec![Value::Num(2.0), Value::String("peak".to_string())],
                Some(2),
            )
            .expect("envelope"),
        );
        let upper = tensor_data(outputs[0].clone());
        let lower = tensor_data(outputs[1].clone());
        assert!((upper[1] - 2.0).abs() < 1.0e-12);
        assert!((upper[5] - 3.0).abs() < 1.0e-12);
        assert!((lower[3] + 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn matrix_inputs_are_processed_column_wise() {
        let matrix =
            Tensor::new(vec![1.0, 0.0, -1.0, 0.0, 2.0, 3.0, 2.0, 1.0], vec![4, 2]).expect("matrix");
        let out = call(Value::Tensor(matrix), Vec::new(), Some(2)).expect("envelope");
        let outputs = output_list(out);
        let Value::Tensor(upper) = outputs[0].clone() else {
            panic!("expected tensor");
        };
        assert_eq!(upper.shape, vec![4, 2]);
        assert!(upper.data[..4]
            .iter()
            .all(|value| (*value - 1.0).abs() < 1.0e-12));
        assert!(upper.data[4..].iter().all(|value| *value >= 2.0));
    }

    #[test]
    fn output_count_zero_returns_empty_output_list() {
        crate::builtins::plotting::tests::ensure_plot_test_env();
        let _plot_lock = crate::builtins::plotting::tests::lock_plot_registry();
        let out = call(col(vec![1.0, 0.0, -1.0, 0.0]), Vec::new(), Some(0)).expect("envelope");
        assert_eq!(out, Value::OutputList(Vec::new()));
    }

    #[test]
    fn matrix_statement_form_plots_each_channel_without_length_mismatch() {
        crate::builtins::plotting::tests::ensure_plot_test_env();
        let _plot_lock = crate::builtins::plotting::tests::lock_plot_registry();
        let matrix =
            Tensor::new(vec![1.0, 0.0, -1.0, 0.0, 2.0, 3.0, 2.0, 1.0], vec![4, 2]).expect("matrix");
        let out = call(Value::Tensor(matrix), Vec::new(), Some(0)).expect("envelope");
        assert_eq!(out, Value::OutputList(Vec::new()));
    }

    #[test]
    fn analytic_filter_length_uses_fir_path() {
        let values = vec![0.0, 1.0, 0.0, -0.5, 0.0, 0.25, 0.0, -0.125];
        let dft = tensor_data(call(row(values.clone()), Vec::new(), None).expect("envelope"));
        let fir = tensor_data(
            call(
                row(values),
                vec![Value::Num(5.0), Value::String("analytic".to_string())],
                None,
            )
            .expect("envelope"),
        );
        assert_eq!(dft.len(), fir.len());
        assert!(
            dft.iter()
                .zip(fir.iter())
                .any(|(left, right)| (left - right).abs() > 1.0e-6),
            "filter-length analytic form should not silently use the DFT path"
        );
    }

    #[test]
    fn too_many_outputs_has_stable_identifier() {
        let err = call(row(vec![1.0, 0.0, -1.0, 0.0]), Vec::new(), Some(3))
            .expect_err("too many outputs should fail");
        assert_eq!(err.identifier(), ENVELOPE_ERROR_TOO_MANY_OUTPUTS.identifier);
    }

    #[test]
    fn invalid_method_has_stable_identifier() {
        let err = call(
            row(vec![1.0, 2.0, 3.0]),
            vec![Value::Num(3.0), Value::String("bad".to_string())],
            None,
        )
        .expect_err("invalid method should fail");
        assert_eq!(err.identifier(), ENVELOPE_ERROR_INVALID_METHOD.identifier);
    }
}
