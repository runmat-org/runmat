//! MATLAB-compatible sample-rate conversion builtins.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, NumericDType, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::builtins::math::signal::type_resolvers::{
    downsample_type, resample_type, upsample_type,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const UPSAMPLE_NAME: &str = "upsample";
const DOWNSAMPLE_NAME: &str = "downsample";
const RESAMPLE_NAME: &str = "resample";
const DEFAULT_RESAMPLE_N: usize = 10;
const DEFAULT_RESAMPLE_BETA: f64 = 5.0;

const SAMPLE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sample-rate-adjusted signal.",
}];

const SAMPLE_INPUTS_CORE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input signal array.",
    },
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integer sample-rate factor.",
    },
];

const SAMPLE_INPUTS_PHASE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input signal array.",
    },
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integer sample-rate factor.",
    },
    BuiltinParamDescriptor {
        name: "PHASE",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("0"),
        description: "Integer phase offset in the range 0 <= PHASE < N.",
    },
];

const UPSAMPLE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = upsample(X, N)",
        inputs: &SAMPLE_INPUTS_CORE,
        outputs: &SAMPLE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = upsample(X, N, PHASE)",
        inputs: &SAMPLE_INPUTS_PHASE,
        outputs: &SAMPLE_OUTPUT,
    },
];

const DOWNSAMPLE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = downsample(X, N)",
        inputs: &SAMPLE_INPUTS_CORE,
        outputs: &SAMPLE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = downsample(X, N, PHASE)",
        inputs: &SAMPLE_INPUTS_PHASE,
        outputs: &SAMPLE_OUTPUT,
    },
];

const RESAMPLE_INPUT_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input signal array.",
};

const RESAMPLE_INPUT_P: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "P",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Positive integer upsampling factor.",
};

const RESAMPLE_INPUT_Q: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Q",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Positive integer downsampling factor.",
};

const RESAMPLE_INPUT_N: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "N",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("10"),
    description: "Half-length filter order factor.",
};

const RESAMPLE_INPUT_BETA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "BETA",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("5"),
    description: "Kaiser window shape parameter.",
};

const RESAMPLE_INPUT_B: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "FIR antialiasing filter coefficients.",
};

const RESAMPLE_INPUTS_CORE: [BuiltinParamDescriptor; 3] =
    [RESAMPLE_INPUT_X, RESAMPLE_INPUT_P, RESAMPLE_INPUT_Q];

const RESAMPLE_INPUTS_N: [BuiltinParamDescriptor; 4] = [
    RESAMPLE_INPUT_X,
    RESAMPLE_INPUT_P,
    RESAMPLE_INPUT_Q,
    RESAMPLE_INPUT_N,
];

const RESAMPLE_INPUTS_N_BETA: [BuiltinParamDescriptor; 5] = [
    RESAMPLE_INPUT_X,
    RESAMPLE_INPUT_P,
    RESAMPLE_INPUT_Q,
    RESAMPLE_INPUT_N,
    RESAMPLE_INPUT_BETA,
];

const RESAMPLE_INPUTS_FILTER: [BuiltinParamDescriptor; 4] = [
    RESAMPLE_INPUT_X,
    RESAMPLE_INPUT_P,
    RESAMPLE_INPUT_Q,
    RESAMPLE_INPUT_B,
];

const RESAMPLE_INPUTS_FILTER_VARIADIC: [BuiltinParamDescriptor; 4] = [
    RESAMPLE_INPUT_X,
    RESAMPLE_INPUT_P,
    RESAMPLE_INPUT_Q,
    BuiltinParamDescriptor {
        arity: BuiltinParamArity::Variadic,
        ..RESAMPLE_INPUT_B
    },
];

const RESAMPLE_OUTPUT_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Resampled signal.",
};

const RESAMPLE_OUTPUT_B: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "FIR antialiasing filter coefficients used for resampling.",
};

const RESAMPLE_OUTPUTS_Y: [BuiltinParamDescriptor; 1] = [RESAMPLE_OUTPUT_Y];
const RESAMPLE_OUTPUTS_Y_B: [BuiltinParamDescriptor; 2] = [RESAMPLE_OUTPUT_Y, RESAMPLE_OUTPUT_B];

const RESAMPLE_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "Y = resample(X, P, Q)",
        inputs: &RESAMPLE_INPUTS_CORE,
        outputs: &RESAMPLE_OUTPUTS_Y,
    },
    BuiltinSignatureDescriptor {
        label: "Y = resample(X, P, Q, N)",
        inputs: &RESAMPLE_INPUTS_N,
        outputs: &RESAMPLE_OUTPUTS_Y,
    },
    BuiltinSignatureDescriptor {
        label: "Y = resample(X, P, Q, N, BETA)",
        inputs: &RESAMPLE_INPUTS_N_BETA,
        outputs: &RESAMPLE_OUTPUTS_Y,
    },
    BuiltinSignatureDescriptor {
        label: "Y = resample(X, P, Q, B)",
        inputs: &RESAMPLE_INPUTS_FILTER,
        outputs: &RESAMPLE_OUTPUTS_Y,
    },
    BuiltinSignatureDescriptor {
        label: "[Y, B] = resample(X, P, Q, ___)",
        inputs: &RESAMPLE_INPUTS_N_BETA,
        outputs: &RESAMPLE_OUTPUTS_Y_B,
    },
    BuiltinSignatureDescriptor {
        label: "[Y, B] = resample(X, P, Q, B, ___)",
        inputs: &RESAMPLE_INPUTS_FILTER_VARIADIC,
        outputs: &RESAMPLE_OUTPUTS_Y_B,
    },
];

const SAMPLE_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAMPLE_RATE.ARG_COUNT",
    identifier: Some("RunMat:sampleRate:ArgCount"),
    when: "Required arguments are missing or more than three inputs are provided.",
    message: "sample-rate builtin: expected X, N, and optional PHASE",
};

const SAMPLE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAMPLE_RATE.INVALID_INPUT",
    identifier: Some("RunMat:sampleRate:InvalidInput"),
    when: "X is not numeric, logical, complex, or a gpuArray containing numeric values.",
    message: "sample-rate builtin: X must be numeric or logical",
};

const SAMPLE_ERROR_INVALID_FACTOR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAMPLE_RATE.INVALID_FACTOR",
    identifier: Some("RunMat:sampleRate:InvalidFactor"),
    when: "N is not a finite positive integer scalar.",
    message: "sample-rate builtin: N must be a positive integer scalar",
};

const SAMPLE_ERROR_INVALID_PHASE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAMPLE_RATE.INVALID_PHASE",
    identifier: Some("RunMat:sampleRate:InvalidPhase"),
    when: "PHASE is not an integer scalar in the range 0 <= PHASE < N.",
    message: "sample-rate builtin: PHASE must be an integer scalar with 0 <= PHASE < N",
};

const SAMPLE_ERROR_GATHER_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAMPLE_RATE.GATHER_FAILED",
    identifier: Some("RunMat:sampleRate:GatherFailed"),
    when: "A gpuArray input cannot be gathered for host sample-rate conversion.",
    message: "sample-rate builtin: failed to gather GPU input",
};

const SAMPLE_ERROR_SIZE_OVERFLOW: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAMPLE_RATE.SIZE_OVERFLOW",
    identifier: Some("RunMat:sampleRate:SizeOverflow"),
    when: "The requested output shape overflows host allocation limits.",
    message: "sample-rate builtin: requested output is too large",
};

const SAMPLE_ERROR_BUILD_OUTPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAMPLE_RATE.BUILD_OUTPUT",
    identifier: Some("RunMat:sampleRate:BuildOutput"),
    when: "Output tensor construction fails.",
    message: "sample-rate builtin: failed to build output",
};

const SAMPLE_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAMPLE_RATE.INVALID_OPTION",
    identifier: Some("RunMat:sampleRate:InvalidOption"),
    when: "A sample-rate conversion option, filter vector, or dimension selector is invalid.",
    message: "sample-rate builtin: invalid option",
};

const SAMPLE_ERROR_OUTPUT_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAMPLE_RATE.OUTPUT_COUNT",
    identifier: Some("RunMat:sampleRate:OutputCount"),
    when: "More outputs are requested than the builtin supports.",
    message: "sample-rate builtin: too many output arguments",
};

const BASIC_SAMPLE_ERRORS: [BuiltinErrorDescriptor; 7] = [
    SAMPLE_ERROR_ARG_COUNT,
    SAMPLE_ERROR_INVALID_INPUT,
    SAMPLE_ERROR_INVALID_FACTOR,
    SAMPLE_ERROR_INVALID_PHASE,
    SAMPLE_ERROR_GATHER_FAILED,
    SAMPLE_ERROR_SIZE_OVERFLOW,
    SAMPLE_ERROR_BUILD_OUTPUT,
];

const RESAMPLE_ERRORS: [BuiltinErrorDescriptor; 9] = [
    SAMPLE_ERROR_ARG_COUNT,
    SAMPLE_ERROR_INVALID_INPUT,
    SAMPLE_ERROR_INVALID_FACTOR,
    SAMPLE_ERROR_INVALID_PHASE,
    SAMPLE_ERROR_GATHER_FAILED,
    SAMPLE_ERROR_SIZE_OVERFLOW,
    SAMPLE_ERROR_BUILD_OUTPUT,
    SAMPLE_ERROR_INVALID_OPTION,
    SAMPLE_ERROR_OUTPUT_COUNT,
];

pub const UPSAMPLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &UPSAMPLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &BASIC_SAMPLE_ERRORS,
};

pub const DOWNSAMPLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DOWNSAMPLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &BASIC_SAMPLE_ERRORS,
};

pub const RESAMPLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RESAMPLE_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &RESAMPLE_ERRORS,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SampleOp {
    Up,
    Down,
}

enum SampleInput {
    Real {
        data: Vec<f64>,
        shape: Vec<usize>,
        dtype: NumericDType,
    },
    Complex {
        data: Vec<(f64, f64)>,
        shape: Vec<usize>,
    },
}

impl SampleInput {
    async fn from_value(value: Value, builtin: &'static str) -> BuiltinResult<Self> {
        match value {
            Value::GpuTensor(handle) => {
                let tensor = gpu_helpers::gather_tensor_async(&handle)
                    .await
                    .map_err(|flow| {
                        let detail = flow.message().to_owned();
                        sample_error_with_source(
                            builtin,
                            &SAMPLE_ERROR_GATHER_FAILED,
                            detail,
                            map_control_flow_with_builtin(flow, builtin),
                        )
                    })?;
                Ok(Self::Real {
                    data: tensor.data,
                    shape: tensor.shape,
                    dtype: tensor.dtype,
                })
            }
            Value::Tensor(tensor) => Ok(Self::Real {
                data: tensor.data,
                shape: tensor.shape,
                dtype: tensor.dtype,
            }),
            Value::LogicalArray(logical) => {
                let tensor = tensor::logical_to_tensor(&logical).map_err(|err| {
                    sample_error_with_detail(builtin, &SAMPLE_ERROR_INVALID_INPUT, err)
                })?;
                Ok(Self::Real {
                    data: tensor.data,
                    shape: tensor.shape,
                    dtype: NumericDType::F64,
                })
            }
            Value::ComplexTensor(tensor) => Ok(Self::Complex {
                data: tensor.data,
                shape: tensor.shape,
            }),
            Value::Num(value) => Ok(Self::Real {
                data: vec![value],
                shape: vec![1, 1],
                dtype: NumericDType::F64,
            }),
            Value::Int(value) => Ok(Self::Real {
                data: vec![value.to_f64()],
                shape: vec![1, 1],
                dtype: NumericDType::F64,
            }),
            Value::Bool(value) => Ok(Self::Real {
                data: vec![if value { 1.0 } else { 0.0 }],
                shape: vec![1, 1],
                dtype: NumericDType::F64,
            }),
            Value::Complex(re, im) => Ok(Self::Complex {
                data: vec![(re, im)],
                shape: vec![1, 1],
            }),
            other => Err(sample_error_with_detail(
                builtin,
                &SAMPLE_ERROR_INVALID_INPUT,
                format!("received {other:?}"),
            )),
        }
    }
}

fn sample_error(builtin: &'static str, error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    sample_error_with_message(builtin, error.message, error)
}

fn sample_error_with_detail(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    sample_error_with_message(
        builtin,
        format!("{}: {}", error.message, detail.as_ref()),
        error,
    )
}

fn sample_error_with_message(
    builtin: &'static str,
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(builtin);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn sample_error_with_source(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
    source: RuntimeError,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(builtin)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "upsample",
    category = "math/signal",
    summary = "Increase sample rate by inserting zeros between samples.",
    keywords = "upsample,sample rate,zero insertion,signal processing",
    type_resolver(upsample_type),
    descriptor(crate::builtins::math::signal::sample_rate::UPSAMPLE_DESCRIPTOR),
    builtin_path = "crate::builtins::math::signal::sample_rate"
)]
async fn upsample_builtin(x: Value, n: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    sample_rate_builtin(UPSAMPLE_NAME, SampleOp::Up, x, n, rest).await
}

#[runtime_builtin(
    name = "downsample",
    category = "math/signal",
    summary = "Decrease sample rate by keeping every Nth sample.",
    keywords = "downsample,sample rate,decimation,signal processing",
    type_resolver(downsample_type),
    descriptor(crate::builtins::math::signal::sample_rate::DOWNSAMPLE_DESCRIPTOR),
    builtin_path = "crate::builtins::math::signal::sample_rate"
)]
async fn downsample_builtin(x: Value, n: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    sample_rate_builtin(DOWNSAMPLE_NAME, SampleOp::Down, x, n, rest).await
}

#[runtime_builtin(
    name = "resample",
    category = "math/signal",
    summary = "Resample a signal by a rational factor with FIR antialiasing.",
    keywords = "resample,sample rate,polyphase,antialiasing,FIR,signal processing",
    type_resolver(resample_type),
    descriptor(crate::builtins::math::signal::sample_rate::RESAMPLE_DESCRIPTOR),
    builtin_path = "crate::builtins::math::signal::sample_rate"
)]
async fn resample_builtin(x: Value, p: Value, q: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let requested_outputs = crate::output_count::current_output_count();
    match requested_outputs {
        Some(0) => return Ok(Value::OutputList(Vec::new())),
        Some(count) if count > 2 => {
            return Err(sample_error(RESAMPLE_NAME, &SAMPLE_ERROR_OUTPUT_COUNT));
        }
        _ => {}
    }

    let input = SampleInput::from_value(x, RESAMPLE_NAME).await?;
    let mut p = parse_factor(&p, RESAMPLE_NAME).await?;
    let mut q = parse_factor(&q, RESAMPLE_NAME).await?;
    let divisor = gcd(p, q);
    p /= divisor;
    q /= divisor;
    let options = parse_resample_options(rest, p, q).await?;
    let eval = apply_resample(input, p, q, &options)?;
    match requested_outputs {
        Some(1) => Ok(Value::OutputList(vec![eval.y])),
        Some(2) => Ok(Value::OutputList(vec![
            eval.y,
            Value::Tensor(filter_tensor(eval.filter)?),
        ])),
        None => Ok(eval.y),
        _ => unreachable!("output count was validated before evaluation"),
    }
}

async fn sample_rate_builtin(
    builtin: &'static str,
    op: SampleOp,
    x: Value,
    n: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(sample_error(builtin, &SAMPLE_ERROR_ARG_COUNT));
    }
    let factor = parse_factor(&n, builtin).await?;
    let phase = match rest.first() {
        Some(value) => parse_phase(value, factor, builtin).await?,
        None => 0,
    };
    if op == SampleOp::Down {
        if let Value::GpuTensor(handle) = &x {
            if let Some(output) = downsample_gpu(handle, factor, phase, builtin)? {
                return Ok(output);
            }
        }
    }
    let input = SampleInput::from_value(x, builtin).await?;
    apply_sample_rate(input, factor, phase, op, builtin)
}

#[derive(Clone, Debug)]
struct ResampleOptions {
    filter: Vec<f64>,
    dim: Option<usize>,
}

struct ResampleEval {
    y: Value,
    filter: Vec<f64>,
}

async fn parse_resample_options(
    rest: Vec<Value>,
    p: usize,
    q: usize,
) -> BuiltinResult<ResampleOptions> {
    let mut idx = 0usize;
    let mut positional = Vec::new();
    let mut dim = None;
    while idx < rest.len() {
        if let Some(key) = value_as_text(&rest[idx]) {
            if key.eq_ignore_ascii_case("dimension") {
                let Some(value) = rest.get(idx + 1) else {
                    return Err(sample_error_with_detail(
                        RESAMPLE_NAME,
                        &SAMPLE_ERROR_INVALID_OPTION,
                        "Dimension name-value option requires a value",
                    ));
                };
                dim = Some(
                    tensor::parse_dimension(value, RESAMPLE_NAME)
                        .map_err(|err| {
                            sample_error_with_detail(
                                RESAMPLE_NAME,
                                &SAMPLE_ERROR_INVALID_OPTION,
                                err,
                            )
                        })?
                        .saturating_sub(1),
                );
                idx += 2;
                continue;
            }
        }
        positional.push(rest[idx].clone());
        idx += 1;
    }

    let filter = if positional.is_empty() {
        design_resample_filter(p, q, DEFAULT_RESAMPLE_N, DEFAULT_RESAMPLE_BETA)?
    } else {
        parse_resample_filter_spec(&positional, p, q).await?
    };
    Ok(ResampleOptions { filter, dim })
}

async fn parse_resample_filter_spec(args: &[Value], p: usize, q: usize) -> BuiltinResult<Vec<f64>> {
    match args.len() {
        0 => design_resample_filter(p, q, DEFAULT_RESAMPLE_N, DEFAULT_RESAMPLE_BETA),
        1 => {
            if let Some(n) = scalar_integer_option(&args[0]).await? {
                design_resample_filter(p, q, n, DEFAULT_RESAMPLE_BETA)
            } else {
                parse_filter_vector(args[0].clone())
            }
        }
        2 => {
            let Some(n) = scalar_integer_option(&args[0]).await? else {
                return Err(sample_error_with_detail(
                    RESAMPLE_NAME,
                    &SAMPLE_ERROR_INVALID_OPTION,
                    "N must be a nonnegative integer scalar when BETA is provided",
                ));
            };
            let beta = scalar_f64_option(&args[1]).await?;
            design_resample_filter(p, q, n, beta)
        }
        _ => Err(sample_error_with_detail(
            RESAMPLE_NAME,
            &SAMPLE_ERROR_ARG_COUNT,
            "resample accepts X, P, Q, optional N/BETA or B, and optional Dimension",
        )),
    }
}

async fn scalar_integer_option(value: &Value) -> BuiltinResult<Option<usize>> {
    match value {
        Value::Tensor(tensor) if tensor.data.len() != 1 => return Ok(None),
        Value::ComplexTensor(_) => return Ok(None),
        _ => {}
    }
    let Some(raw) = tensor::scalar_f64_from_value_async(value)
        .await
        .map_err(|err| {
            sample_error_with_detail(RESAMPLE_NAME, &SAMPLE_ERROR_INVALID_OPTION, err)
        })?
    else {
        return Ok(None);
    };
    parse_nonnegative_integer(raw, RESAMPLE_NAME, &SAMPLE_ERROR_INVALID_OPTION).map(Some)
}

async fn scalar_f64_option(value: &Value) -> BuiltinResult<f64> {
    let Some(raw) = tensor::scalar_f64_from_value_async(value)
        .await
        .map_err(|err| {
            sample_error_with_detail(RESAMPLE_NAME, &SAMPLE_ERROR_INVALID_OPTION, err)
        })?
    else {
        return Err(sample_error_with_detail(
            RESAMPLE_NAME,
            &SAMPLE_ERROR_INVALID_OPTION,
            format!("expected scalar option, received {value:?}"),
        ));
    };
    if !raw.is_finite() || raw < 0.0 {
        return Err(sample_error_with_detail(
            RESAMPLE_NAME,
            &SAMPLE_ERROR_INVALID_OPTION,
            "BETA must be a finite nonnegative scalar",
        ));
    }
    Ok(raw)
}

fn parse_filter_vector(value: Value) -> BuiltinResult<Vec<f64>> {
    let tensor = tensor::value_into_tensor_for(RESAMPLE_NAME, value).map_err(|err| {
        sample_error_with_detail(RESAMPLE_NAME, &SAMPLE_ERROR_INVALID_OPTION, err)
    })?;
    if tensor.data.is_empty() {
        return Err(sample_error_with_detail(
            RESAMPLE_NAME,
            &SAMPLE_ERROR_INVALID_OPTION,
            "filter coefficient vector cannot be empty",
        ));
    }
    if tensor.shape.iter().filter(|dim| **dim > 1).count() > 1 {
        return Err(sample_error_with_detail(
            RESAMPLE_NAME,
            &SAMPLE_ERROR_INVALID_OPTION,
            "filter coefficients must be a vector",
        ));
    }
    Ok(tensor.data)
}

fn apply_resample(
    input: SampleInput,
    p: usize,
    q: usize,
    options: &ResampleOptions,
) -> BuiltinResult<ResampleEval> {
    match input {
        SampleInput::Real { data, shape, dtype } => {
            let (output, shape) = resample_rational_column_major(data, shape, p, q, options)?;
            let tensor = Tensor::new_with_dtype(output, shape, dtype).map_err(|err| {
                sample_error_with_detail(RESAMPLE_NAME, &SAMPLE_ERROR_BUILD_OUTPUT, err)
            })?;
            Ok(ResampleEval {
                y: tensor::tensor_into_value(tensor),
                filter: options.filter.clone(),
            })
        }
        SampleInput::Complex { data, shape } => {
            let (output, shape) = resample_rational_column_major(data, shape, p, q, options)?;
            let tensor = ComplexTensor::new(output, shape).map_err(|err| {
                sample_error_with_detail(RESAMPLE_NAME, &SAMPLE_ERROR_BUILD_OUTPUT, err)
            })?;
            let y = if tensor.data.len() == 1 {
                let (re, im) = tensor.data[0];
                Value::Complex(re, im)
            } else {
                Value::ComplexTensor(tensor)
            };
            Ok(ResampleEval {
                y,
                filter: options.filter.clone(),
            })
        }
    }
}

async fn parse_factor(value: &Value, builtin: &'static str) -> BuiltinResult<usize> {
    let Some(raw) = tensor::scalar_f64_from_value_async(value)
        .await
        .map_err(|detail| {
            sample_error_with_detail(builtin, &SAMPLE_ERROR_INVALID_FACTOR, detail)
        })?
    else {
        return Err(sample_error_with_detail(
            builtin,
            &SAMPLE_ERROR_INVALID_FACTOR,
            format!("received {value:?}"),
        ));
    };
    parse_positive_integer(raw, builtin, &SAMPLE_ERROR_INVALID_FACTOR)
}

async fn parse_phase(value: &Value, factor: usize, builtin: &'static str) -> BuiltinResult<usize> {
    let Some(raw) = tensor::scalar_f64_from_value_async(value)
        .await
        .map_err(|detail| sample_error_with_detail(builtin, &SAMPLE_ERROR_INVALID_PHASE, detail))?
    else {
        return Err(sample_error_with_detail(
            builtin,
            &SAMPLE_ERROR_INVALID_PHASE,
            format!("received {value:?}"),
        ));
    };
    let phase = parse_nonnegative_integer(raw, builtin, &SAMPLE_ERROR_INVALID_PHASE)?;
    if phase >= factor {
        return Err(sample_error_with_detail(
            builtin,
            &SAMPLE_ERROR_INVALID_PHASE,
            format!("phase {phase} is not less than N ({factor})"),
        ));
    }
    Ok(phase)
}

fn parse_positive_integer(
    raw: f64,
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<usize> {
    let value = parse_nonnegative_integer(raw, builtin, error)?;
    if value == 0 {
        return Err(sample_error(builtin, error));
    }
    Ok(value)
}

fn parse_nonnegative_integer(
    raw: f64,
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<usize> {
    if !raw.is_finite() || raw < 0.0 {
        return Err(sample_error(builtin, error));
    }
    if raw.trunc() != raw || raw > usize::MAX as f64 {
        return Err(sample_error(builtin, error));
    }
    Ok(raw as usize)
}

fn apply_sample_rate(
    input: SampleInput,
    factor: usize,
    phase: usize,
    op: SampleOp,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    match input {
        SampleInput::Real { data, shape, dtype } => {
            let (output, shape) =
                resample_column_major(data, shape, factor, phase, op, 0.0, builtin)?;
            let tensor = Tensor::new_with_dtype(output, shape, dtype).map_err(|err| {
                sample_error_with_detail(builtin, &SAMPLE_ERROR_BUILD_OUTPUT, err)
            })?;
            Ok(tensor::tensor_into_value(tensor))
        }
        SampleInput::Complex { data, shape } => {
            let (output, shape) =
                resample_column_major(data, shape, factor, phase, op, (0.0, 0.0), builtin)?;
            let tensor = ComplexTensor::new(output, shape).map_err(|err| {
                sample_error_with_detail(builtin, &SAMPLE_ERROR_BUILD_OUTPUT, err)
            })?;
            if tensor.data.len() == 1 {
                let (re, im) = tensor.data[0];
                Ok(Value::Complex(re, im))
            } else {
                Ok(Value::ComplexTensor(tensor))
            }
        }
    }
}

fn downsample_gpu(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    factor: usize,
    phase: usize,
    builtin: &'static str,
) -> BuiltinResult<Option<Value>> {
    let handle_len = checked_product(&handle.shape)
        .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let shape = canonical_shape(handle.shape.clone(), handle_len);
    let dim = first_non_singleton_dim(&shape);
    let input_len = shape[dim];
    let output_len = output_len(input_len, factor, phase, SampleOp::Down)
        .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let mut output_shape = shape.clone();
    output_shape[dim] = output_len;

    if factor == 1 && phase == 0 {
        return Ok(Some(wrap_downsampled_gpu(handle, handle.clone())));
    }

    if runmat_accelerate_api::handle_storage(handle)
        != runmat_accelerate_api::GpuTensorStorage::Real
    {
        return Ok(None);
    }

    let Some(provider) = runmat_accelerate_api::provider_for_handle(handle) else {
        return Ok(None);
    };
    let Some(indices) =
        downsample_linear_indices(&shape, dim, input_len, output_len, factor, phase, builtin)?
    else {
        return Ok(None);
    };

    match provider.gather_linear(handle, &indices, &output_shape) {
        Ok(output) => Ok(Some(wrap_downsampled_gpu(handle, output))),
        Err(_) => Ok(None),
    }
}

fn wrap_downsampled_gpu(
    source: &runmat_accelerate_api::GpuTensorHandle,
    output: runmat_accelerate_api::GpuTensorHandle,
) -> Value {
    if let Some(precision) = runmat_accelerate_api::handle_precision(source) {
        runmat_accelerate_api::set_handle_precision(&output, precision);
    }
    if runmat_accelerate_api::handle_is_logical(source) {
        gpu_helpers::logical_gpu_value(output)
    } else {
        runmat_accelerate_api::set_handle_storage(
            &output,
            runmat_accelerate_api::handle_storage(source),
        );
        gpu_helpers::resident_gpu_value(output)
    }
}

fn downsample_linear_indices(
    shape: &[usize],
    dim: usize,
    input_len: usize,
    output_len: usize,
    factor: usize,
    phase: usize,
    builtin: &'static str,
) -> BuiltinResult<Option<Vec<u32>>> {
    let output_shape_count = {
        let mut output_shape = shape.to_vec();
        output_shape[dim] = output_len;
        checked_element_count(&output_shape)
            .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?
    };
    if output_shape_count > u32::MAX as usize {
        return Ok(None);
    }

    let leading = checked_product(&shape[..dim])
        .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let trailing = checked_product(&shape[dim + 1..])
        .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let mut indices = Vec::new();
    indices
        .try_reserve_exact(output_shape_count)
        .map_err(|_| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;

    for trail in 0..trailing {
        for output_pos in 0..output_len {
            let input_pos = output_pos
                .checked_mul(factor)
                .and_then(|offset| offset.checked_add(phase))
                .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
            let trail_offset = input_len
                .checked_mul(trail)
                .and_then(|offset| offset.checked_add(input_pos))
                .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
            for before in 0..leading {
                let Some(src) = before.checked_add(
                    leading
                        .checked_mul(trail_offset)
                        .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?,
                ) else {
                    return Err(sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW));
                };
                let Ok(src) = u32::try_from(src) else {
                    return Ok(None);
                };
                indices.push(src);
            }
        }
    }

    Ok(Some(indices))
}

fn resample_column_major<T: Copy>(
    data: Vec<T>,
    shape: Vec<usize>,
    factor: usize,
    phase: usize,
    op: SampleOp,
    zero: T,
    builtin: &'static str,
) -> BuiltinResult<(Vec<T>, Vec<usize>)> {
    let shape = canonical_shape(shape, data.len());
    let dim = first_non_singleton_dim(&shape);
    let input_len = shape[dim];
    let output_len = output_len(input_len, factor, phase, op)
        .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let mut output_shape = shape.clone();
    output_shape[dim] = output_len;
    let output_count = checked_element_count(&output_shape)
        .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_count)
        .map_err(|_| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    output.resize(output_count, zero);
    if input_len == 0 || output_len == 0 {
        return Ok((output, output_shape));
    }

    let leading = checked_product(&shape[..dim])
        .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let trailing = checked_product(&shape[dim + 1..])
        .ok_or_else(|| sample_error(builtin, &SAMPLE_ERROR_SIZE_OVERFLOW))?;

    for trail in 0..trailing {
        for before in 0..leading {
            match op {
                SampleOp::Up => {
                    for input_pos in 0..input_len {
                        let output_pos = phase + input_pos * factor;
                        let src = before + leading * (input_pos + input_len * trail);
                        let dst = before + leading * (output_pos + output_len * trail);
                        output[dst] = data[src];
                    }
                }
                SampleOp::Down => {
                    for output_pos in 0..output_len {
                        let input_pos = phase + output_pos * factor;
                        let src = before + leading * (input_pos + input_len * trail);
                        let dst = before + leading * (output_pos + output_len * trail);
                        output[dst] = data[src];
                    }
                }
            }
        }
    }

    Ok((output, output_shape))
}

trait ResampleSample: Copy {
    fn zero() -> Self;
    fn add_scaled(self, sample: Self, coeff: f64) -> Self;
}

impl ResampleSample for f64 {
    fn zero() -> Self {
        0.0
    }

    fn add_scaled(self, sample: Self, coeff: f64) -> Self {
        self + sample * coeff
    }
}

impl ResampleSample for (f64, f64) {
    fn zero() -> Self {
        (0.0, 0.0)
    }

    fn add_scaled(self, sample: Self, coeff: f64) -> Self {
        (self.0 + sample.0 * coeff, self.1 + sample.1 * coeff)
    }
}

fn resample_rational_column_major<T: ResampleSample>(
    data: Vec<T>,
    shape: Vec<usize>,
    p: usize,
    q: usize,
    options: &ResampleOptions,
) -> BuiltinResult<(Vec<T>, Vec<usize>)> {
    let mut shape = canonical_shape(shape, data.len());
    let dim = options
        .dim
        .unwrap_or_else(|| first_non_singleton_dim(&shape));
    if dim >= shape.len() {
        shape.resize(dim + 1, 1);
    }
    let input_len = shape[dim];
    let output_len = resample_output_len(input_len, p, q)
        .ok_or_else(|| sample_error(RESAMPLE_NAME, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let mut output_shape = shape.clone();
    output_shape[dim] = output_len;
    let output_count = checked_element_count(&output_shape)
        .ok_or_else(|| sample_error(RESAMPLE_NAME, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_count)
        .map_err(|_| sample_error(RESAMPLE_NAME, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    output.resize(output_count, T::zero());
    if input_len == 0 || output_len == 0 {
        return Ok((output, output_shape));
    }

    let leading = checked_product(&shape[..dim])
        .ok_or_else(|| sample_error(RESAMPLE_NAME, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let trailing = checked_product(&shape[dim + 1..])
        .ok_or_else(|| sample_error(RESAMPLE_NAME, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let delay = options.filter.len() / 2;

    for trail in 0..trailing {
        for before in 0..leading {
            for output_pos in 0..output_len {
                let Some(center) = output_pos
                    .checked_mul(q)
                    .and_then(|value| value.checked_add(delay))
                else {
                    return Err(sample_error(RESAMPLE_NAME, &SAMPLE_ERROR_SIZE_OVERFLOW));
                };
                let mut acc = T::zero();
                for (tap, coeff) in options.filter.iter().enumerate() {
                    if tap > center {
                        break;
                    }
                    let upsampled_pos = center - tap;
                    if upsampled_pos % p != 0 {
                        continue;
                    }
                    let input_pos = upsampled_pos / p;
                    if input_pos >= input_len {
                        continue;
                    }
                    let src = before + leading * (input_pos + input_len * trail);
                    acc = acc.add_scaled(data[src], *coeff);
                }
                let dst = before + leading * (output_pos + output_len * trail);
                output[dst] = acc;
            }
        }
    }

    Ok((output, output_shape))
}

fn resample_output_len(input_len: usize, p: usize, q: usize) -> Option<usize> {
    input_len
        .checked_mul(p)?
        .checked_add(q.checked_sub(1)?)?
        .checked_div(q)
}

fn design_resample_filter(p: usize, q: usize, n: usize, beta: f64) -> BuiltinResult<Vec<f64>> {
    let max_factor = p.max(q);
    let half_len = n
        .checked_mul(max_factor)
        .ok_or_else(|| sample_error(RESAMPLE_NAME, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let len = half_len
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| sample_error(RESAMPLE_NAME, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let cutoff = 1.0 / max_factor as f64;
    let mut filter = Vec::new();
    filter
        .try_reserve_exact(len)
        .map_err(|_| sample_error(RESAMPLE_NAME, &SAMPLE_ERROR_SIZE_OVERFLOW))?;
    let denom = modified_bessel_i0(beta);
    for idx in 0..len {
        let centered = idx as isize - half_len as isize;
        let sinc = if centered == 0 {
            cutoff
        } else {
            let x = std::f64::consts::PI * cutoff * centered as f64;
            x.sin() / (std::f64::consts::PI * centered as f64)
        };
        let window = if half_len == 0 {
            1.0
        } else {
            let ratio = centered as f64 / half_len as f64;
            modified_bessel_i0(beta * (1.0 - ratio * ratio).max(0.0).sqrt()) / denom
        };
        filter.push(sinc * window);
    }
    let sum = filter.iter().sum::<f64>();
    if !sum.is_finite() || sum == 0.0 {
        return Err(sample_error_with_detail(
            RESAMPLE_NAME,
            &SAMPLE_ERROR_INVALID_OPTION,
            "designed filter has zero or nonfinite gain",
        ));
    }
    let gain = p as f64 / sum;
    for coeff in &mut filter {
        *coeff *= gain;
    }
    Ok(filter)
}

fn modified_bessel_i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let y = (x / 3.75).powi(2);
        1.0 + y
            * (3.515_622_9
                + y * (3.089_942_4
                    + y * (1.206_749_2 + y * (0.265_973_2 + y * (0.036_076_8 + y * 0.004_581_3)))))
    } else {
        let y = 3.75 / ax;
        (ax.exp() / ax.sqrt())
            * (0.398_942_28
                + y * (0.013_285_92
                    + y * (0.002_253_19
                        + y * (-0.001_575_65
                            + y * (0.009_162_81
                                + y * (-0.020_577_06
                                    + y * (0.026_355_37
                                        + y * (-0.016_476_33 + y * 0.003_923_77))))))))
    }
}

fn filter_tensor(filter: Vec<f64>) -> BuiltinResult<Tensor> {
    let len = filter.len();
    Tensor::new(filter, vec![1, len])
        .map_err(|err| sample_error_with_detail(RESAMPLE_NAME, &SAMPLE_ERROR_BUILD_OUTPUT, err))
}

fn value_as_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::CharArray(chars) if chars.rows == 1 => Some(chars.data.iter().collect()),
        _ => None,
    }
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let next = a % b;
        a = b;
        b = next;
    }
    a.max(1)
}

fn canonical_shape(shape: Vec<usize>, len: usize) -> Vec<usize> {
    if shape.is_empty() {
        if len == 0 {
            vec![0, 1]
        } else {
            vec![1, 1]
        }
    } else {
        shape
    }
}

fn first_non_singleton_dim(shape: &[usize]) -> usize {
    shape.iter().position(|dim| *dim != 1).unwrap_or(0)
}

fn output_len(input_len: usize, factor: usize, phase: usize, op: SampleOp) -> Option<usize> {
    match op {
        SampleOp::Up => input_len.checked_mul(factor),
        SampleOp::Down => {
            if input_len <= phase {
                Some(0)
            } else {
                Some(((input_len - 1 - phase) / factor) + 1)
            }
        }
    }
}

fn checked_element_count(shape: &[usize]) -> Option<usize> {
    checked_product(shape)
}

fn checked_product(values: &[usize]) -> Option<usize> {
    values
        .iter()
        .try_fold(1usize, |acc, value| acc.checked_mul(*value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;

    fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new(data, shape).unwrap())
    }

    fn call_upsample(args: Vec<Value>) -> Value {
        block_on(upsample_builtin(
            args[0].clone(),
            args[1].clone(),
            args[2..].to_vec(),
        ))
        .unwrap()
    }

    fn call_downsample(args: Vec<Value>) -> Value {
        block_on(downsample_builtin(
            args[0].clone(),
            args[1].clone(),
            args[2..].to_vec(),
        ))
        .unwrap()
    }

    fn call_resample(args: Vec<Value>) -> Value {
        block_on(resample_builtin(
            args[0].clone(),
            args[1].clone(),
            args[2].clone(),
            args[3..].to_vec(),
        ))
        .unwrap()
    }

    #[test]
    fn upsample_row_vector_inserts_zeros_after_each_sample() {
        let out = call_upsample(vec![
            tensor(vec![1.0, 2.0, 3.0], vec![1, 3]),
            Value::Num(2.0),
        ]);
        let Value::Tensor(upsampled) = out else {
            panic!("expected tensor");
        };
        assert_eq!(upsampled.shape, vec![1, 6]);
        assert_eq!(upsampled.data, vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn upsample_column_vector_honors_phase() {
        let out = call_upsample(vec![
            tensor(vec![1.0, 2.0, 3.0], vec![3, 1]),
            Value::Num(3.0),
            Value::Num(1.0),
        ]);
        let Value::Tensor(tensor) = out else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![9, 1]);
        assert_eq!(
            tensor.data,
            vec![0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0]
        );
    }

    #[test]
    fn downsample_row_vector_keeps_every_nth_sample() {
        let out = call_downsample(vec![
            tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5]),
            Value::Num(2.0),
        ]);
        let Value::Tensor(tensor) = out else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![1, 3]);
        assert_eq!(tensor.data, vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn downsample_phase_offsets_first_kept_sample() {
        let out = call_downsample(vec![
            tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5]),
            Value::Num(2.0),
            Value::Num(1.0),
        ]);
        let Value::Tensor(tensor) = out else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![1, 2]);
        assert_eq!(tensor.data, vec![2.0, 4.0]);
    }

    #[test]
    fn downsample_gpu_uses_provider_linear_gather_and_preserves_residency() {
        test_support::with_test_provider(|provider| {
            let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5]).unwrap();
            let handle = provider
                .upload(&HostTensorView {
                    data: &input.data,
                    shape: &input.shape,
                })
                .expect("upload input");
            provider.reset_telemetry();

            let out = call_downsample(vec![
                Value::GpuTensor(handle.clone()),
                Value::Num(2.0),
                Value::Num(1.0),
            ]);
            let Value::GpuTensor(out_handle) = out else {
                panic!("expected resident gpu tensor");
            };
            assert_eq!(out_handle.shape, vec![1, 2]);
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

            let gathered =
                test_support::gather(Value::GpuTensor(out_handle)).expect("gather output");
            assert_eq!(gathered.shape, vec![1, 2]);
            assert_eq!(gathered.data, vec![2.0, 4.0]);
            let _ = provider.free(&handle);
        });
    }

    #[test]
    fn sample_rate_operates_down_matrix_columns() {
        let out = call_upsample(vec![
            tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
            Value::Num(2.0),
        ]);
        let Value::Tensor(tensor) = out else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![4, 2]);
        assert_eq!(tensor.data, vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0]);

        let down = call_downsample(vec![Value::Tensor(tensor), Value::Num(2.0)]);
        let Value::Tensor(tensor) = down else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn sample_rate_operates_along_first_non_singleton_nd_dimension() {
        let out = call_upsample(vec![
            tensor(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]),
            Value::Num(2.0),
        ]);
        let Value::Tensor(tensor) = out else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![1, 4, 2]);
        assert_eq!(tensor.data, vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0]);
    }

    #[test]
    fn sample_rate_preserves_empty_shape() {
        let out = call_upsample(vec![tensor(Vec::new(), vec![1, 0]), Value::Num(3.0)]);
        let Value::Tensor(array) = out else {
            panic!("expected tensor");
        };
        assert_eq!(array.shape, vec![1, 0]);
        assert!(array.data.is_empty());

        let out = call_downsample(vec![tensor(Vec::new(), vec![0, 1]), Value::Num(2.0)]);
        let Value::Tensor(array) = out else {
            panic!("expected tensor");
        };
        assert_eq!(array.shape, vec![0, 1]);
        assert!(array.data.is_empty());
    }

    #[test]
    fn sample_rate_preserves_complex_values() {
        let input = Value::ComplexTensor(
            ComplexTensor::new(vec![(1.0, 2.0), (3.0, 4.0)], vec![1, 2]).unwrap(),
        );
        let out = call_upsample(vec![input, Value::Num(2.0)]);
        let Value::ComplexTensor(tensor) = out else {
            panic!("expected complex tensor");
        };
        assert_eq!(tensor.shape, vec![1, 4]);
        assert_eq!(
            tensor.data,
            vec![(1.0, 2.0), (0.0, 0.0), (3.0, 4.0), (0.0, 0.0)]
        );
    }

    #[test]
    fn sample_rate_rejects_invalid_factor_and_phase() {
        let err = block_on(upsample_builtin(
            tensor(vec![1.0, 2.0], vec![1, 2]),
            Value::Num(0.0),
            Vec::new(),
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), SAMPLE_ERROR_INVALID_FACTOR.identifier);

        let err = block_on(upsample_builtin(
            tensor(vec![1.0, 2.0], vec![1, 2]),
            Value::Num(2.0000004),
            Vec::new(),
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), SAMPLE_ERROR_INVALID_FACTOR.identifier);

        let err = block_on(downsample_builtin(
            tensor(vec![1.0, 2.0], vec![1, 2]),
            Value::Num(2.0),
            vec![Value::Num(2.0)],
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), SAMPLE_ERROR_INVALID_PHASE.identifier);

        let err = block_on(downsample_builtin(
            tensor(vec![1.0, 2.0], vec![1, 2]),
            Value::Num(2.0),
            vec![Value::Num(1.0000004)],
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), SAMPLE_ERROR_INVALID_PHASE.identifier);
    }

    #[test]
    fn resample_custom_unit_filter_matches_zero_insert_and_drop() {
        let out = call_resample(vec![
            tensor(vec![1.0, 2.0, 3.0], vec![1, 3]),
            Value::Num(2.0),
            Value::Num(1.0),
            tensor(vec![0.0, 1.0, 0.0], vec![1, 3]),
        ]);
        let Value::Tensor(upsampled) = out else {
            panic!("expected tensor");
        };
        assert_eq!(upsampled.shape, vec![1, 6]);
        assert_eq!(upsampled.data, vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]);

        let down = call_resample(vec![
            tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5]),
            Value::Num(1.0),
            Value::Num(2.0),
            tensor(vec![0.0, 1.0, 0.0], vec![1, 3]),
        ]);
        let Value::Tensor(downsampled) = down else {
            panic!("expected tensor");
        };
        assert_eq!(downsampled.shape, vec![1, 3]);
        assert_eq!(downsampled.data, vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn resample_dimension_name_value_operates_on_requested_axis() {
        let out = call_resample(vec![
            tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
            Value::Num(2.0),
            Value::Num(1.0),
            tensor(vec![0.0, 1.0, 0.0], vec![1, 3]),
            Value::CharArray(runmat_builtins::CharArray::new_row("Dimension")),
            Value::Num(2.0),
        ]);
        let Value::Tensor(tensor) = out else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![2, 4]);
        assert_eq!(tensor.data, vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0]);
    }

    #[test]
    fn resample_preserves_complex_domain() {
        let input = Value::ComplexTensor(
            ComplexTensor::new(vec![(1.0, 2.0), (3.0, 4.0)], vec![1, 2]).unwrap(),
        );
        let out = call_resample(vec![
            input,
            Value::Num(2.0),
            Value::Num(1.0),
            tensor(vec![0.0, 1.0, 0.0], vec![1, 3]),
        ]);
        let Value::ComplexTensor(tensor) = out else {
            panic!("expected complex tensor");
        };
        assert_eq!(tensor.shape, vec![1, 4]);
        assert_eq!(
            tensor.data,
            vec![(1.0, 2.0), (0.0, 0.0), (3.0, 4.0), (0.0, 0.0)]
        );
    }

    #[test]
    fn resample_two_outputs_return_signal_and_filter() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let result = block_on(resample_builtin(
            tensor(vec![1.0, 2.0, 3.0], vec![3, 1]),
            Value::Num(3.0),
            Value::Num(2.0),
            vec![Value::Num(1.0), Value::Num(0.0)],
        ))
        .unwrap();
        let Value::OutputList(values) = result else {
            panic!("expected output list");
        };
        assert_eq!(values.len(), 2);
        let Value::Tensor(signal) = &values[0] else {
            panic!("expected signal tensor");
        };
        let Value::Tensor(filter) = &values[1] else {
            panic!("expected filter tensor");
        };
        assert_eq!(signal.shape, vec![5, 1]);
        assert_eq!(filter.shape, vec![1, 7]);
        let sum = filter.data.iter().sum::<f64>();
        assert!((sum - 3.0).abs() < 1e-10);
    }

    #[test]
    fn resample_default_filter_has_matlab_compatible_length() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let result = block_on(resample_builtin(
            tensor(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]),
            Value::Num(4.0),
            Value::Num(2.0),
            Vec::new(),
        ))
        .unwrap();
        let Value::OutputList(values) = result else {
            panic!("expected output list");
        };
        let Value::Tensor(signal) = &values[0] else {
            panic!("expected signal tensor");
        };
        let Value::Tensor(filter) = &values[1] else {
            panic!("expected filter tensor");
        };
        assert_eq!(signal.shape, vec![1, 8]);
        assert_eq!(filter.shape, vec![1, 41]);
        let sum = filter.data.iter().sum::<f64>();
        assert!((sum - 2.0).abs() < 1e-10);
    }

    #[test]
    fn resample_filter_design_rejects_huge_allocation() {
        let err = design_resample_filter(1usize << (usize::BITS - 3), 1, 1, 5.0).unwrap_err();
        assert_eq!(err.identifier(), SAMPLE_ERROR_SIZE_OVERFLOW.identifier);
    }

    #[test]
    fn resample_rejects_invalid_options_and_output_count() {
        let err = block_on(resample_builtin(
            tensor(vec![1.0, 2.0], vec![1, 2]),
            Value::Num(2.0),
            Value::Num(0.0),
            Vec::new(),
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), SAMPLE_ERROR_INVALID_FACTOR.identifier);

        let err = block_on(resample_builtin(
            tensor(vec![1.0, 2.0], vec![1, 2]),
            Value::Num(2.0),
            Value::Num(1.0),
            vec![Value::CharArray(runmat_builtins::CharArray::new_row(
                "Dimension",
            ))],
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), SAMPLE_ERROR_INVALID_OPTION.identifier);

        let _guard = crate::output_count::push_output_count(Some(3));
        let err = block_on(resample_builtin(
            tensor(vec![1.0, 2.0], vec![1, 2]),
            Value::Num(2.0),
            Value::Num(1.0),
            Vec::new(),
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), SAMPLE_ERROR_OUTPUT_COUNT.identifier);
    }
}
