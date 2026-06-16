//! MATLAB-compatible `upsample` and `downsample` builtins.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, NumericDType, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::builtins::math::signal::type_resolvers::{downsample_type, upsample_type};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const UPSAMPLE_NAME: &str = "upsample";
const DOWNSAMPLE_NAME: &str = "downsample";

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

const SAMPLE_ERRORS: [BuiltinErrorDescriptor; 7] = [
    SAMPLE_ERROR_ARG_COUNT,
    SAMPLE_ERROR_INVALID_INPUT,
    SAMPLE_ERROR_INVALID_FACTOR,
    SAMPLE_ERROR_INVALID_PHASE,
    SAMPLE_ERROR_GATHER_FAILED,
    SAMPLE_ERROR_SIZE_OVERFLOW,
    SAMPLE_ERROR_BUILD_OUTPUT,
];

pub const UPSAMPLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &UPSAMPLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SAMPLE_ERRORS,
};

pub const DOWNSAMPLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DOWNSAMPLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SAMPLE_ERRORS,
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
    let input = SampleInput::from_value(x, builtin).await?;
    apply_sample_rate(input, factor, phase, op, builtin)
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
    use futures::executor::block_on;

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

    #[test]
    fn upsample_row_vector_inserts_zeros_after_each_sample() {
        let out = call_upsample(vec![
            tensor(vec![1.0, 2.0, 3.0], vec![1, 3]),
            Value::Num(2.0),
        ]);
        let Value::Tensor(tensor) = out else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![1, 6]);
        assert_eq!(tensor.data, vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]);
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
}
