//! MATLAB-compatible `unwrap` builtin for phase-continuity correction.

use std::f64::consts::PI;

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::fft::common::default_dimension;
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "unwrap";
const TWO_PI: f64 = 2.0 * PI;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::signal::unwrap")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: BUILTIN_NAME,
    op_kind: GpuOpKind::Custom("phase-unwrap"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Corrects phase discontinuities along one dimension. GPU inputs gather through the active provider until dedicated provider hooks are available.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::signal::unwrap")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: BUILTIN_NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Phase unwrapping is stateful along a dimension and is not fusible as an element-wise operation.",
};

const UNWRAP_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Q",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Phase values with jumps larger than the tolerance corrected by multiples of 2*pi.",
}];

const UNWRAP_INPUTS_CORE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "P",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Wrapped phase angles in radians.",
}];

const UNWRAP_INPUTS_WITH_TOL: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "P",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Wrapped phase angles in radians.",
    },
    BuiltinParamDescriptor {
        name: "tol",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("pi"),
        description: "Jump tolerance. Values below pi are treated as pi.",
    },
];

const UNWRAP_INPUTS_WITH_TOL_DIM: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "P",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Wrapped phase angles in radians.",
    },
    BuiltinParamDescriptor {
        name: "tol",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("pi"),
        description: "Jump tolerance. Use [] to keep the default.",
    },
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("first non-singleton dimension"),
        description: "Dimension along which to unwrap.",
    },
];

const UNWRAP_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "Q = unwrap(P)",
        inputs: &UNWRAP_INPUTS_CORE,
        outputs: &UNWRAP_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Q = unwrap(P, tol)",
        inputs: &UNWRAP_INPUTS_WITH_TOL,
        outputs: &UNWRAP_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Q = unwrap(P, tol, dim)",
        inputs: &UNWRAP_INPUTS_WITH_TOL_DIM,
        outputs: &UNWRAP_OUTPUT,
    },
];

const UNWRAP_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNWRAP.ARG_COUNT",
    identifier: Some("RunMat:unwrap:ArgCount"),
    when: "More than three input arguments are supplied.",
    message: "unwrap: expected unwrap(P), unwrap(P, tol), or unwrap(P, tol, dim)",
};

const UNWRAP_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNWRAP.INVALID_INPUT",
    identifier: Some("RunMat:unwrap:InvalidInput"),
    when: "Input cannot be converted to a real numeric/logical phase array.",
    message: "unwrap: expected real numeric input",
};

const UNWRAP_ERROR_INVALID_TOLERANCE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNWRAP.INVALID_TOLERANCE",
    identifier: Some("RunMat:unwrap:InvalidTolerance"),
    when: "Tolerance is not a finite, nonnegative scalar.",
    message: "unwrap: tolerance must be a finite nonnegative scalar",
};

const UNWRAP_ERROR_INVALID_DIMENSION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNWRAP.INVALID_DIMENSION",
    identifier: Some("RunMat:unwrap:InvalidDimension"),
    when: "Dimension argument is missing, non-numeric, non-integer, or less than one.",
    message: "unwrap: dimension must be a positive integer scalar",
};

const UNWRAP_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNWRAP.INTERNAL",
    identifier: Some("RunMat:unwrap:Internal"),
    when: "Internal gather, tensor conversion, or allocation fails.",
    message: "unwrap: internal error",
};

const UNWRAP_ERRORS: [BuiltinErrorDescriptor; 5] = [
    UNWRAP_ERROR_ARG_COUNT,
    UNWRAP_ERROR_INVALID_INPUT,
    UNWRAP_ERROR_INVALID_TOLERANCE,
    UNWRAP_ERROR_INVALID_DIMENSION,
    UNWRAP_ERROR_INTERNAL,
];

pub const UNWRAP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &UNWRAP_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &UNWRAP_ERRORS,
};

fn unwrap_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    unwrap_error_with_message(error.message, error)
}

fn unwrap_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    unwrap_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn unwrap_error_with_source(
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

fn unwrap_error_with_message(
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
    name = "unwrap",
    category = "math/signal",
    summary = "Correct phase-angle jumps by adding multiples of 2*pi.",
    keywords = "unwrap,phase,angle,signal processing,radians",
    type_resolver(numeric_unary_type),
    descriptor(crate::builtins::math::signal::unwrap::UNWRAP_DESCRIPTOR),
    builtin_path = "crate::builtins::math::signal::unwrap"
)]
async fn unwrap_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let options = parse_arguments(&rest).await?;
    match value {
        Value::GpuTensor(handle) => unwrap_gpu(handle, options).await,
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(unwrap_error_with_detail(
            &UNWRAP_ERROR_INVALID_INPUT,
            "input must be real-valued phase angles",
        )),
        other => unwrap_host(other, options),
    }
}

#[derive(Clone, Copy, Debug)]
struct UnwrapOptions {
    tolerance: f64,
    dimension: Option<usize>,
}

async fn parse_arguments(args: &[Value]) -> BuiltinResult<UnwrapOptions> {
    match args.len() {
        0 => Ok(UnwrapOptions {
            tolerance: PI,
            dimension: None,
        }),
        1 => Ok(UnwrapOptions {
            tolerance: parse_tolerance(&args[0]).await?.unwrap_or(PI),
            dimension: None,
        }),
        2 => Ok(UnwrapOptions {
            tolerance: parse_tolerance(&args[0]).await?.unwrap_or(PI),
            dimension: parse_dimension(&args[1]).await?,
        }),
        _ => Err(unwrap_error(&UNWRAP_ERROR_ARG_COUNT)),
    }
}

async fn parse_tolerance(value: &Value) -> BuiltinResult<Option<f64>> {
    if is_empty_value(value) {
        return Ok(None);
    }
    let raw = tensor::scalar_f64_from_value_async(value)
        .await
        .map_err(|detail| unwrap_error_with_detail(&UNWRAP_ERROR_INVALID_TOLERANCE, detail))?
        .ok_or_else(|| unwrap_error(&UNWRAP_ERROR_INVALID_TOLERANCE))?;
    if !raw.is_finite() || raw < 0.0 {
        return Err(unwrap_error_with_detail(
            &UNWRAP_ERROR_INVALID_TOLERANCE,
            format!("got {raw}"),
        ));
    }
    Ok(Some(raw.max(PI)))
}

async fn parse_dimension(value: &Value) -> BuiltinResult<Option<usize>> {
    if is_empty_value(value) {
        return Ok(None);
    }
    tensor::dimension_from_value_async(value, BUILTIN_NAME, false)
        .await
        .map_err(|detail| unwrap_error_with_detail(&UNWRAP_ERROR_INVALID_DIMENSION, detail))?
        .map(Some)
        .ok_or_else(|| unwrap_error(&UNWRAP_ERROR_INVALID_DIMENSION))
}

fn is_empty_value(value: &Value) -> bool {
    match value {
        Value::Tensor(t) => t.data.is_empty(),
        Value::LogicalArray(l) => l.data.is_empty(),
        Value::ComplexTensor(t) => t.data.is_empty(),
        _ => false,
    }
}

async fn unwrap_gpu(handle: GpuTensorHandle, options: UnwrapOptions) -> BuiltinResult<Value> {
    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|source| {
            unwrap_error_with_source(&UNWRAP_ERROR_INVALID_INPUT, "gpu gather failed", source)
        })?;
    unwrap_tensor(tensor, options).map(tensor::tensor_into_value)
}

fn unwrap_host(value: Value, options: UnwrapOptions) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value)
        .map_err(|detail| unwrap_error_with_detail(&UNWRAP_ERROR_INVALID_INPUT, detail))?;
    unwrap_tensor(tensor, options).map(tensor::tensor_into_value)
}

fn unwrap_tensor(tensor: Tensor, options: UnwrapOptions) -> BuiltinResult<Tensor> {
    let Tensor {
        data,
        mut shape,
        dtype,
        ..
    } = tensor;
    if crate::builtins::common::shape::is_scalar_shape(&shape) {
        shape = crate::builtins::common::shape::normalize_scalar_shape(&shape);
    }
    let dim_index = match options.dimension {
        Some(0) => return Err(unwrap_error(&UNWRAP_ERROR_INVALID_DIMENSION)),
        Some(dim) => dim - 1,
        None => default_dimension(&shape) - 1,
    };

    let mut logical_shape = shape.clone();
    while logical_shape.len() <= dim_index {
        logical_shape.push(1);
    }

    let len = logical_shape[dim_index];
    if len <= 1 || data.is_empty() {
        return Tensor::new_with_dtype(data, shape, dtype)
            .map_err(|err| unwrap_error_with_detail(&UNWRAP_ERROR_INTERNAL, err));
    }

    let inner_stride = checked_product(&logical_shape[..dim_index])?;
    let outer_stride = checked_product(&logical_shape[dim_index + 1..])?;
    let slice_span = len
        .checked_mul(inner_stride)
        .ok_or_else(|| unwrap_error_with_detail(&UNWRAP_ERROR_INTERNAL, "slice span overflow"))?;

    let mut output = data.clone();
    for outer in 0..outer_stride {
        let base = outer.checked_mul(slice_span).ok_or_else(|| {
            unwrap_error_with_detail(&UNWRAP_ERROR_INTERNAL, "slice offset overflow")
        })?;
        for inner in 0..inner_stride {
            let first_idx = base + inner;
            if first_idx >= data.len() {
                continue;
            }
            let mut correction = 0.0;
            let mut previous = data[first_idx];
            output[first_idx] = previous;
            for k in 1..len {
                let idx = base + inner + k * inner_stride;
                if idx >= data.len() {
                    break;
                }
                let current = data[idx];
                if current.is_finite() && previous.is_finite() {
                    let delta = current - previous;
                    if delta.abs() > options.tolerance {
                        correction += principal_phase_delta(delta) - delta;
                    }
                }
                output[idx] = current + correction;
                previous = current;
            }
        }
    }

    Tensor::new_with_dtype(output, shape, dtype)
        .map_err(|err| unwrap_error_with_detail(&UNWRAP_ERROR_INTERNAL, err))
}

fn checked_product(dims: &[usize]) -> BuiltinResult<usize> {
    dims.iter().copied().try_fold(1usize, |acc, dim| {
        acc.checked_mul(dim).ok_or_else(|| {
            unwrap_error_with_detail(&UNWRAP_ERROR_INTERNAL, "shape product overflow")
        })
    })
}

fn principal_phase_delta(delta: f64) -> f64 {
    let mut wrapped = (delta + PI).rem_euclid(TWO_PI) - PI;
    if wrapped == -PI && delta > 0.0 {
        wrapped = PI;
    }
    wrapped
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{ComplexTensor, LogicalArray, ResolveContext, Type};

    const TOL: f64 = 1.0e-12;

    fn unwrap_call(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(unwrap_builtin(value, rest))
    }

    fn as_tensor(value: Value) -> Tensor {
        match value {
            Value::Tensor(tensor) => tensor,
            Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
            other => panic!("expected tensor output, got {other:?}"),
        }
    }

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (&a, &e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() <= TOL, "actual={a} expected={e}");
        }
    }

    #[test]
    fn unwrap_type_preserves_numeric_shape() {
        let out = numeric_unary_type(
            &[Type::Tensor {
                shape: Some(vec![Some(1), Some(3)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(1), Some(3)])
            }
        );
    }

    #[test]
    fn unwrap_row_vector_corrects_two_pi_jump() {
        let input =
            Tensor::new(vec![0.0, 1.0, 2.0, 2.0 - TWO_PI, 3.0 - TWO_PI], vec![1, 5]).unwrap();
        let out = as_tensor(unwrap_call(Value::Tensor(input), Vec::new()).unwrap());
        assert_eq!(out.shape, vec![1, 5]);
        assert_close(&out.data, &[0.0, 1.0, 2.0, 2.0, 3.0]);
    }

    #[test]
    fn unwrap_matrix_defaults_down_columns() {
        let input = Tensor::new(
            vec![0.0, 1.0, 1.0 - TWO_PI, 0.0, -1.0, -1.0 + TWO_PI],
            vec![3, 2],
        )
        .unwrap();
        let out = as_tensor(unwrap_call(Value::Tensor(input), Vec::new()).unwrap());
        assert_eq!(out.shape, vec![3, 2]);
        assert_close(&out.data, &[0.0, 1.0, 1.0, 0.0, -1.0, -1.0]);
    }

    #[test]
    fn unwrap_dimension_argument_operates_across_rows() {
        let input = Tensor::new(
            vec![0.0, 0.0, 1.0, -1.0, 1.0 - TWO_PI, -1.0 + TWO_PI],
            vec![2, 3],
        )
        .unwrap();
        let out = as_tensor(
            unwrap_call(
                Value::Tensor(input),
                vec![
                    Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
                    Value::Num(2.0),
                ],
            )
            .unwrap(),
        );
        assert_eq!(out.shape, vec![2, 3]);
        assert_close(&out.data, &[0.0, 0.0, 1.0, -1.0, 1.0, -1.0]);
    }

    #[test]
    fn unwrap_high_tolerance_preserves_large_jump() {
        let input = Tensor::new(vec![0.0, 1.0, 1.0 - TWO_PI], vec![1, 3]).unwrap();
        let out = as_tensor(unwrap_call(Value::Tensor(input), vec![Value::Num(10.0)]).unwrap());
        assert_eq!(out.shape, vec![1, 3]);
        assert_close(&out.data, &[0.0, 1.0, 1.0 - TWO_PI]);
    }

    #[test]
    fn unwrap_nonfinite_values_break_continuity_correction() {
        let input =
            Tensor::new(vec![0.0, f64::NAN, 1.0 - TWO_PI, 2.0 - TWO_PI], vec![1, 4]).unwrap();
        let out = as_tensor(unwrap_call(Value::Tensor(input), Vec::new()).unwrap());
        assert_eq!(out.shape, vec![1, 4]);
        assert_eq!(out.data[0], 0.0);
        assert!(out.data[1].is_nan());
        assert_close(&out.data[2..], &[1.0 - TWO_PI, 2.0 - TWO_PI]);
    }

    #[test]
    fn unwrap_accepts_logical_input() {
        let input = LogicalArray::new(vec![0, 1, 0], vec![1, 3]).unwrap();
        let out = as_tensor(unwrap_call(Value::LogicalArray(input), Vec::new()).unwrap());
        assert_eq!(out.shape, vec![1, 3]);
        assert_close(&out.data, &[0.0, 1.0, 0.0]);
    }

    #[test]
    fn unwrap_rejects_complex_input() {
        let input = ComplexTensor::new(vec![(1.0, 1.0)], vec![1, 1]).unwrap();
        let err = unwrap_call(Value::ComplexTensor(input), Vec::new()).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:unwrap:InvalidInput"));
    }

    #[test]
    fn unwrap_rejects_invalid_tolerance() {
        let input = Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap();
        let err = unwrap_call(Value::Tensor(input), vec![Value::Num(f64::INFINITY)]).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:unwrap:InvalidTolerance"));
    }

    #[test]
    fn unwrap_gpu_input_gathers_and_corrects_phase() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let input = Tensor::new(vec![0.0, 1.0, 1.0 - TWO_PI], vec![1, 3]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &input.data,
                shape: &input.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let out = as_tensor(unwrap_call(Value::GpuTensor(handle), Vec::new()).unwrap());
            assert_eq!(out.shape, vec![1, 3]);
            assert_close(&out.data, &[0.0, 1.0, 1.0]);
        });
    }
}
