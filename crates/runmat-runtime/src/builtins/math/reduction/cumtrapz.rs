//! MATLAB-compatible `cumtrapz` builtin for cumulative trapezoidal integration.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, ResolveContext, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::math::reduction::integration_common::{
    canonical_shape_complex, canonical_shape_tensor, default_dimension_from_shape, dim_product,
    gather_host_value, interval_width, is_dimension_candidate, is_scalar_like, pad_shape_for_dim,
    parse_optional_dim, promote_real_value_to_gpu, spacing_from_value, value_has_gpu_tensor,
    value_into_complex_tensor, SpacingSpec,
};
use crate::builtins::math::reduction::type_resolvers::cumulative_numeric_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "cumtrapz";

const CUMTRAPZ_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Q",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cumulative trapezoidal integral output.",
}];

const CUMTRAPZ_INPUTS_Y: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sample values.",
}];

const CUMTRAPZ_INPUTS_Y_DIM: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample values.",
    },
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integration dimension.",
    },
];

const CUMTRAPZ_INPUTS_X_Y: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample points or spacing.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample values.",
    },
];

const CUMTRAPZ_INPUTS_X_Y_DIM: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample points or spacing.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample values.",
    },
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Integration dimension.",
    },
];

const CUMTRAPZ_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "Q = cumtrapz(Y)",
        inputs: &CUMTRAPZ_INPUTS_Y,
        outputs: &CUMTRAPZ_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Q = cumtrapz(Y, dim)",
        inputs: &CUMTRAPZ_INPUTS_Y_DIM,
        outputs: &CUMTRAPZ_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Q = cumtrapz(X, Y)",
        inputs: &CUMTRAPZ_INPUTS_X_Y,
        outputs: &CUMTRAPZ_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Q = cumtrapz(X, Y, dim)",
        inputs: &CUMTRAPZ_INPUTS_X_Y_DIM,
        outputs: &CUMTRAPZ_OUTPUT,
    },
];

const CUMTRAPZ_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CUMTRAPZ.INVALID_ARGUMENT",
    identifier: Some("RunMat:cumtrapz:InvalidArgument"),
    when: "Input argument count, dimension selector, or spacing arguments are invalid.",
    message: "cumtrapz: invalid argument",
};

const CUMTRAPZ_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CUMTRAPZ.INVALID_INPUT",
    identifier: Some("RunMat:cumtrapz:InvalidInput"),
    when: "Input values cannot be converted to supported numeric integration domains.",
    message: "cumtrapz: invalid input",
};

const CUMTRAPZ_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CUMTRAPZ.INTERNAL",
    identifier: Some("RunMat:cumtrapz:Internal"),
    when: "Integration execution fails during gather, allocation, or provider promotion.",
    message: "cumtrapz: internal integration failure",
};

const CUMTRAPZ_ERRORS: [BuiltinErrorDescriptor; 3] = [
    CUMTRAPZ_ERROR_INVALID_ARGUMENT,
    CUMTRAPZ_ERROR_INVALID_INPUT,
    CUMTRAPZ_ERROR_INTERNAL,
];

pub const CUMTRAPZ_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CUMTRAPZ_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CUMTRAPZ_ERRORS,
};

fn cumtrapz_type(args: &[Type], ctx: &ResolveContext) -> Type {
    cumulative_numeric_type(args, ctx)
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::reduction::cumtrapz")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "cumtrapz",
    op_kind: GpuOpKind::Custom("cumulative-trapezoidal-integral"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "GPU inputs currently gather to the host for cumulative trapezoidal integration and re-upload real-valued outputs so downstream code can remain device-resident.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::reduction::cumtrapz")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "cumtrapz",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Cumulative discrete integration currently lowers to the runtime implementation rather than fusion kernels.",
};

fn cumtrapz_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    cumtrapz_error_with_message(error.message, error)
}

fn cumtrapz_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    cumtrapz_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn cumtrapz_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn cumtrapz_internal_error(detail: impl AsRef<str>) -> RuntimeError {
    cumtrapz_error_with_detail(&CUMTRAPZ_ERROR_INTERNAL, detail)
}

#[runtime_builtin(
    name = "cumtrapz",
    category = "math/reduction",
    summary = "Cumulative trapezoidal numerical integration of sampled data.",
    keywords = "cumtrapz,cumulative trapezoidal integration,numerical integration,gpu",
    accel = "none",
    type_resolver(cumtrapz_type),
    descriptor(crate::builtins::math::reduction::cumtrapz::CUMTRAPZ_DESCRIPTOR),
    builtin_path = "crate::builtins::math::reduction::cumtrapz"
)]
async fn cumtrapz_builtin(first: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let parsed = parse_arguments(first, rest)?;
    let wants_gpu_result = value_has_gpu_tensor(&parsed.y)
        || parsed
            .spacing
            .as_ref()
            .map(value_has_gpu_tensor)
            .unwrap_or(false);

    let y_value = gather_host_value(parsed.y)
        .await
        .map_err(|err| cumtrapz_internal_error(err.message()))?;
    let spacing_value = match parsed.spacing {
        Some(value) => Some(
            gather_host_value(value)
                .await
                .map_err(|err| cumtrapz_internal_error(err.message()))?,
        ),
        None => None,
    };

    let result = match y_value {
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            let tensor = value_into_complex_tensor(NAME, y_value).map_err(|err| {
                cumtrapz_error_with_detail(&CUMTRAPZ_ERROR_INVALID_INPUT, err.message())
            })?;
            let shape = canonical_shape_complex(&tensor);
            let dim = parsed
                .dim
                .unwrap_or_else(|| default_dimension_from_shape(&shape));
            let spacing = spacing_from_value(NAME, spacing_value, &shape, dim).map_err(|err| {
                cumtrapz_error_with_detail(&CUMTRAPZ_ERROR_INVALID_ARGUMENT, err.message())
            })?;
            complex_tensor_into_value(cumtrapz_complex_tensor(&tensor, &spacing, dim)?)
        }
        other => {
            let tensor = crate::builtins::common::tensor::value_into_tensor_for(NAME, other)
                .map_err(|err| cumtrapz_error_with_detail(&CUMTRAPZ_ERROR_INVALID_INPUT, err))?;
            let shape = canonical_shape_tensor(&tensor);
            let dim = parsed
                .dim
                .unwrap_or_else(|| default_dimension_from_shape(&shape));
            let spacing = spacing_from_value(NAME, spacing_value, &shape, dim).map_err(|err| {
                cumtrapz_error_with_detail(&CUMTRAPZ_ERROR_INVALID_ARGUMENT, err.message())
            })?;
            crate::builtins::common::tensor::tensor_into_value(cumtrapz_tensor(
                &tensor, &spacing, dim,
            )?)
        }
    };

    if wants_gpu_result && !matches!(result, Value::Complex(_, _) | Value::ComplexTensor(_)) {
        promote_real_value_to_gpu(NAME, result)
            .map_err(|err| cumtrapz_internal_error(err.message()))
    } else {
        Ok(result)
    }
}

struct ParsedCumtrapzArgs {
    spacing: Option<Value>,
    y: Value,
    dim: Option<usize>,
}

fn parse_arguments(first: Value, rest: Vec<Value>) -> BuiltinResult<ParsedCumtrapzArgs> {
    match rest.len() {
        0 => Ok(ParsedCumtrapzArgs {
            spacing: None,
            y: first,
            dim: None,
        }),
        1 => {
            let second = rest.into_iter().next().expect("one arg");
            if is_dimension_candidate(&second) && !is_scalar_like(&first) {
                Ok(ParsedCumtrapzArgs {
                    spacing: None,
                    y: first,
                    dim: parse_optional_dim(NAME, &second).map_err(|err| {
                        cumtrapz_error_with_detail(&CUMTRAPZ_ERROR_INVALID_ARGUMENT, err.message())
                    })?,
                })
            } else {
                Ok(ParsedCumtrapzArgs {
                    spacing: Some(first),
                    y: second,
                    dim: None,
                })
            }
        }
        2 => {
            let mut iter = rest.into_iter();
            let y = iter.next().expect("y arg");
            let dim_arg = iter.next().expect("dim arg");
            Ok(ParsedCumtrapzArgs {
                spacing: Some(first),
                y,
                dim: parse_optional_dim(NAME, &dim_arg).map_err(|err| {
                    cumtrapz_error_with_detail(&CUMTRAPZ_ERROR_INVALID_ARGUMENT, err.message())
                })?,
            })
        }
        _ => Err(cumtrapz_error(&CUMTRAPZ_ERROR_INVALID_ARGUMENT)),
    }
}

fn cumtrapz_tensor(tensor: &Tensor, spacing: &SpacingSpec, dim: usize) -> BuiltinResult<Tensor> {
    if dim == 0 {
        return Err(cumtrapz_error_with_detail(
            &CUMTRAPZ_ERROR_INVALID_ARGUMENT,
            "dimension must be >= 1",
        ));
    }

    let shape = pad_shape_for_dim(&canonical_shape_tensor(tensor), dim);
    let dim_index = dim - 1;
    let len_dim = shape[dim_index];
    let stride_before = dim_product(&shape[..dim_index]);
    let stride_after = dim_product(&shape[dim..]);
    let block = stride_before * len_dim;
    let mut output = vec![0.0f64; tensor.data.len()];

    if len_dim > 0 {
        for after in 0..stride_after {
            let base = after * block;
            for before in 0..stride_before {
                let first_idx = base + before;
                output[first_idx] = 0.0;
                let mut acc = 0.0f64;
                for k in 0..len_dim.saturating_sub(1) {
                    let idx0 = base + before + k * stride_before;
                    let idx1 = idx0 + stride_before;
                    let width = interval_width(spacing, idx0, idx1, k);
                    acc += 0.5 * width * (tensor.data[idx0] + tensor.data[idx1]);
                    output[idx1] = acc;
                }
            }
        }
    }

    Tensor::new(output, shape).map_err(|err| cumtrapz_internal_error(err.to_string()))
}

fn cumtrapz_complex_tensor(
    tensor: &ComplexTensor,
    spacing: &SpacingSpec,
    dim: usize,
) -> BuiltinResult<ComplexTensor> {
    if dim == 0 {
        return Err(cumtrapz_error_with_detail(
            &CUMTRAPZ_ERROR_INVALID_ARGUMENT,
            "dimension must be >= 1",
        ));
    }

    let shape = pad_shape_for_dim(&canonical_shape_complex(tensor), dim);
    let dim_index = dim - 1;
    let len_dim = shape[dim_index];
    let stride_before = dim_product(&shape[..dim_index]);
    let stride_after = dim_product(&shape[dim..]);
    let block = stride_before * len_dim;
    let mut output = vec![(0.0f64, 0.0f64); tensor.data.len()];

    if len_dim > 0 {
        for after in 0..stride_after {
            let base = after * block;
            for before in 0..stride_before {
                let first_idx = base + before;
                output[first_idx] = (0.0, 0.0);
                let mut acc = (0.0f64, 0.0f64);
                for k in 0..len_dim.saturating_sub(1) {
                    let idx0 = base + before + k * stride_before;
                    let idx1 = idx0 + stride_before;
                    let width = interval_width(spacing, idx0, idx1, k);
                    let (re0, im0) = tensor.data[idx0];
                    let (re1, im1) = tensor.data[idx1];
                    acc.0 += 0.5 * width * (re0 + re1);
                    acc.1 += 0.5 * width * (im0 + im1);
                    output[idx1] = acc;
                }
            }
        }
    }

    ComplexTensor::new(output, shape).map_err(|err| cumtrapz_internal_error(err.to_string()))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::IntValue;

    fn run_cumtrapz(first: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::cumtrapz_builtin(first, rest))
    }

    #[test]
    fn cumtrapz_type_preserves_shape() {
        let out = cumtrapz_type(
            &[Type::Tensor {
                shape: Some(vec![Some(2), Some(3)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(3)])
            }
        );
    }

    #[test]
    fn cumtrapz_scalar_is_zero() {
        let value = run_cumtrapz(Value::Num(5.0), Vec::new()).expect("cumtrapz");
        assert_eq!(value, Value::Num(0.0));
    }

    #[test]
    fn cumtrapz_row_vector_unit_spacing() {
        let y = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let value = run_cumtrapz(Value::Tensor(y), Vec::new()).expect("cumtrapz");
        let Value::Tensor(out) = value else {
            panic!("expected tensor result");
        };
        assert_eq!(out.shape, vec![1, 3]);
        assert_eq!(out.data, vec![0.0, 1.5, 4.0]);
    }

    #[test]
    fn cumtrapz_nonuniform_x_vector() {
        let x = Tensor::new(vec![0.0, 1.0, 3.0], vec![1, 3]).unwrap();
        let y = Tensor::new(vec![0.0, 1.0, 2.0], vec![1, 3]).unwrap();
        let value = run_cumtrapz(Value::Tensor(x), vec![Value::Tensor(y)]).expect("cumtrapz");
        let Value::Tensor(out) = value else {
            panic!("expected tensor result");
        };
        assert_eq!(out.data, vec![0.0, 0.5, 3.5]);
    }

    #[test]
    fn cumtrapz_matrix_dimension_two() {
        let y = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let value =
            run_cumtrapz(Value::Tensor(y), vec![Value::Int(IntValue::I32(2))]).expect("cumtrapz");
        let Value::Tensor(out) = value else {
            panic!("expected tensor result");
        };
        assert_eq!(out.shape, vec![2, 3]);
        assert_eq!(out.data, vec![0.0, 0.0, 1.5, 4.5, 4.0, 10.0]);
    }

    #[test]
    fn cumtrapz_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = CUMTRAPZ_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Q = cumtrapz(Y)"));
        assert!(labels.contains(&"Q = cumtrapz(Y, dim)"));
        assert!(labels.contains(&"Q = cumtrapz(X, Y)"));
        assert!(labels.contains(&"Q = cumtrapz(X, Y, dim)"));
    }

    #[test]
    fn cumtrapz_descriptor_errors_have_stable_codes() {
        assert!(CUMTRAPZ_DESCRIPTOR
            .errors
            .iter()
            .any(|error| error.code == CUMTRAPZ_ERROR_INVALID_ARGUMENT.code));
        assert!(CUMTRAPZ_DESCRIPTOR
            .errors
            .iter()
            .any(|error| error.code == CUMTRAPZ_ERROR_INVALID_INPUT.code));
        assert!(CUMTRAPZ_DESCRIPTOR
            .errors
            .iter()
            .any(|error| error.code == CUMTRAPZ_ERROR_INTERNAL.code));
    }

    #[test]
    fn cumtrapz_invalid_dim_uses_descriptor_identifier() {
        let y = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let err = run_cumtrapz(Value::Tensor(y), vec![Value::Int(IntValue::I32(0))])
            .expect_err("cumtrapz");
        assert_eq!(err.identifier(), CUMTRAPZ_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn cumtrapz_too_many_inputs_uses_descriptor_identifier() {
        let err = run_cumtrapz(
            Value::Num(1.0),
            vec![Value::Num(2.0), Value::Num(3.0), Value::Num(4.0)],
        )
        .expect_err("cumtrapz");
        assert_eq!(err.identifier(), CUMTRAPZ_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn cumtrapz_gpu_input_reuploads_real_result() {
        test_support::with_test_provider(|provider| {
            let y = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
            let handle = provider
                .upload(&HostTensorView {
                    data: &y.data,
                    shape: &y.shape,
                })
                .expect("upload");
            let result = run_cumtrapz(Value::GpuTensor(handle), Vec::new()).expect("cumtrapz gpu");
            let Value::GpuTensor(out) = result else {
                panic!("expected gpu result");
            };
            let gathered = test_support::gather(Value::GpuTensor(out)).expect("gather");
            assert_eq!(gathered.shape, vec![1, 3]);
            assert_eq!(gathered.data, vec![0.0, 1.5, 4.0]);
        });
    }
}
