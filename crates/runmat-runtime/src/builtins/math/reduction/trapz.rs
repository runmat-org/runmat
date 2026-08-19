//! MATLAB-compatible `trapz` builtin for discrete trapezoidal integration.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Type,
};
use runmat_macros::runtime_builtin;
use runmat_value::NumericDType;
use runmat_value::{ComplexTensor, Tensor, Value};

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::math::reduction::integration_common::{
    canonical_shape_complex, canonical_shape_tensor, default_dimension_from_shape, dim_product,
    gather_host_value, interval_width, is_dimension_candidate, is_scalar_like, pad_shape_for_dim,
    parse_optional_dim, promote_real_value_to_gpu, real_tensor_values,
    spacing_from_gpu_or_host_value, spacing_from_value, value_has_gpu_tensor,
    value_into_complex_tensor, SpacingSpec,
};
use crate::builtins::math::reduction::type_resolvers::reduce_numeric_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "trapz";

const TRAPZ_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Q",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Trapezoidal integral output.",
}];

const TRAPZ_INPUTS_Y: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sample values.",
}];

const TRAPZ_INPUTS_Y_DIM: [BuiltinParamDescriptor; 2] = [
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

const TRAPZ_INPUTS_X_Y: [BuiltinParamDescriptor; 2] = [
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

const TRAPZ_INPUTS_X_Y_DIM: [BuiltinParamDescriptor; 3] = [
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

const TRAPZ_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "Q = trapz(Y)",
        inputs: &TRAPZ_INPUTS_Y,
        outputs: &TRAPZ_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Q = trapz(Y, dim)",
        inputs: &TRAPZ_INPUTS_Y_DIM,
        outputs: &TRAPZ_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Q = trapz(X, Y)",
        inputs: &TRAPZ_INPUTS_X_Y,
        outputs: &TRAPZ_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Q = trapz(X, Y, dim)",
        inputs: &TRAPZ_INPUTS_X_Y_DIM,
        outputs: &TRAPZ_OUTPUT,
    },
];

const TRAPZ_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRAPZ.INVALID_ARGUMENT",
    identifier: Some("RunMat:trapz:InvalidArgument"),
    when: "Input argument count, dimension selector, or spacing arguments are invalid.",
    message: "trapz: invalid argument",
};

const TRAPZ_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRAPZ.INVALID_INPUT",
    identifier: Some("RunMat:trapz:InvalidInput"),
    when: "Input values cannot be converted to supported numeric integration domains.",
    message: "trapz: invalid input",
};

const TRAPZ_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRAPZ.INTERNAL",
    identifier: Some("RunMat:trapz:Internal"),
    when: "Integration execution fails during gather, allocation, or provider promotion.",
    message: "trapz: internal integration failure",
};

const TRAPZ_ERRORS: [BuiltinErrorDescriptor; 3] = [
    TRAPZ_ERROR_INVALID_ARGUMENT,
    TRAPZ_ERROR_INVALID_INPUT,
    TRAPZ_ERROR_INTERNAL,
];

pub const TRAPZ_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TRAPZ_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TRAPZ_ERRORS,
};

const TRAPZ_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "trapz-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "trapz with native typed-integer sample data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TrapzIntegerDataExtension"),
};
const TRAPZ_INTEGER_SPACING_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "trapz-integer-spacing",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "trapz with native typed-integer point spacing is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TrapzIntegerSpacingExtension"),
};
pub const TRAPZ_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    TRAPZ_INTEGER_DATA_EXTENSION,
    TRAPZ_INTEGER_SPACING_EXTENSION,
];
const TRAPZ_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Y",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented sample-data domain is single or double. RunMat mode admits exact typed integers at a checked binary64 integration boundary.",
    }];
const TRAPZ_INTEGER_SPACING_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented spacing domain is single or double. RunMat mode admits exact typed integer scalar or vector spacing at a checked binary64 boundary.",
    }];
const TRAPZ_INTEGER_DIM_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "dim",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented positive integer scalar dimension is decoded exactly from native integer storage and must fit the runtime index range.",
    }];
pub const TRAPZ_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Q = trapz(integer_Y)",
        inputs: &TRAPZ_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Compatibility admission and exactness validation precede provider dispatch. The trapezoidal sum and result use binary64.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Q = trapz(integer_X, Y)",
        inputs: &TRAPZ_INTEGER_SPACING_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer coordinates remain exact until checked binary64 conversion; result shape and floating class follow the sampled-data path.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Q = trapz(..., integer_dim)",
        inputs: &TRAPZ_INTEGER_DIM_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::FunctionSpecific,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The dimension selects a reduction axis without entering floating arithmetic.",
    },
];

fn trapz_type(args: &[Type], ctx: &ResolveContext) -> Type {
    reduce_numeric_type(args, ctx)
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::reduction::trapz")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "trapz",
    op_kind: GpuOpKind::Custom("trapezoidal-integral"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("trapz_dim")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Real, logical, and complex-interleaved GPU sample inputs route through provider `trapz_dim` for unit, scalar, vector, and tensor real spacing; provider-missing cases fall back to host semantics.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::reduction::trapz")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "trapz",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Discrete integration currently lowers to the runtime implementation rather than fusion kernels.",
};

fn trapz_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    trapz_error_with_message(error.message, error)
}

fn trapz_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    trapz_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn trapz_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn trapz_internal_error(detail: impl AsRef<str>) -> RuntimeError {
    trapz_error_with_detail(&TRAPZ_ERROR_INTERNAL, detail)
}

#[runtime_builtin(
    name = "trapz",
    category = "math/reduction",
    summary = "Approximate integrals of sampled data using the trapezoidal rule.",
    keywords = "trapz,trapezoidal integration,numerical integration,gpu",
    accel = "none",
    type_resolver(trapz_type),
    descriptor(crate::builtins::math::reduction::trapz::TRAPZ_DESCRIPTOR),
    extensions(crate::builtins::math::reduction::trapz::TRAPZ_EXTENSIONS),
    integer_capabilities(crate::builtins::math::reduction::trapz::TRAPZ_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::reduction::trapz"
)]
async fn trapz_builtin(first: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let parsed = parse_arguments(first, rest)?;
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        &parsed.y,
        &TRAPZ_INTEGER_DATA_EXTENSION,
        NAME,
        "sample data",
    )
    .await?;
    if let Some(spacing) = &parsed.spacing {
        crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
            spacing,
            &TRAPZ_INTEGER_SPACING_EXTENSION,
            NAME,
            "point spacing",
        )
        .await?;
    }
    if crate::builtins::common::validation::is_typed_complex_integer(&parsed.y) {
        return Err(trapz_error_with_detail(
            &TRAPZ_ERROR_INVALID_INPUT,
            "operations involving complex numbers with integer types are not supported",
        ));
    }
    if let Value::GpuTensor(handle) = &parsed.y {
        if let Some(provider) = runmat_accelerate_api::provider() {
            let shape = if handle.shape.is_empty() {
                vec![1, 1]
            } else {
                handle.shape.clone()
            };
            let dim = parsed
                .dim
                .unwrap_or_else(|| default_dimension_from_shape(&shape));
            let spacing = spacing_from_gpu_or_host_value(NAME, parsed.spacing.clone(), &shape, dim)
                .map_err(|err| {
                    trapz_error_with_detail(&TRAPZ_ERROR_INVALID_ARGUMENT, err.message())
                })?;
            if let Ok(result) =
                provider.trapz_dim(handle, dim.saturating_sub(1), spacing.as_provider_spacing())
            {
                return Ok(Value::GpuTensor(result));
            }
        }
    }
    let wants_gpu_result = value_has_gpu_tensor(&parsed.y)
        || parsed
            .spacing
            .as_ref()
            .map(value_has_gpu_tensor)
            .unwrap_or(false);

    let y_value = gather_host_value(parsed.y)
        .await
        .map_err(|err| trapz_internal_error(err.message()))?;
    let spacing_value = match parsed.spacing {
        Some(value) => Some(
            gather_host_value(value)
                .await
                .map_err(|err| trapz_internal_error(err.message()))?,
        ),
        None => None,
    };
    let result = match y_value {
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            let tensor = value_into_complex_tensor(NAME, y_value).map_err(|err| {
                trapz_error_with_detail(&TRAPZ_ERROR_INVALID_INPUT, err.message())
            })?;
            let shape = canonical_shape_complex(&tensor);
            let dim = parsed
                .dim
                .unwrap_or_else(|| default_dimension_from_shape(&shape));
            let spacing =
                spacing_from_value(NAME, spacing_value.clone(), &shape, dim).map_err(|err| {
                    trapz_error_with_detail(&TRAPZ_ERROR_INVALID_ARGUMENT, err.message())
                })?;
            complex_tensor_into_value(trapz_complex_tensor(&tensor, &spacing, dim)?)
        }
        other => {
            let tensor = crate::builtins::common::tensor::value_into_tensor_for(NAME, other)
                .map_err(|err| trapz_error_with_detail(&TRAPZ_ERROR_INVALID_INPUT, err))?;
            let shape = canonical_shape_tensor(&tensor);
            let dim = parsed
                .dim
                .unwrap_or_else(|| default_dimension_from_shape(&shape));
            let spacing = spacing_from_value(NAME, spacing_value, &shape, dim).map_err(|err| {
                trapz_error_with_detail(&TRAPZ_ERROR_INVALID_ARGUMENT, err.message())
            })?;
            crate::builtins::common::tensor::tensor_into_value(trapz_tensor(
                &tensor, &spacing, dim,
            )?)
        }
    };

    if wants_gpu_result && !matches!(result, Value::Complex(_, _) | Value::ComplexTensor(_)) {
        promote_real_value_to_gpu(NAME, result).map_err(|err| trapz_internal_error(err.message()))
    } else {
        Ok(result)
    }
}

struct ParsedTrapzArgs {
    spacing: Option<Value>,
    y: Value,
    dim: Option<usize>,
}

fn parse_arguments(first: Value, rest: Vec<Value>) -> BuiltinResult<ParsedTrapzArgs> {
    match rest.len() {
        0 => Ok(ParsedTrapzArgs {
            spacing: None,
            y: first,
            dim: None,
        }),
        1 => {
            let second = rest.into_iter().next().expect("one arg");
            if is_dimension_candidate(&second) && !is_scalar_like(&first) {
                Ok(ParsedTrapzArgs {
                    spacing: None,
                    y: first,
                    dim: parse_optional_dim(NAME, &second).map_err(|err| {
                        trapz_error_with_detail(&TRAPZ_ERROR_INVALID_ARGUMENT, err.message())
                    })?,
                })
            } else {
                Ok(ParsedTrapzArgs {
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
            Ok(ParsedTrapzArgs {
                spacing: Some(first),
                y,
                dim: parse_optional_dim(NAME, &dim_arg).map_err(|err| {
                    trapz_error_with_detail(&TRAPZ_ERROR_INVALID_ARGUMENT, err.message())
                })?,
            })
        }
        _ => Err(trapz_error(&TRAPZ_ERROR_INVALID_ARGUMENT)),
    }
}

pub(crate) fn trapz_tensor(
    tensor: &Tensor,
    spacing: &SpacingSpec,
    dim: usize,
) -> BuiltinResult<Tensor> {
    if dim == 0 {
        return Err(trapz_error_with_detail(
            &TRAPZ_ERROR_INVALID_ARGUMENT,
            "dimension must be >= 1",
        ));
    }

    let shape = pad_shape_for_dim(&canonical_shape_tensor(tensor), dim);
    let dim_index = dim - 1;
    let len_dim = shape[dim_index];
    let stride_before = dim_product(&shape[..dim_index]);
    let stride_after = dim_product(&shape[dim..]);
    let block = stride_before * len_dim;
    let mut output = vec![0.0f64; stride_before * stride_after];
    let values = real_tensor_values(tensor);

    if len_dim > 1 {
        for after in 0..stride_after {
            let base = after * block;
            for before in 0..stride_before {
                let mut acc = 0.0f64;
                for k in 0..(len_dim - 1) {
                    let idx0 = base + before + k * stride_before;
                    let idx1 = idx0 + stride_before;
                    let width = interval_width(spacing, idx0, idx1, k);
                    acc += 0.5 * width * (values[idx0] + values[idx1]);
                }
                output[after * stride_before + before] = acc;
            }
        }
    }

    let mut out_shape = shape;
    out_shape[dim_index] = 1;
    if tensor.numeric_dtype() == NumericDType::F32 {
        Tensor::from_f32(
            output.into_iter().map(|value| value as f32).collect(),
            out_shape,
        )
        .map_err(|err| trapz_internal_error(&err))
    } else {
        Tensor::new(output, out_shape).map_err(|err| trapz_internal_error(&err))
    }
}

fn trapz_complex_tensor(
    tensor: &ComplexTensor,
    spacing: &SpacingSpec,
    dim: usize,
) -> BuiltinResult<ComplexTensor> {
    if dim == 0 {
        return Err(trapz_error_with_detail(
            &TRAPZ_ERROR_INVALID_ARGUMENT,
            "dimension must be >= 1",
        ));
    }

    let shape = pad_shape_for_dim(&canonical_shape_complex(tensor), dim);
    let dim_index = dim - 1;
    let len_dim = shape[dim_index];
    let stride_before = dim_product(&shape[..dim_index]);
    let stride_after = dim_product(&shape[dim..]);
    let block = stride_before * len_dim;
    let mut output = vec![(0.0f64, 0.0f64); stride_before * stride_after];

    if len_dim > 1 {
        for after in 0..stride_after {
            let base = after * block;
            for before in 0..stride_before {
                let mut acc = (0.0f64, 0.0f64);
                for k in 0..(len_dim - 1) {
                    let idx0 = base + before + k * stride_before;
                    let idx1 = idx0 + stride_before;
                    let width = interval_width(spacing, idx0, idx1, k);
                    let (re0, im0) = tensor.materialize_f64()[idx0];
                    let (re1, im1) = tensor.materialize_f64()[idx1];
                    acc.0 += 0.5 * width * (re0 + re1);
                    acc.1 += 0.5 * width * (im0 + im1);
                }
                output[after * stride_before + before] = acc;
            }
        }
    }

    let mut out_shape = shape;
    out_shape[dim_index] = 1;
    if tensor.numeric_dtype() == NumericDType::F32 {
        ComplexTensor::from_f32(
            output
                .into_iter()
                .map(|(real, imaginary)| (real as f32, imaginary as f32))
                .collect(),
            out_shape,
        )
        .map_err(|err| trapz_internal_error(&err))
    } else {
        ComplexTensor::new(output, out_shape).map_err(|err| trapz_internal_error(&err))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::AccelProvider;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::LiteralValue;
    use runmat_value::{IntValue, IntegerStorage};

    fn run_trapz(first: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::trapz_builtin(first, rest))
    }

    #[test]
    fn trapz_type_reduces_default_dimension() {
        let out = trapz_type(
            &[Type::Tensor {
                shape: Some(vec![Some(3), Some(4)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(1), Some(4)])
            }
        );
    }

    #[test]
    fn trapz_scalar_is_zero() {
        let value = run_trapz(Value::Num(5.0), Vec::new()).expect("trapz");
        assert_eq!(value, Value::Num(0.0));
    }

    #[test]
    fn trapz_row_vector_unit_spacing() {
        let y = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let value = run_trapz(Value::Tensor(y), Vec::new()).expect("trapz");
        assert_eq!(value, Value::Num(4.0));
    }

    #[test]
    fn trapz_nonuniform_x_vector() {
        let x = Tensor::new(vec![0.0, 1.0, 3.0], vec![1, 3]).unwrap();
        let y = Tensor::new(vec![0.0, 1.0, 2.0], vec![1, 3]).unwrap();
        let value = run_trapz(Value::Tensor(x), vec![Value::Tensor(y)]).expect("trapz");
        assert_eq!(value, Value::Num(3.5));
    }

    #[test]
    fn trapz_reads_typed_integer_values_and_spacing_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let x = Tensor::new_integer(IntegerStorage::U16(vec![0, 1, 3]), vec![1, 3]).expect("x");
        let y = Tensor::new_integer(IntegerStorage::I16(vec![0, 1, 2]), vec![1, 3]).expect("y");

        let value = run_trapz(Value::Tensor(x), vec![Value::Tensor(y)]).expect("trapz");

        assert_eq!(value, Value::Num(3.5));
    }

    #[test]
    fn trapz_preserves_documented_single_output_class() {
        let y = Tensor::from_f32(vec![1.0, 2.0, 3.0], vec![1, 3]).expect("single input");
        let value = run_trapz(Value::Tensor(y), Vec::new()).expect("trapz");
        let Value::Tensor(output) = value else {
            panic!("expected tensor output");
        };
        assert_eq!(
            output.into_numeric_storage().expect("single output"),
            runmat_value::NumericStorage::F32(vec![4.0])
        );
    }

    #[test]
    fn trapz_integer_data_and_spacing_are_independently_compatibility_gated() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let y = Tensor::new_integer(IntegerStorage::I16(vec![1, 2, 3]), vec![1, 3])
            .expect("integer samples");
        let data_error =
            run_trapz(Value::Tensor(y), Vec::new()).expect_err("typed sample data is an extension");
        assert_eq!(
            data_error.identifier(),
            TRAPZ_INTEGER_DATA_EXTENSION.error_identifier
        );

        let x = Tensor::new_integer(IntegerStorage::U16(vec![0, 1, 3]), vec![1, 3])
            .expect("integer spacing");
        let y = Tensor::new(vec![0.0, 1.0, 2.0], vec![1, 3]).expect("double samples");
        let spacing_error = run_trapz(Value::Tensor(x), vec![Value::Tensor(y)])
            .expect_err("typed spacing is an extension");
        assert_eq!(
            spacing_error.identifier(),
            TRAPZ_INTEGER_SPACING_EXTENSION.error_identifier
        );
    }

    #[test]
    fn trapz_rejects_inexact_wide_integer_before_floating_integration() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let y = Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_993, 9_007_199_254_740_994]),
            vec![1, 2],
        )
        .expect("wide samples");
        let error = run_trapz(Value::Tensor(y), Vec::new())
            .expect_err("inexact binary64 conversion must reject");
        assert!(error
            .message
            .contains("must be exactly representable as double"));
        assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
    }

    #[test]
    fn trapz_matrix_dimension_two() {
        let y = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let value = run_trapz(Value::Tensor(y), vec![Value::Int(IntValue::I32(2))]).expect("trapz");
        let Value::Tensor(out) = value else {
            panic!("expected tensor result");
        };
        assert_eq!(out.shape, vec![2, 1]);
        assert_eq!(out.materialize_f64(), vec![4.0, 10.0]);
    }

    #[test]
    fn trapz_complex_values() {
        let y = ComplexTensor::new(vec![(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)], vec![1, 3]).unwrap();
        let value = run_trapz(Value::ComplexTensor(y), Vec::new()).expect("trapz");
        assert_eq!(value, Value::Complex(4.0, 4.0));
    }

    #[test]
    fn trapz_type_with_explicit_dim_keeps_rank() {
        let ctx = ResolveContext::new(vec![LiteralValue::Unknown, LiteralValue::Number(2.0)]);
        let out = trapz_type(
            &[
                Type::Tensor {
                    shape: Some(vec![Some(2), Some(3)]),
                },
                Type::Int,
            ],
            &ctx,
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(1)])
            }
        );
    }

    #[test]
    fn trapz_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = TRAPZ_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Q = trapz(Y)"));
        assert!(labels.contains(&"Q = trapz(Y, dim)"));
        assert!(labels.contains(&"Q = trapz(X, Y)"));
        assert!(labels.contains(&"Q = trapz(X, Y, dim)"));
    }

    #[test]
    fn trapz_descriptor_errors_have_stable_codes() {
        assert!(TRAPZ_DESCRIPTOR
            .errors
            .iter()
            .any(|error| error.code == TRAPZ_ERROR_INVALID_ARGUMENT.code));
        assert!(TRAPZ_DESCRIPTOR
            .errors
            .iter()
            .any(|error| error.code == TRAPZ_ERROR_INVALID_INPUT.code));
        assert!(TRAPZ_DESCRIPTOR
            .errors
            .iter()
            .any(|error| error.code == TRAPZ_ERROR_INTERNAL.code));
    }

    #[test]
    fn trapz_invalid_dim_uses_descriptor_identifier() {
        let y = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let err =
            run_trapz(Value::Tensor(y), vec![Value::Int(IntValue::I32(0))]).expect_err("trapz");
        assert_eq!(err.identifier(), TRAPZ_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn trapz_too_many_inputs_uses_descriptor_identifier() {
        let err = run_trapz(
            Value::Num(1.0),
            vec![Value::Num(2.0), Value::Num(3.0), Value::Num(4.0)],
        )
        .expect_err("trapz");
        assert_eq!(err.identifier(), TRAPZ_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn trapz_gpu_input_preserves_real_result_residency() {
        test_support::with_test_provider(|provider| {
            let y = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
            let handle = provider
                .upload(&HostTensorView {
                    data: &y.materialize_f64(),
                    shape: &y.shape,
                })
                .expect("upload");
            let result = run_trapz(Value::GpuTensor(handle), Vec::new()).expect("trapz gpu");
            let Value::GpuTensor(out) = result else {
                panic!("expected gpu result");
            };
            let gathered = test_support::gather(Value::GpuTensor(out)).expect("gather");
            assert_eq!(gathered.shape, vec![1, 1]);
            assert_eq!(gathered.materialize_f64(), vec![4.0]);
        });
    }

    #[test]
    fn trapz_complex_gpu_input_preserves_result_residency() {
        test_support::with_test_provider(|provider| {
            let y =
                ComplexTensor::new(vec![(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)], vec![1, 3]).unwrap();
            let handle = crate::builtins::common::gpu_helpers::upload_complex_tensor(provider, &y)
                .expect("upload complex");
            provider.reset_telemetry();

            let result =
                run_trapz(Value::GpuTensor(handle.clone()), Vec::new()).expect("trapz complex gpu");
            let Value::GpuTensor(out) = result else {
                panic!("expected gpu result");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&out),
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            );
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

            let gathered = block_on(provider.download(&out)).expect("download");
            assert_eq!(
                gathered.storage,
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            );
            assert_eq!(gathered.shape, vec![1, 1]);
            assert_eq!(gathered.data, vec![4.0, 4.0]);
            let _ = provider.free(&handle);
            let _ = provider.free(&out);
        });
    }

    #[test]
    fn trapz_gpu_input_uses_vector_spacing_on_provider() {
        test_support::with_test_provider(|provider| {
            let x = Tensor::new(vec![0.0, 1.0, 3.0], vec![1, 3]).unwrap();
            let y = Tensor::new(vec![0.0, 1.0, 2.0], vec![1, 3]).unwrap();
            let handle = provider
                .upload(&HostTensorView {
                    data: &y.materialize_f64(),
                    shape: &y.shape,
                })
                .expect("upload y");
            let result =
                run_trapz(Value::Tensor(x), vec![Value::GpuTensor(handle)]).expect("trapz gpu");
            let Value::GpuTensor(out) = result else {
                panic!("expected gpu result");
            };
            let gathered = test_support::gather(Value::GpuTensor(out)).expect("gather");
            assert_eq!(gathered.shape, vec![1, 1]);
            assert_eq!(gathered.materialize_f64(), vec![3.5]);
        });
    }

    #[test]
    fn trapz_gpu_input_uses_tensor_spacing_on_provider() {
        test_support::with_test_provider(|provider| {
            let x = Tensor::new(vec![0.0, 0.0, 1.0, 1.0, 3.0, 3.0], vec![2, 3]).unwrap();
            let y = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
            let y_handle = provider
                .upload(&HostTensorView {
                    data: &y.materialize_f64(),
                    shape: &y.shape,
                })
                .expect("upload y");
            let x_handle = provider
                .upload(&HostTensorView {
                    data: &x.materialize_f64(),
                    shape: &x.shape,
                })
                .expect("upload x");
            let result = run_trapz(
                Value::GpuTensor(x_handle),
                vec![Value::GpuTensor(y_handle), Value::Int(IntValue::I32(2))],
            )
            .expect("trapz gpu");
            let Value::GpuTensor(out) = result else {
                panic!("expected gpu result");
            };
            let gathered = test_support::gather(Value::GpuTensor(out)).expect("gather");
            assert_eq!(gathered.shape, vec![2, 1]);
            assert_eq!(gathered.materialize_f64(), vec![6.5, 15.5]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn trapz_wgpu_matches_cpu_vector_spacing() {
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        runmat_accelerate_api::set_thread_provider(Some(provider));
        let x = Tensor::new(vec![0.0, 1.0, 3.0], vec![1, 3]).unwrap();
        let y = Tensor::new(vec![0.0, 1.0, 2.0], vec![1, 3]).unwrap();
        let cpu = run_trapz(Value::Tensor(x.clone()), vec![Value::Tensor(y.clone())]).unwrap();
        let y_handle = provider
            .upload(&HostTensorView {
                data: &y.materialize_f64(),
                shape: &y.shape,
            })
            .unwrap();
        let gpu = run_trapz(Value::Tensor(x), vec![Value::GpuTensor(y_handle)]).unwrap();
        let gathered = test_support::gather(gpu).expect("gather gpu");
        let expected = match cpu {
            Value::Num(value) => value,
            other => panic!("unexpected cpu result {other:?}"),
        };
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-9,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        assert_eq!(gathered.shape, vec![1, 1]);
        assert!((gathered.materialize_f64()[0] - expected).abs() < tol);
    }
}
