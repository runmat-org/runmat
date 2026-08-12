//! MATLAB-compatible `cumtrapz` builtin for cumulative trapezoidal integration.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, NumericDType, ResolveContext, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::math::reduction::integration_common::{
    canonical_shape_complex, canonical_shape_tensor, default_dimension_from_shape, dim_product,
    gather_host_value, interval_width, is_dimension_candidate, is_scalar_like, pad_shape_for_dim,
    parse_optional_dim, real_tensor_values, spacing_from_gpu_or_host_value_for_provider,
    spacing_from_value, value_has_gpu_tensor, value_into_complex_tensor, SpacingSpec,
};
use crate::builtins::math::reduction::type_resolvers::cumulative_numeric_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "cumtrapz";

pub const CUMTRAPZ_INTEGER_Y_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cumtrapz-integer-y",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cumtrapz with typed-integer sample data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CumtrapzIntegerYExtension"),
};
pub const CUMTRAPZ_LOGICAL_Y_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cumtrapz-logical-y",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cumtrapz with logical sample data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CumtrapzLogicalYExtension"),
};
pub const CUMTRAPZ_INTEGER_X_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cumtrapz-integer-spacing",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cumtrapz with typed-integer spacing is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CumtrapzIntegerSpacingExtension"),
};
pub const CUMTRAPZ_LOGICAL_X_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cumtrapz-logical-spacing",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cumtrapz with logical spacing is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CumtrapzLogicalSpacingExtension"),
};
pub const CUMTRAPZ_INTEGER_DIM_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cumtrapz-integer-dim",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cumtrapz with a typed-integer dimension is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CumtrapzIntegerDimExtension"),
};
pub const CUMTRAPZ_TENSOR_SPACING_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "cumtrapz-tensor-spacing",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "cumtrapz with same-size tensor spacing is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CumtrapzTensorSpacingExtension"),
    };
pub const CUMTRAPZ_EXTENSIONS: [BuiltinExtensionDescriptor; 6] = [
    CUMTRAPZ_INTEGER_Y_EXTENSION,
    CUMTRAPZ_LOGICAL_Y_EXTENSION,
    CUMTRAPZ_INTEGER_X_EXTENSION,
    CUMTRAPZ_LOGICAL_X_EXTENSION,
    CUMTRAPZ_INTEGER_DIM_EXTENSION,
    CUMTRAPZ_TENSOR_SPACING_EXTENSION,
];

const INTEGER_Y_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "Y",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All eight real integer classes cross an explicitly gated binary64 integration boundary after exact-representability validation.",
}];
const INTEGER_X_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All eight real integer spacing classes are independently gated and validated before binary64 coordinate differences are formed.",
}];
const INTEGER_DIM_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "dim",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "The public contract accepts a positive integer-valued scalar but does not list typed integer classes; ordinary double dim remains documented.",
}];
pub const CUMTRAPZ_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Q = cumtrapz(integer_Y, dim?)",
        inputs: &INTEGER_Y_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer samples remain authoritative through validation, then produce shape-preserving double output; eligible resident fallback returns through the first resident owner.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Q = cumtrapz(integer_X, Y, dim?)",
        inputs: &INTEGER_X_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer coordinates are independently gated and cross the floating spacing boundary; Y controls shape and documented floating precision.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Q = cumtrapz(___, integer_dim)",
        inputs: &INTEGER_DIM_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed dim is read exactly and controls traversal only; it does not determine the numeric result class.",
    },
];

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
    provider_hooks: &[ProviderHook::Custom("cumtrapz_dim")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Floating real and complex-interleaved GPU sample inputs can route through the owning provider's `cumtrapz_dim` hook for eligible real spacing; integer and logical resident inputs, ineligible spacing, and unavailable or rejected provider hooks use validated host fallback.",
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

fn enable_extension(extension: &BuiltinExtensionDescriptor) -> BuiltinResult<()> {
    crate::compatibility::ensure_builtin_extension_enabled(extension, NAME)
}

fn is_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn is_logical_value(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
}

fn value_shape(value: &Value) -> Option<Vec<usize>> {
    match value {
        Value::Tensor(tensor) => Some(canonical_shape_tensor(tensor)),
        Value::LogicalArray(array) => Some(if array.shape.is_empty() {
            vec![array.data.len(), 1]
        } else {
            array.shape.clone()
        }),
        Value::GpuTensor(handle) => Some(if handle.shape.is_empty() {
            vec![1, 1]
        } else {
            handle.shape.clone()
        }),
        _ => None,
    }
}

fn same_size_tensor_spacing(parsed: &ParsedCumtrapzArgs) -> bool {
    let (Some(spacing), Some(mut x_shape), Some(mut y_shape)) = (
        parsed.spacing.as_ref(),
        parsed.spacing.as_ref().and_then(value_shape),
        value_shape(&parsed.y),
    ) else {
        return false;
    };
    if matches!(spacing, Value::Tensor(value_tensor) if crate::builtins::common::tensor::is_scalar_tensor(value_tensor))
        || matches!(spacing, Value::LogicalArray(array) if array.data.len() == 1)
        || matches!(spacing, Value::GpuTensor(handle) if crate::builtins::common::tensor::element_count(&handle.shape) == 1)
    {
        return false;
    }
    if x_shape.iter().filter(|extent| **extent > 1).count() <= 1 {
        return false;
    }
    while x_shape.last() == Some(&1) {
        x_shape.pop();
    }
    while y_shape.last() == Some(&1) {
        y_shape.pop();
    }
    x_shape == y_shape
}

fn ensure_cumtrapz_extensions(parsed: &ParsedCumtrapzArgs) -> BuiltinResult<()> {
    if is_integer_value(&parsed.y) {
        enable_extension(&CUMTRAPZ_INTEGER_Y_EXTENSION)?;
    }
    if is_logical_value(&parsed.y) {
        enable_extension(&CUMTRAPZ_LOGICAL_Y_EXTENSION)?;
    }
    if let Some(spacing) = parsed.spacing.as_ref() {
        if is_integer_value(spacing) {
            enable_extension(&CUMTRAPZ_INTEGER_X_EXTENSION)?;
        }
        if is_logical_value(spacing) {
            enable_extension(&CUMTRAPZ_LOGICAL_X_EXTENSION)?;
        }
    }
    if parsed.dim_value.as_ref().is_some_and(is_integer_value) {
        enable_extension(&CUMTRAPZ_INTEGER_DIM_EXTENSION)?;
    }
    if same_size_tensor_spacing(parsed) {
        enable_extension(&CUMTRAPZ_TENSOR_SPACING_EXTENSION)?;
    }
    Ok(())
}

fn integer_is_exact_f64(value: &runmat_builtins::IntValue) -> bool {
    let magnitude = match value {
        runmat_builtins::IntValue::I8(value) => u64::from(value.unsigned_abs()),
        runmat_builtins::IntValue::I16(value) => u64::from(value.unsigned_abs()),
        runmat_builtins::IntValue::I32(value) => u64::from(value.unsigned_abs()),
        runmat_builtins::IntValue::I64(value) => value.unsigned_abs(),
        runmat_builtins::IntValue::U8(value) => u64::from(*value),
        runmat_builtins::IntValue::U16(value) => u64::from(*value),
        runmat_builtins::IntValue::U32(value) => u64::from(*value),
        runmat_builtins::IntValue::U64(value) => *value,
    };
    if magnitude == 0 {
        return true;
    }
    let significant_bits = u64::BITS - magnitude.leading_zeros();
    significant_bits <= f64::MANTISSA_DIGITS
        || magnitude.trailing_zeros() >= significant_bits - f64::MANTISSA_DIGITS
}

fn ensure_exact_integer_boundary(value: &Value, role: &str) -> BuiltinResult<()> {
    let exact = match value {
        Value::Int(value) => integer_is_exact_f64(value),
        Value::Tensor(tensor) => tensor
            .integer_storage()
            .is_none_or(|storage| storage.exact_values().iter().all(integer_is_exact_f64)),
        _ => true,
    };
    if exact {
        Ok(())
    } else {
        Err(cumtrapz_error_with_detail(
            &CUMTRAPZ_ERROR_INVALID_INPUT,
            format!("{role} contains an integer that is not exactly representable as double"),
        ))
    }
}

fn value_uses_single(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
        || matches!(value, Value::ComplexTensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_precision(handle) == Some(runmat_accelerate_api::ProviderPrecision::F32))
}

fn first_resident_handle(
    parsed: &ParsedCumtrapzArgs,
) -> Option<runmat_accelerate_api::GpuTensorHandle> {
    match (&parsed.y, parsed.spacing.as_ref()) {
        (Value::GpuTensor(handle), _) => Some(handle.clone()),
        (_, Some(Value::GpuTensor(handle))) => Some(handle.clone()),
        _ => None,
    }
}

fn native_y_is_eligible(handle: &runmat_accelerate_api::GpuTensorHandle) -> bool {
    runmat_accelerate_api::handle_integer_type(handle).is_none()
        && !runmat_accelerate_api::handle_is_logical(handle)
        && runmat_accelerate_api::handle_precision(handle).is_some()
        && matches!(
            runmat_accelerate_api::handle_storage(handle),
            runmat_accelerate_api::GpuTensorStorage::Real
                | runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        )
}

fn native_spacing_is_eligible(
    spacing: &Option<Value>,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
) -> bool {
    let Some(spacing) = spacing else {
        return true;
    };
    if is_integer_value(spacing) || is_logical_value(spacing) {
        return false;
    }
    match spacing {
        Value::GpuTensor(handle) => {
            runmat_accelerate_api::handle_storage(handle)
                == runmat_accelerate_api::GpuTensorStorage::Real
                && runmat_accelerate_api::provider_for_handle(handle)
                    .is_some_and(|owner| std::ptr::eq(owner, provider))
        }
        _ => true,
    }
}

fn native_result_is_valid(
    result: &runmat_accelerate_api::GpuTensorHandle,
    input: &runmat_accelerate_api::GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    expected_shape: &[usize],
) -> bool {
    result.device_id == input.device_id
        && result.shape == expected_shape
        && runmat_accelerate_api::handle_storage(result)
            == runmat_accelerate_api::handle_storage(input)
        && runmat_accelerate_api::handle_precision(result)
            == runmat_accelerate_api::handle_precision(input)
        && runmat_accelerate_api::provider_for_handle(result)
            .is_some_and(|owner| std::ptr::eq(owner, provider))
}

fn free_rejected_native_result(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    invoked_provider: &'static dyn runmat_accelerate_api::AccelProvider,
) {
    let owner = runmat_accelerate_api::provider_for_handle(handle).unwrap_or(invoked_provider);
    if let Err(error) = owner.free(handle) {
        log::trace!("cumtrapz: failed to free rejected provider result: {error}");
    }
}

fn restore_to_owner(
    value: Value,
    owner: Option<&runmat_accelerate_api::GpuTensorHandle>,
    single_output: bool,
) -> BuiltinResult<Value> {
    let Some(provider) = owner.and_then(runmat_accelerate_api::provider_for_handle) else {
        return Ok(value);
    };
    let expected = if single_output {
        runmat_accelerate_api::ProviderPrecision::F32
    } else {
        runmat_accelerate_api::ProviderPrecision::F64
    };
    if provider.precision() != expected {
        return Ok(value);
    }
    match value {
        Value::Num(number) => {
            let tensor = if single_output {
                Tensor::from_f32(vec![number as f32], vec![1, 1])
            } else {
                Tensor::new(vec![number], vec![1, 1])
            }
            .map_err(|error| cumtrapz_internal_error(&error))?;
            crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
                .map(crate::builtins::common::gpu_helpers::resident_gpu_value)
                .map_err(|error| cumtrapz_internal_error(&error))
        }
        Value::Complex(real, imaginary) => {
            let tensor = if single_output {
                ComplexTensor::from_f32(vec![(real as f32, imaginary as f32)], vec![1, 1])
            } else {
                ComplexTensor::new(vec![(real, imaginary)], vec![1, 1])
            }
            .map_err(|error| cumtrapz_internal_error(&error))?;
            crate::builtins::common::gpu_helpers::upload_complex_tensor(provider, &tensor)
                .map(crate::builtins::common::gpu_helpers::complex_gpu_value)
                .map_err(|error| cumtrapz_internal_error(error.to_string()))
        }
        Value::Tensor(tensor) => {
            crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
                .map(crate::builtins::common::gpu_helpers::resident_gpu_value)
                .map_err(|error| cumtrapz_internal_error(&error))
        }
        Value::ComplexTensor(tensor) => {
            crate::builtins::common::gpu_helpers::upload_complex_tensor(provider, &tensor)
                .map(crate::builtins::common::gpu_helpers::complex_gpu_value)
                .map_err(|error| cumtrapz_internal_error(error.to_string()))
        }
        other => Ok(other),
    }
}

#[runtime_builtin(
    name = "cumtrapz",
    category = "math/reduction",
    summary = "Compute cumulative trapezoidal integration.",
    keywords = "cumtrapz,cumulative trapezoidal integration,numerical integration,gpu",
    accel = "none",
    type_resolver(cumtrapz_type),
    descriptor(crate::builtins::math::reduction::cumtrapz::CUMTRAPZ_DESCRIPTOR),
    extensions(crate::builtins::math::reduction::cumtrapz::CUMTRAPZ_EXTENSIONS),
    integer_capabilities(
        crate::builtins::math::reduction::cumtrapz::CUMTRAPZ_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::math::reduction::cumtrapz"
)]
async fn cumtrapz_builtin(first: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.len() == 2 && is_integer_value(&rest[1]) {
        enable_extension(&CUMTRAPZ_INTEGER_DIM_EXTENSION)?;
    }
    let parsed = parse_arguments(first, rest)?;
    ensure_cumtrapz_extensions(&parsed)?;
    if crate::builtins::common::validation::is_typed_complex_integer(&parsed.y) {
        return Err(cumtrapz_error_with_detail(
            &CUMTRAPZ_ERROR_INVALID_INPUT,
            "operations involving complex numbers with integer types are not supported",
        ));
    }
    let single_output = value_uses_single(&parsed.y);
    let owner_handle = first_resident_handle(&parsed);
    if let Value::GpuTensor(handle) = &parsed.y {
        if let Some(provider) = runmat_accelerate_api::provider_for_handle(handle) {
            let shape = if handle.shape.is_empty() {
                vec![1, 1]
            } else {
                handle.shape.clone()
            };
            let dim = parsed
                .dim
                .unwrap_or_else(|| default_dimension_from_shape(&shape));
            if native_y_is_eligible(handle) && native_spacing_is_eligible(&parsed.spacing, provider)
            {
                if let Ok(spacing) = spacing_from_gpu_or_host_value_for_provider(
                    NAME,
                    parsed.spacing.clone(),
                    &shape,
                    dim,
                    provider,
                ) {
                    let native_result = provider.cumtrapz_dim(
                        handle,
                        dim.saturating_sub(1),
                        spacing.as_provider_spacing(),
                    );
                    if let Err(error) = spacing.free_owned(provider) {
                        log::trace!("cumtrapz: failed to free temporary spacing upload: {error}");
                    }
                    match native_result {
                        Ok(result) if native_result_is_valid(&result, handle, provider, &shape) => {
                            return Ok(Value::GpuTensor(result));
                        }
                        Ok(result) => free_rejected_native_result(&result, provider),
                        Err(error) => log::trace!(
                            "cumtrapz: provider hook unavailable, using host fallback: {error}"
                        ),
                    }
                }
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
        .map_err(|err| cumtrapz_internal_error(err.message()))?;
    let spacing_value = match parsed.spacing {
        Some(value) => Some(
            gather_host_value(value)
                .await
                .map_err(|err| cumtrapz_internal_error(err.message()))?,
        ),
        None => None,
    };
    ensure_exact_integer_boundary(&y_value, "Y")?;
    if let Some(spacing) = spacing_value.as_ref() {
        ensure_exact_integer_boundary(spacing, "X")?;
    }

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
            complex_tensor_into_value(cumtrapz_complex_tensor(
                &tensor,
                &spacing,
                dim,
                single_output,
            )?)
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
                &tensor,
                &spacing,
                dim,
                single_output,
            )?)
        }
    };

    if wants_gpu_result {
        restore_to_owner(result, owner_handle.as_ref(), single_output)
    } else {
        Ok(result)
    }
}

struct ParsedCumtrapzArgs {
    spacing: Option<Value>,
    y: Value,
    dim: Option<usize>,
    dim_value: Option<Value>,
}

fn parse_arguments(first: Value, rest: Vec<Value>) -> BuiltinResult<ParsedCumtrapzArgs> {
    match rest.len() {
        0 => Ok(ParsedCumtrapzArgs {
            spacing: None,
            y: first,
            dim: None,
            dim_value: None,
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
                    dim_value: Some(second),
                })
            } else {
                Ok(ParsedCumtrapzArgs {
                    spacing: Some(first),
                    y: second,
                    dim: None,
                    dim_value: None,
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
                dim_value: Some(dim_arg),
            })
        }
        _ => Err(cumtrapz_error(&CUMTRAPZ_ERROR_INVALID_ARGUMENT)),
    }
}

fn cumtrapz_tensor(
    tensor: &Tensor,
    spacing: &SpacingSpec,
    dim: usize,
    single_output: bool,
) -> BuiltinResult<Tensor> {
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
    let values = real_tensor_values(tensor);
    let mut output = vec![0.0f64; values.len()];

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
                    acc += 0.5 * width * (values[idx0] + values[idx1]);
                    output[idx1] = acc;
                }
            }
        }
    }

    if single_output {
        Tensor::from_f32(
            output.into_iter().map(|value| value as f32).collect(),
            shape,
        )
        .map_err(|err| cumtrapz_internal_error(&err))
    } else {
        Tensor::new(output, shape).map_err(|err| cumtrapz_internal_error(&err))
    }
}

fn cumtrapz_complex_tensor(
    tensor: &ComplexTensor,
    spacing: &SpacingSpec,
    dim: usize,
    single_output: bool,
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
    let mut output = vec![(0.0f64, 0.0f64); tensor.materialize_f64().len()];

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
                    let (re0, im0) = tensor.materialize_f64()[idx0];
                    let (re1, im1) = tensor.materialize_f64()[idx1];
                    acc.0 += 0.5 * width * (re0 + re1);
                    acc.1 += 0.5 * width * (im0 + im1);
                    output[idx1] = acc;
                }
            }
        }
    }

    if single_output {
        ComplexTensor::from_f32(
            output
                .into_iter()
                .map(|(real, imaginary)| (real as f32, imaginary as f32))
                .collect(),
            shape,
        )
        .map_err(|err| cumtrapz_internal_error(&err))
    } else {
        ComplexTensor::new(output, shape).map_err(|err| cumtrapz_internal_error(&err))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::{AccelProvider as _, HostTensorView};
    use runmat_builtins::{IntValue, IntegerStorage};
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Clone, Copy)]
    enum NativeBehavior {
        Success,
        Error,
        Reject,
    }

    struct CountingCumtrapzProvider {
        inner: runmat_accelerate::simple_provider::InProcessProvider,
        frees: AtomicUsize,
        behavior: NativeBehavior,
    }

    impl runmat_accelerate_api::AccelProvider for CountingCumtrapzProvider {
        fn upload(
            &self,
            host: &runmat_accelerate_api::HostTensorView,
        ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
            runmat_accelerate_api::AccelProvider::upload(&self.inner, host)
        }

        fn download<'a>(
            &'a self,
            handle: &'a runmat_accelerate_api::GpuTensorHandle,
        ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
            runmat_accelerate_api::AccelProvider::download(&self.inner, handle)
        }

        fn free(&self, handle: &runmat_accelerate_api::GpuTensorHandle) -> anyhow::Result<()> {
            self.frees.fetch_add(1, Ordering::SeqCst);
            runmat_accelerate_api::AccelProvider::free(&self.inner, handle)
        }

        fn device_info(&self) -> String {
            runmat_accelerate_api::AccelProvider::device_info(&self.inner)
        }

        fn device_id(&self) -> u32 {
            runmat_accelerate_api::AccelProvider::device_id(&self.inner)
        }

        fn precision(&self) -> runmat_accelerate_api::ProviderPrecision {
            runmat_accelerate_api::AccelProvider::precision(&self.inner)
        }

        fn cumtrapz_dim(
            &self,
            input: &runmat_accelerate_api::GpuTensorHandle,
            dim: usize,
            spacing: runmat_accelerate_api::ProviderTrapezoidSpacing<'_>,
        ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
            match self.behavior {
                NativeBehavior::Success => runmat_accelerate_api::AccelProvider::cumtrapz_dim(
                    &self.inner,
                    input,
                    dim,
                    spacing,
                ),
                NativeBehavior::Error => anyhow::bail!("deliberate cumtrapz failure"),
                NativeBehavior::Reject => self.upload(&HostTensorView {
                    data: &[0.0],
                    shape: &[1, 1],
                }),
            }
        }
    }

    fn with_counting_provider<R>(
        behavior: NativeBehavior,
        body: impl FnOnce(&'static CountingCumtrapzProvider) -> R,
    ) -> R {
        let _guard = test_support::accel_test_lock();
        let provider = Box::leak(Box::new(CountingCumtrapzProvider {
            inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
            frees: AtomicUsize::new(0),
            behavior,
        }));
        unsafe { runmat_accelerate_api::register_provider(provider) };
        let _thread_provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        body(provider)
    }

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
        assert_eq!(out.materialize_f64(), vec![0.0, 1.5, 4.0]);
    }

    #[test]
    fn cumtrapz_nonuniform_x_vector() {
        let x = Tensor::new(vec![0.0, 1.0, 3.0], vec![1, 3]).unwrap();
        let y = Tensor::new(vec![0.0, 1.0, 2.0], vec![1, 3]).unwrap();
        let value = run_cumtrapz(Value::Tensor(x), vec![Value::Tensor(y)]).expect("cumtrapz");
        let Value::Tensor(out) = value else {
            panic!("expected tensor result");
        };
        assert_eq!(out.materialize_f64(), vec![0.0, 0.5, 3.5]);
    }

    #[test]
    fn cumtrapz_reads_typed_integer_values_and_spacing_exactly() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let x = Tensor::new_integer(IntegerStorage::U16(vec![0, 1, 3]), vec![1, 3]).expect("x");
        let y = Tensor::new_integer(IntegerStorage::I16(vec![0, 1, 2]), vec![1, 3]).expect("y");

        let value = run_cumtrapz(Value::Tensor(x), vec![Value::Tensor(y)]).expect("cumtrapz");

        let Value::Tensor(out) = value else {
            panic!("expected tensor result");
        };
        assert_eq!(out.materialize_f64(), vec![0.0, 0.5, 3.5]);
    }

    #[test]
    fn cumtrapz_matrix_dimension_two() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let y = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let value =
            run_cumtrapz(Value::Tensor(y), vec![Value::Int(IntValue::I32(2))]).expect("cumtrapz");
        let Value::Tensor(out) = value else {
            panic!("expected tensor result");
        };
        assert_eq!(out.shape, vec![2, 3]);
        assert_eq!(out.materialize_f64(), vec![0.0, 0.0, 1.5, 4.5, 4.0, 10.0]);
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
    fn cumtrapz_gpu_spec_describes_logical_fallback() {
        assert!(GPU_SPEC
            .notes
            .contains("integer and logical resident inputs"));
        assert!(GPU_SPEC.notes.contains("validated host fallback"));
        assert!(!GPU_SPEC
            .notes
            .contains("Real, logical, and complex-interleaved"));
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
    fn cumtrapz_gpu_input_preserves_real_result_residency() {
        test_support::with_test_provider(|provider| {
            let y = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
            let handle = provider
                .upload(&HostTensorView {
                    data: &y.materialize_f64(),
                    shape: &y.shape,
                })
                .expect("upload");
            let result = run_cumtrapz(Value::GpuTensor(handle), Vec::new()).expect("cumtrapz gpu");
            let Value::GpuTensor(out) = result else {
                panic!("expected gpu result");
            };
            let gathered = test_support::gather(Value::GpuTensor(out)).expect("gather");
            assert_eq!(gathered.shape, vec![1, 3]);
            assert_eq!(gathered.materialize_f64(), vec![0.0, 1.5, 4.0]);
        });
    }

    #[test]
    fn cumtrapz_complex_gpu_input_preserves_result_residency() {
        test_support::with_test_provider(|provider| {
            let y =
                ComplexTensor::new(vec![(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)], vec![1, 3]).unwrap();
            let handle = crate::builtins::common::gpu_helpers::upload_complex_tensor(provider, &y)
                .expect("upload complex");
            provider.reset_telemetry();

            let result = run_cumtrapz(Value::GpuTensor(handle.clone()), Vec::new())
                .expect("cumtrapz complex gpu");
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
            assert_eq!(gathered.shape, vec![1, 3]);
            assert_eq!(gathered.data, vec![0.0, 0.0, 1.5, 1.5, 4.0, 4.0]);
            let _ = provider.free(&handle);
            let _ = provider.free(&out);
        });
    }

    #[test]
    fn cumtrapz_gpu_input_uses_vector_spacing_on_provider() {
        test_support::with_test_provider(|provider| {
            let x = Tensor::new(vec![0.0, 1.0, 3.0], vec![1, 3]).unwrap();
            let y = Tensor::new(vec![0.0, 1.0, 2.0], vec![1, 3]).unwrap();
            let handle = provider
                .upload(&HostTensorView {
                    data: &y.materialize_f64(),
                    shape: &y.shape,
                })
                .expect("upload y");
            let result = run_cumtrapz(Value::Tensor(x), vec![Value::GpuTensor(handle)])
                .expect("cumtrapz gpu");
            let Value::GpuTensor(out) = result else {
                panic!("expected gpu result");
            };
            let gathered = test_support::gather(Value::GpuTensor(out)).expect("gather");
            assert_eq!(gathered.shape, vec![1, 3]);
            assert_eq!(gathered.materialize_f64(), vec![0.0, 0.5, 3.5]);
        });
    }

    #[test]
    fn cumtrapz_gpu_input_uses_tensor_spacing_on_provider() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
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
            let result = run_cumtrapz(
                Value::GpuTensor(x_handle),
                vec![Value::GpuTensor(y_handle), Value::Int(IntValue::I32(2))],
            )
            .expect("cumtrapz gpu");
            let Value::GpuTensor(out) = result else {
                panic!("expected gpu result");
            };
            let gathered = test_support::gather(Value::GpuTensor(out)).expect("gather");
            assert_eq!(gathered.shape, vec![2, 3]);
            assert_eq!(
                gathered.materialize_f64(),
                vec![0.0, 0.0, 1.5, 4.5, 6.5, 15.5]
            );
        });
    }

    #[test]
    fn cumtrapz_extensions_are_independently_mode_gated() {
        let integer = || {
            Value::Tensor(Tensor::new_integer(IntegerStorage::U8(vec![1, 2]), vec![1, 2]).unwrap())
        };
        let logical = || {
            Value::LogicalArray(runmat_builtins::LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap())
        };
        let y = || Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
        let tensor_x = || Value::Tensor(Tensor::new(vec![0.0, 1.0, 0.0, 1.0], vec![2, 2]).unwrap());
        let matrix_y = || Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap());
        let cases = [
            (
                integer(),
                Vec::new(),
                CUMTRAPZ_INTEGER_Y_EXTENSION.error_identifier,
            ),
            (
                logical(),
                Vec::new(),
                CUMTRAPZ_LOGICAL_Y_EXTENSION.error_identifier,
            ),
            (
                integer(),
                vec![y()],
                CUMTRAPZ_INTEGER_X_EXTENSION.error_identifier,
            ),
            (
                logical(),
                vec![y()],
                CUMTRAPZ_LOGICAL_X_EXTENSION.error_identifier,
            ),
            (
                y(),
                vec![Value::Int(IntValue::U8(2))],
                CUMTRAPZ_INTEGER_DIM_EXTENSION.error_identifier,
            ),
            (
                tensor_x(),
                vec![matrix_y()],
                CUMTRAPZ_TENSOR_SPACING_EXTENSION.error_identifier,
            ),
        ];
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        for (first, rest, identifier) in cases {
            let error = run_cumtrapz(first, rest).expect_err("strict mode rejects extension");
            assert_eq!(error.identifier(), identifier);
        }
    }

    #[test]
    fn cumtrapz_integer_capabilities_cover_all_classes_and_preserve_single() {
        assert_eq!(CUMTRAPZ_INTEGER_CAPABILITIES.len(), 3);
        assert!(CUMTRAPZ_INTEGER_CAPABILITIES
            .iter()
            .all(|capability| capability.inputs[0].classes.len() == 8));
        let y = Tensor::from_f32(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let Value::Tensor(output) = run_cumtrapz(Value::Tensor(y), Vec::new()).unwrap() else {
            panic!("expected tensor");
        };
        assert_eq!(output.numeric_dtype(), NumericDType::F32);
        let complex = ComplexTensor::from_f32(vec![(1.0, 1.0), (2.0, 2.0)], vec![1, 2]).unwrap();
        let Value::ComplexTensor(output) =
            run_cumtrapz(Value::ComplexTensor(complex), Vec::new()).unwrap()
        else {
            panic!("expected complex tensor");
        };
        assert_eq!(output.numeric_dtype(), NumericDType::F32);
    }

    #[test]
    fn cumtrapz_accepts_all_integer_storage_classes_and_rejects_inexact_boundary() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let storages = [
            IntegerStorage::I8(vec![0, 1, 2]),
            IntegerStorage::I16(vec![0, 1, 2]),
            IntegerStorage::I32(vec![0, 1, 2]),
            IntegerStorage::I64(vec![0, 1, 2]),
            IntegerStorage::U8(vec![0, 1, 2]),
            IntegerStorage::U16(vec![0, 1, 2]),
            IntegerStorage::U32(vec![0, 1, 2]),
            IntegerStorage::U64(vec![0, 1, 2]),
        ];
        for storage in storages {
            let y = Tensor::new_integer(storage, vec![1, 3]).unwrap();
            let Value::Tensor(output) = run_cumtrapz(Value::Tensor(y), Vec::new()).unwrap() else {
                panic!("expected tensor");
            };
            assert_eq!(output.materialize_f64(), vec![0.0, 0.5, 2.0]);
        }
        let inexact = Tensor::new_integer(
            IntegerStorage::U64(vec![0, 9_007_199_254_740_993]),
            vec![1, 2],
        )
        .unwrap();
        let error = run_cumtrapz(Value::Tensor(inexact), Vec::new()).expect_err("inexact boundary");
        assert_eq!(error.identifier(), CUMTRAPZ_ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn cumtrapz_resident_extension_roles_reject_before_provider_access() {
        fn resident(shape: Vec<usize>, buffer_id: u64, integer: bool, logical: bool) -> Value {
            let handle = runmat_accelerate_api::GpuTensorHandle {
                shape,
                device_id: u32::MAX,
                buffer_id,
            };
            if integer {
                runmat_accelerate_api::set_handle_integer_type(
                    &handle,
                    runmat_accelerate_api::IntegerElementType::U64,
                );
            }
            if logical {
                runmat_accelerate_api::set_handle_logical(&handle, true);
            }
            Value::GpuTensor(handle)
        }
        let y = || Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
        let cases = [
            (
                resident(vec![1, 2], u64::MAX - 520, true, false),
                Vec::new(),
                CUMTRAPZ_INTEGER_Y_EXTENSION.error_identifier,
            ),
            (
                resident(vec![1, 2], u64::MAX - 521, false, true),
                Vec::new(),
                CUMTRAPZ_LOGICAL_Y_EXTENSION.error_identifier,
            ),
            (
                resident(vec![1, 2], u64::MAX - 522, true, false),
                vec![y()],
                CUMTRAPZ_INTEGER_X_EXTENSION.error_identifier,
            ),
            (
                resident(vec![1, 2], u64::MAX - 523, false, true),
                vec![y()],
                CUMTRAPZ_LOGICAL_X_EXTENSION.error_identifier,
            ),
            (
                Value::Num(1.0),
                vec![y(), resident(vec![1, 1], u64::MAX - 524, true, false)],
                CUMTRAPZ_INTEGER_DIM_EXTENSION.error_identifier,
            ),
        ];
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        for (first, rest, identifier) in cases {
            let error = run_cumtrapz(first, rest).expect_err("resident extension gate");
            assert_eq!(error.identifier(), identifier);
        }
    }

    #[test]
    fn cumtrapz_frees_host_spacing_upload_for_every_native_outcome() {
        for (behavior, expected_frees) in [
            (NativeBehavior::Success, 1),
            (NativeBehavior::Error, 1),
            (NativeBehavior::Reject, 2),
        ] {
            with_counting_provider(behavior, |provider| {
                let y = [0.0, 1.0, 2.0];
                let y_handle = provider
                    .upload(&HostTensorView {
                        data: &y,
                        shape: &[1, 3],
                    })
                    .expect("resident Y");
                let x = Value::Tensor(Tensor::new(vec![0.0, 1.0, 3.0], vec![1, 3]).unwrap());
                let result = run_cumtrapz(x, vec![Value::GpuTensor(y_handle.clone())])
                    .expect("native or fallback cumtrapz");
                assert_eq!(provider.frees.load(Ordering::SeqCst), expected_frees);
                if let Value::GpuTensor(result_handle) = result {
                    let _ = provider.free(&result_handle);
                }
                let _ = provider.free(&y_handle);
            });
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn cumtrapz_wgpu_matches_cpu_vector_spacing() {
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        runmat_accelerate_api::set_thread_provider(Some(provider));
        let x = Tensor::new(vec![0.0, 1.0, 3.0], vec![1, 3]).unwrap();
        let y = Tensor::new(vec![0.0, 1.0, 2.0], vec![1, 3]).unwrap();
        let cpu = run_cumtrapz(Value::Tensor(x.clone()), vec![Value::Tensor(y.clone())]).unwrap();
        let y_handle = provider
            .upload(&HostTensorView {
                data: &y.materialize_f64(),
                shape: &y.shape,
            })
            .unwrap();
        let gpu = run_cumtrapz(Value::Tensor(x), vec![Value::GpuTensor(y_handle)]).unwrap();
        let gathered = test_support::gather(gpu).expect("gather gpu");
        let expected = match cpu {
            Value::Tensor(tensor) => tensor,
            other => panic!("unexpected cpu result {other:?}"),
        };
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-9,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        assert_eq!(gathered.shape, expected.shape);
        for (actual, expected) in gathered
            .materialize_f64()
            .iter()
            .zip(expected.materialize_f64().iter())
        {
            assert!((actual - expected).abs() < tol);
        }
    }
}
