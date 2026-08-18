//! MATLAB-compatible `cross` builtin with GPU-aware semantics for RunMat.
//!
//! Implements 3-element vector cross products for row vectors, column vectors,
//! matrices of vectors, and higher-rank tensors. GPU inputs dispatch to a
//! provider-side `cross` hook when available and otherwise fall back to the
//! host implementation with result re-upload when the owning provider can
//! preserve the documented floating class and complexity.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexStorage, ComplexTensor, NumericDType, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::linalg::type_resolvers::cross_type;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const CROSS_NAME: &str = "cross";

const CROSS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cross product result.",
}];

const CROSS_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Left operand.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right operand.",
    },
];

const CROSS_INPUTS_DIM: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Left operand.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right operand.",
    },
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Dimension with extent 3.",
    },
];

const CROSS_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "C = cross(A, B)",
        inputs: &CROSS_INPUTS,
        outputs: &CROSS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = cross(A, B, dim)",
        inputs: &CROSS_INPUTS_DIM,
        outputs: &CROSS_OUTPUT,
    },
];

const CROSS_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CROSS.INVALID_ARGUMENT",
    identifier: Some("RunMat:cross:InvalidArgument"),
    when: "Argument count or dimension argument is invalid.",
    message: "cross: invalid argument",
};

const CROSS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CROSS.INVALID_INPUT",
    identifier: Some("RunMat:cross:InvalidInput"),
    when: "Inputs are unsupported or incompatible for cross product.",
    message: "cross: A and B must be the same size.",
};

const CROSS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CROSS.INTERNAL",
    identifier: Some("RunMat:cross:Internal"),
    when: "Runtime cannot materialize cross outputs.",
    message: "cross: internal runtime failure",
};

const CROSS_ERRORS: [BuiltinErrorDescriptor; 3] = [
    CROSS_ERROR_INVALID_ARGUMENT,
    CROSS_ERROR_INVALID_INPUT,
    CROSS_ERROR_INTERNAL,
];

pub const CROSS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CROSS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CROSS_ERRORS,
};

pub const CROSS_INTEGER_A_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cross-integer-a",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cross with a typed-integer A operand is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CrossIntegerAExtension"),
};
pub const CROSS_INTEGER_B_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cross-integer-b",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cross with a typed-integer B operand is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CrossIntegerBExtension"),
};
pub const CROSS_LOGICAL_A_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cross-logical-a",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cross with a logical A operand is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CrossLogicalAExtension"),
};
pub const CROSS_LOGICAL_B_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cross-logical-b",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cross with a logical B operand is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CrossLogicalBExtension"),
};
pub const CROSS_INTEGER_DIM_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cross-integer-dim",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cross with a typed-integer dim argument is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CrossIntegerDimExtension"),
};
pub const CROSS_LOGICAL_DIM_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cross-logical-dim",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cross with a logical dim argument is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CrossLogicalDimExtension"),
};
pub const CROSS_EXTENSIONS: [BuiltinExtensionDescriptor; 6] = [
    CROSS_INTEGER_A_EXTENSION,
    CROSS_INTEGER_B_EXTENSION,
    CROSS_LOGICAL_A_EXTENSION,
    CROSS_LOGICAL_B_EXTENSION,
    CROSS_INTEGER_DIM_EXTENSION,
    CROSS_LOGICAL_DIM_EXTENSION,
];
const CROSS_INTEGER_A_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "A", classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::Rejected, notes: "MATLAB documents only single and double data operands; RunMat mode admits every real or complex integer class at an explicit floating boundary, with single output when the other data operand is single and double otherwise." }];
const CROSS_INTEGER_B_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "B", classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::Rejected, notes: "The independently gated B operand accepts every real or complex integer class in RunMat mode." }];
const CROSS_INTEGER_DIM_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "dim", classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "The public cross reference documents a positive-integer scalar dim but does not list typed integer classes. RunMat mode accepts every typed integer class exactly; an ordinary double-valued integer scalar remains documented behavior." }];
pub const CROSS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor { form: "C = cross(integer_A, B, dim?)", inputs: &CROSS_INTEGER_A_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "RunMat-only integer data crosses a floating boundary. The result is single when either data operand is single and double otherwise; matching shapes and the length-three operating dimension are preserved. Resident integer inputs gather exactly before conversion and eligible results return to the owning provider." },
    BuiltinIntegerCapabilityDescriptor { form: "C = cross(A, integer_B, dim?)", inputs: &CROSS_INTEGER_B_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "The B role has the same independently gated RunMat-only floating-boundary behavior and single-dominant output-class rule." },
    BuiltinIntegerCapabilityDescriptor { form: "C = cross(A, B, integer_dim)", inputs: &CROSS_INTEGER_DIM_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "dim controls traversal only; floating data output preserves the supported input precision and residency path." },
];

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::ops::cross")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "cross",
    op_kind: GpuOpKind::Custom("cross"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("cross")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Dispatches same-provider, same-device real floating inputs to a provider-side cross implementation only when the returned handle has the required precision, real storage, input device, and provider ownership; otherwise gathers inputs, evaluates on the host, and re-uploads only when the owner can preserve class and complexity.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::linalg::ops::cross")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "cross",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Cross products allocate a fresh tensor and terminate fusion graphs.",
};

fn cross_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    cross_error_with_message(error.message, error)
}

fn cross_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(CROSS_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn cross_invalid_argument(message: impl Into<String>) -> RuntimeError {
    cross_error_with_message(message, &CROSS_ERROR_INVALID_ARGUMENT)
}

fn cross_invalid_input(message: impl Into<String>) -> RuntimeError {
    cross_error_with_message(message, &CROSS_ERROR_INVALID_INPUT)
}

fn cross_internal_error(message: impl Into<String>) -> RuntimeError {
    cross_error_with_message(message, &CROSS_ERROR_INTERNAL)
}

fn is_typed_integer_operand(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || crate::builtins::common::validation::is_typed_complex_integer(value)
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn is_logical_operand(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
}

fn ensure_cross_data_extensions(lhs: &Value, rhs: &Value) -> BuiltinResult<()> {
    if is_typed_integer_operand(lhs) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CROSS_INTEGER_A_EXTENSION,
            CROSS_NAME,
        )?;
    }
    if is_typed_integer_operand(rhs) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CROSS_INTEGER_B_EXTENSION,
            CROSS_NAME,
        )?;
    }
    if is_logical_operand(lhs) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CROSS_LOGICAL_A_EXTENSION,
            CROSS_NAME,
        )?;
    }
    if is_logical_operand(rhs) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CROSS_LOGICAL_B_EXTENSION,
            CROSS_NAME,
        )?;
    }
    Ok(())
}

fn ensure_cross_dimension_extension(dim: &Value) -> BuiltinResult<()> {
    if is_typed_integer_operand(dim) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CROSS_INTEGER_DIM_EXTENSION,
            CROSS_NAME,
        )?;
    }
    if is_logical_operand(dim) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CROSS_LOGICAL_DIM_EXTENSION,
            CROSS_NAME,
        )?;
    }
    Ok(())
}

async fn parse_dimension_arg(value: &Value) -> BuiltinResult<usize> {
    match value {
        Value::Int(_)
        | Value::Num(_)
        | Value::Bool(_)
        | Value::Tensor(_)
        | Value::LogicalArray(_)
        | Value::GpuTensor(_) => {
            let dim = tensor::dimension_from_value_async(value, CROSS_NAME, false)
                .await
                .map_err(cross_invalid_argument)?;
            dim.ok_or_else(|| {
                cross_invalid_argument(format!(
                    "{CROSS_NAME}: dimension must be numeric, got {value:?}"
                ))
            })
        }
        _ => Err(cross_invalid_argument(format!(
            "{CROSS_NAME}: dimension must be numeric, got {value:?}"
        ))),
    }
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    if err.message() == "interaction pending..." {
        return build_runtime_error("interaction pending...")
            .with_builtin(CROSS_NAME)
            .build();
    }
    let mut builder = build_runtime_error(err.message()).with_builtin(CROSS_NAME);
    if let Some(identifier) = err.identifier() {
        builder = builder.with_identifier(identifier.to_string());
    }
    if let Some(task_id) = err.context.task_id.clone() {
        builder = builder.with_task_id(task_id);
    }
    if !err.context.call_stack.is_empty() {
        builder = builder.with_call_stack(err.context.call_stack.clone());
    }
    if let Some(phase) = err.context.phase.clone() {
        builder = builder.with_phase(phase);
    }
    builder.with_source(err).build()
}

#[runtime_builtin(
    name = "cross",
    category = "math/linalg/ops",
    summary = "Compute vector cross products.",
    keywords = "cross,vector product,3d vector,gpu,linear algebra",
    accel = "custom",
    type_resolver(cross_type),
    extensions(CROSS_EXTENSIONS),
    integer_capabilities(CROSS_INTEGER_CAPABILITIES),
    descriptor(crate::builtins::math::linalg::ops::cross::CROSS_DESCRIPTOR),
    builtin_path = "crate::builtins::math::linalg::ops::cross"
)]
async fn cross_builtin(lhs: Value, rhs: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(cross_invalid_argument("cross: too many input arguments"));
    }
    ensure_cross_data_extensions(&lhs, &rhs)?;
    let dim = match rest.first() {
        Some(value) => {
            let dim = parse_dimension_arg(value).await?;
            ensure_cross_dimension_extension(value)?;
            Some(dim)
        }
        None => None,
    };
    let owner = match (&lhs, &rhs) {
        (Value::GpuTensor(handle), _) | (_, Value::GpuTensor(handle)) => Some(handle.clone()),
        _ => None,
    };
    let single_output = value_uses_single(&lhs) || value_uses_single(&rhs);
    let complex_output = value_is_complex(&lhs) || value_is_complex(&rhs);

    if let (Value::GpuTensor(lhs_handle), Value::GpuTensor(rhs_handle)) = (&lhs, &rhs) {
        let requires_floating_boundary = runmat_accelerate_api::handle_integer_type(lhs_handle)
            .is_some()
            || runmat_accelerate_api::handle_integer_type(rhs_handle).is_some()
            || runmat_accelerate_api::handle_is_logical(lhs_handle)
            || runmat_accelerate_api::handle_is_logical(rhs_handle);
        if !requires_floating_boundary
            && !complex_output
            && lhs_handle.device_id == rhs_handle.device_id
        {
            if let (Some(provider), Some(rhs_provider)) = (
                runmat_accelerate_api::provider_for_handle(lhs_handle),
                runmat_accelerate_api::provider_for_handle(rhs_handle),
            ) {
                if !std::ptr::eq(provider, rhs_provider) {
                    // Mixed-owner handles must gather independently.
                } else {
                    let expected = if single_output {
                        runmat_accelerate_api::ProviderPrecision::F32
                    } else {
                        runmat_accelerate_api::ProviderPrecision::F64
                    };
                    match provider.cross(lhs_handle, rhs_handle, dim) {
                        Ok(handle)
                            if native_result_matches_provider(
                                &handle,
                                lhs_handle.device_id,
                                provider,
                                expected,
                            ) =>
                        {
                            return Ok(Value::GpuTensor(handle));
                        }
                        Ok(handle) => {
                            free_rejected_native_handle(&handle, provider);
                        }
                        Err(err) => {
                            log::trace!("cross: provider cross fallback triggered: {err}");
                        }
                    }
                }
            }
        }
    }

    let lhs_gpu = matches!(lhs, Value::GpuTensor(_));
    let rhs_gpu = matches!(rhs, Value::GpuTensor(_));

    let lhs_host = gather_if_needed_async(&lhs)
        .await
        .map_err(map_control_flow)?;
    let rhs_host = gather_if_needed_async(&rhs)
        .await
        .map_err(map_control_flow)?;

    let has_complex = value_is_complex(&lhs_host) || value_is_complex(&rhs_host);

    let value = if has_complex {
        let lhs_complex = value_into_complex_tensor(lhs_host)?;
        let rhs_complex = value_into_complex_tensor(rhs_host)?;
        let result = cross_complex_tensor(&lhs_complex, &rhs_complex, dim, single_output)?;
        complex_tensor_into_value(result)
    } else {
        let lhs_tensor =
            tensor::value_into_tensor_for(CROSS_NAME, lhs_host).map_err(cross_invalid_input)?;
        let rhs_tensor =
            tensor::value_into_tensor_for(CROSS_NAME, rhs_host).map_err(cross_invalid_input)?;
        let result = cross_real_tensor(&lhs_tensor, &rhs_tensor, dim, single_output)?;
        if lhs_gpu || rhs_gpu {
            return promote_real_result_to_gpu(result, owner.as_ref(), single_output);
        }
        tensor::tensor_into_value(result)
    };

    if let (Some(owner), Value::ComplexTensor(tensor)) = (owner.as_ref(), &value) {
        if let Some(provider) = runmat_accelerate_api::provider_for_handle(owner) {
            let expected = if single_output {
                runmat_accelerate_api::ProviderPrecision::F32
            } else {
                runmat_accelerate_api::ProviderPrecision::F64
            };
            if provider.precision() != expected {
                return Ok(value);
            }
            return gpu_helpers::upload_complex_tensor(provider, tensor)
                .map(gpu_helpers::complex_gpu_value);
        }
    }
    Ok(value)
}

fn native_result_matches_provider(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    input_device: u32,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    expected_precision: runmat_accelerate_api::ProviderPrecision,
) -> bool {
    handle.device_id == input_device
        && runmat_accelerate_api::handle_precision(handle) == Some(expected_precision)
        && runmat_accelerate_api::handle_storage(handle)
            == runmat_accelerate_api::GpuTensorStorage::Real
        && runmat_accelerate_api::provider_for_handle(handle)
            .is_some_and(|owner| std::ptr::eq(owner, provider))
}

fn free_rejected_native_handle(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    invoked_provider: &'static dyn runmat_accelerate_api::AccelProvider,
) {
    let owner = runmat_accelerate_api::provider_for_handle(handle).unwrap_or(invoked_provider);
    if let Err(err) = owner.free(handle) {
        log::trace!("cross: failed to free rejected provider result: {err}");
    }
}

fn value_is_complex(value: &Value) -> bool {
    matches!(value, Value::Complex(_, _) | Value::ComplexTensor(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_storage(handle) == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved)
}

fn value_uses_single(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
        || matches!(value, Value::ComplexTensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_precision(handle) == Some(runmat_accelerate_api::ProviderPrecision::F32))
}

fn value_into_complex_tensor(value: Value) -> BuiltinResult<ComplexTensor> {
    match value {
        Value::ComplexTensor(t) => Ok(t),
        Value::Complex(re, im) => ComplexTensor::new(vec![(re, im)], vec![1, 1])
            .map_err(|e| cross_invalid_input(format!("{CROSS_NAME}: {e}"))),
        Value::Tensor(t) => real_tensor_to_complex(&t),
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1])
                .map_err(|e| cross_invalid_input(format!("{CROSS_NAME}: {e}")))?;
            real_tensor_to_complex(&tensor)
        }
        Value::Int(i) => {
            let tensor = Tensor::new(vec![i.to_f64()], vec![1, 1])
                .map_err(|e| cross_invalid_input(format!("{CROSS_NAME}: {e}")))?;
            real_tensor_to_complex(&tensor)
        }
        Value::Bool(b) => {
            let tensor = Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(|e| cross_invalid_input(format!("{CROSS_NAME}: {e}")))?;
            real_tensor_to_complex(&tensor)
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical).map_err(cross_invalid_input)?;
            real_tensor_to_complex(&tensor)
        }
        other => Err(cross_invalid_input(format!(
            "{CROSS_NAME}: unsupported input type {:?}; expected numeric or logical values",
            other
        ))),
    }
}

fn real_tensor_to_complex(tensor: &Tensor) -> BuiltinResult<ComplexTensor> {
    let shape = canonical_shape_tensor(tensor);
    let values = tensor::tensor_values_f64_cow(tensor);
    let mut data = Vec::with_capacity(values.len());
    for &value in values.iter() {
        data.push((value, 0.0));
    }
    ComplexTensor::new(data, shape).map_err(|e| cross_internal_error(format!("{CROSS_NAME}: {e}")))
}

pub fn cross_host_real_for_provider(
    a: &Tensor,
    b: &Tensor,
    dim: Option<usize>,
) -> BuiltinResult<Tensor> {
    cross_real_tensor(
        a,
        b,
        dim,
        a.numeric_dtype() == NumericDType::F32 || b.numeric_dtype() == NumericDType::F32,
    )
}

fn cross_real_tensor(
    a: &Tensor,
    b: &Tensor,
    dim: Option<usize>,
    single_output: bool,
) -> BuiltinResult<Tensor> {
    ensure_same_size(a, b)?;

    let shape = canonical_shape_tensor(a);
    let a_values = tensor::tensor_values_f64_cow(a);
    let b_values = tensor::tensor_values_f64_cow(b);
    let target_dim = resolve_dimension(&shape, dim)?;
    let dim_index = target_dim - 1;
    let stride_before = dim_product(&shape[..dim_index]);
    let stride_after = dim_product(&shape[dim_index + 1..]);
    let slice_stride = stride_before * 3;
    let mut output = vec![0.0f64; a_values.len()];

    for after in 0..stride_after {
        let slice_base = after * slice_stride;
        for before in 0..stride_before {
            let idx1 = slice_base + before;
            let idx2 = idx1 + stride_before;
            let idx3 = idx2 + stride_before;

            let a1 = a_values[idx1];
            let a2 = a_values[idx2];
            let a3 = a_values[idx3];
            let b1 = b_values[idx1];
            let b2 = b_values[idx2];
            let b3 = b_values[idx3];

            if single_output {
                output[idx1] = f64::from((a2 as f32) * (b3 as f32) - (a3 as f32) * (b2 as f32));
                output[idx2] = f64::from((a3 as f32) * (b1 as f32) - (a1 as f32) * (b3 as f32));
                output[idx3] = f64::from((a1 as f32) * (b2 as f32) - (a2 as f32) * (b1 as f32));
            } else {
                output[idx1] = a2 * b3 - a3 * b2;
                output[idx2] = a3 * b1 - a1 * b3;
                output[idx3] = a1 * b2 - a2 * b1;
            }
        }
    }

    if single_output {
        Tensor::from_f32(
            output.into_iter().map(|value| value as f32).collect(),
            shape,
        )
        .map_err(|e| cross_internal_error(format!("{CROSS_NAME}: {e}")))
    } else {
        Tensor::new(output, shape).map_err(|e| cross_internal_error(format!("{CROSS_NAME}: {e}")))
    }
}

fn cross_complex_tensor(
    a: &ComplexTensor,
    b: &ComplexTensor,
    dim: Option<usize>,
    single_output: bool,
) -> BuiltinResult<ComplexTensor> {
    ensure_same_size_complex(a, b)?;

    let shape = canonical_shape_complex(a);
    let target_dim = resolve_dimension(&shape, dim)?;
    let dim_index = target_dim - 1;
    let stride_before = dim_product(&shape[..dim_index]);
    let stride_after = dim_product(&shape[dim_index + 1..]);
    let slice_stride = stride_before * 3;
    let mut output = vec![(0.0f64, 0.0f64); a.materialize_f64().len()];

    for after in 0..stride_after {
        let slice_base = after * slice_stride;
        for before in 0..stride_before {
            let idx1 = slice_base + before;
            let idx2 = idx1 + stride_before;
            let idx3 = idx2 + stride_before;

            let a1 = a.materialize_f64()[idx1];
            let a2 = a.materialize_f64()[idx2];
            let a3 = a.materialize_f64()[idx3];
            let b1 = b.materialize_f64()[idx1];
            let b2 = b.materialize_f64()[idx2];
            let b3 = b.materialize_f64()[idx3];

            if single_output {
                output[idx1] = complex_sub_f32(complex_mul_f32(a2, b3), complex_mul_f32(a3, b2));
                output[idx2] = complex_sub_f32(complex_mul_f32(a3, b1), complex_mul_f32(a1, b3));
                output[idx3] = complex_sub_f32(complex_mul_f32(a1, b2), complex_mul_f32(a2, b1));
            } else {
                output[idx1] = complex_sub(complex_mul(a2, b3), complex_mul(a3, b2));
                output[idx2] = complex_sub(complex_mul(a3, b1), complex_mul(a1, b3));
                output[idx3] = complex_sub(complex_mul(a1, b2), complex_mul(a2, b1));
            }
        }
    }

    let storage = if single_output {
        ComplexStorage::F32(
            output
                .into_iter()
                .map(|(re, im)| (re as f32, im as f32))
                .collect(),
        )
    } else {
        ComplexStorage::F64(output)
    };
    ComplexTensor::from_complex_storage(storage, shape)
        .map_err(|e| cross_internal_error(format!("{CROSS_NAME}: {e}")))
}

fn complex_mul(lhs: (f64, f64), rhs: (f64, f64)) -> (f64, f64) {
    (lhs.0 * rhs.0 - lhs.1 * rhs.1, lhs.0 * rhs.1 + lhs.1 * rhs.0)
}

fn complex_sub(lhs: (f64, f64), rhs: (f64, f64)) -> (f64, f64) {
    (lhs.0 - rhs.0, lhs.1 - rhs.1)
}

fn complex_mul_f32(lhs: (f64, f64), rhs: (f64, f64)) -> (f32, f32) {
    let lhs = (lhs.0 as f32, lhs.1 as f32);
    let rhs = (rhs.0 as f32, rhs.1 as f32);
    (lhs.0 * rhs.0 - lhs.1 * rhs.1, lhs.0 * rhs.1 + lhs.1 * rhs.0)
}

fn complex_sub_f32(lhs: (f32, f32), rhs: (f32, f32)) -> (f64, f64) {
    (f64::from(lhs.0 - rhs.0), f64::from(lhs.1 - rhs.1))
}

fn ensure_same_size(a: &Tensor, b: &Tensor) -> BuiltinResult<()> {
    if tensor::tensor_element_len(a) != tensor::tensor_element_len(b)
        || canonical_shape_tensor(a) != canonical_shape_tensor(b)
    {
        return Err(cross_error(&CROSS_ERROR_INVALID_INPUT));
    }
    Ok(())
}

fn ensure_same_size_complex(a: &ComplexTensor, b: &ComplexTensor) -> BuiltinResult<()> {
    if a.materialize_f64().len() != b.materialize_f64().len()
        || canonical_shape_complex(a) != canonical_shape_complex(b)
    {
        return Err(cross_error(&CROSS_ERROR_INVALID_INPUT));
    }
    Ok(())
}

fn canonical_shape_tensor(t: &Tensor) -> Vec<usize> {
    if t.shape.is_empty() {
        vec![t.rows, t.cols]
    } else {
        t.shape.clone()
    }
}

fn canonical_shape_complex(t: &ComplexTensor) -> Vec<usize> {
    if t.shape.is_empty() {
        vec![t.rows, t.cols]
    } else {
        t.shape.clone()
    }
}

fn resolve_dimension(shape: &[usize], dim: Option<usize>) -> BuiltinResult<usize> {
    match dim {
        Some(target_dim) => {
            if target_dim > shape.len() {
                return Err(cross_invalid_input(format!(
                    "cross: dimension {} exceeds the number of array dimensions ({})",
                    target_dim,
                    shape.len()
                )));
            }
            if shape[target_dim - 1] != 3 {
                return Err(cross_invalid_input(format!(
                    "cross: dimension {} must have length 3",
                    target_dim
                )));
            }
            Ok(target_dim)
        }
        None => shape
            .iter()
            .position(|&extent| extent == 3)
            .map(|idx| idx + 1)
            .ok_or_else(|| cross_invalid_input("cross: inputs must have a dimension of length 3")),
    }
}

fn dim_product(dims: &[usize]) -> usize {
    dims.iter()
        .copied()
        .try_fold(1usize, |acc, dim| acc.checked_mul(dim))
        .expect("cross: internal dimension overflow")
}

fn promote_real_result_to_gpu(
    tensor: Tensor,
    owner: Option<&runmat_accelerate_api::GpuTensorHandle>,
    single_output: bool,
) -> BuiltinResult<Value> {
    let provider = match owner.and_then(runmat_accelerate_api::provider_for_handle) {
        Some(provider) => provider,
        None => return Ok(tensor::tensor_into_value(tensor)),
    };
    let expected = if single_output {
        runmat_accelerate_api::ProviderPrecision::F32
    } else {
        runmat_accelerate_api::ProviderPrecision::F64
    };
    if provider.precision() != expected {
        return Ok(tensor::tensor_into_value(tensor));
    }
    match gpu_helpers::upload_tensor(provider, &tensor) {
        Ok(handle) => Ok(Value::GpuTensor(handle)),
        Err(_) => Ok(tensor::tensor_into_value(tensor)),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::{AccelProvider, HostTensorView};
    use runmat_builtins::{
        ComplexStorage, IntValue, IntegerStorage, LiteralValue, LogicalArray, ResolveContext, Type,
    };

    fn unwrap_error(err: crate::RuntimeError) -> crate::RuntimeError {
        err
    }

    #[test]
    fn cross_type_preserves_known_shape() {
        let out = cross_type(
            &[
                Type::Tensor {
                    shape: Some(vec![Some(1), Some(3)]),
                },
                Type::Tensor {
                    shape: Some(vec![Some(1), Some(3)]),
                },
            ],
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
    fn cross_type_uses_literal_dim() {
        let ctx = ResolveContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Number(2.0),
        ]);
        let out = cross_type(
            &[
                Type::Tensor {
                    shape: Some(vec![Some(2), Some(3)]),
                },
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
                shape: Some(vec![Some(2), Some(3)])
            }
        );
    }

    #[test]
    fn cross_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = CROSS_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert!(labels.contains(&"C = cross(A, B)"));
        assert!(labels.contains(&"C = cross(A, B, dim)"));
        assert_eq!(CROSS_INTEGER_CAPABILITIES.len(), 3);
        assert_eq!(CROSS_EXTENSIONS.len(), 6);
    }

    #[test]
    fn cross_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = CROSS_DESCRIPTOR.errors.iter().map(|err| err.code).collect();
        assert!(codes.contains(&"RM.CROSS.INVALID_ARGUMENT"));
        assert!(codes.contains(&"RM.CROSS.INVALID_INPUT"));
        assert!(codes.contains(&"RM.CROSS.INTERNAL"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cross_row_vectors() {
        let lhs = Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap();
        let rhs = Tensor::new(vec![0.0, 1.0, 0.0], vec![1, 3]).unwrap();
        let value =
            cross_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new()).expect("cross");
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                assert_eq!(t.materialize_f64(), vec![0.0, 0.0, 1.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cross_column_vectors() {
        let lhs = Tensor::new(vec![1.0, 0.0, 0.0], vec![3, 1]).unwrap();
        let rhs = Tensor::new(vec![0.0, 1.0, 0.0], vec![3, 1]).unwrap();
        let value =
            cross_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new()).expect("cross");
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 1]);
                assert_eq!(t.materialize_f64(), vec![0.0, 0.0, 1.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cross_rowwise_dimension_argument() {
        let lhs = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 3]).unwrap();
        let rhs = Tensor::new(vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0], vec![2, 3]).unwrap();
        let value = cross_builtin(
            Value::Tensor(lhs),
            Value::Tensor(rhs),
            vec![Value::Num(2.0)],
        )
        .expect("cross");
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 3]);
                assert_eq!(t.materialize_f64(), vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn cross_reads_typed_integer_tensors_and_dimension_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let lhs = Tensor::new_integer(IntegerStorage::I16(vec![1, 0, 0, 1, 0, 0]), vec![2, 3])
            .expect("lhs");
        let rhs = Tensor::new_integer(IntegerStorage::U16(vec![0, 0, 1, 0, 0, 1]), vec![2, 3])
            .expect("rhs");
        let dim = Tensor::new_integer(IntegerStorage::U16(vec![2]), vec![1, 1]).expect("dim");

        let value = cross_builtin(
            Value::Tensor(lhs),
            Value::Tensor(rhs),
            vec![Value::Tensor(dim)],
        )
        .expect("cross");
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 3]);
                assert_eq!(t.materialize_f64(), vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn cross_uses_integer_storage_length_when_mirrors_are_empty() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let lhs = Tensor::new_integer(IntegerStorage::I16(vec![1, 0, 0]), vec![1, 3]).expect("lhs");
        let rhs = Tensor::new_integer(IntegerStorage::U16(vec![0, 1, 0]), vec![1, 3]).expect("rhs");

        let value =
            cross_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new()).expect("cross");
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                assert_eq!(t.materialize_f64(), vec![0.0, 0.0, 1.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn cross_runmat_extension_accepts_all_eight_integer_classes() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let cases = [
            IntegerStorage::I8(vec![1, 0, 0]),
            IntegerStorage::I16(vec![1, 0, 0]),
            IntegerStorage::I32(vec![1, 0, 0]),
            IntegerStorage::I64(vec![1, 0, 0]),
            IntegerStorage::U8(vec![1, 0, 0]),
            IntegerStorage::U16(vec![1, 0, 0]),
            IntegerStorage::U32(vec![1, 0, 0]),
            IntegerStorage::U64(vec![1, 0, 0]),
        ];
        for storage in cases {
            let lhs = Tensor::new_integer(storage, vec![1, 3]).unwrap();
            let rhs = Tensor::new(vec![0.0, 1.0, 0.0], vec![1, 3]).unwrap();
            let Value::Tensor(output) =
                cross_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new()).unwrap()
            else {
                panic!("tensor")
            };
            assert_eq!(output.numeric_dtype(), runmat_builtins::NumericDType::F64);
            assert_eq!(output.materialize_f64(), vec![0.0, 0.0, 1.0]);
        }
    }

    #[test]
    fn cross_integer_roles_are_independently_gated() {
        let integer = Value::Tensor(
            Tensor::new_integer(IntegerStorage::I16(vec![1, 0, 0]), vec![1, 3]).unwrap(),
        );
        let floating = Value::Tensor(Tensor::new(vec![0.0, 1.0, 0.0], vec![1, 3]).unwrap());
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let left = cross_builtin(integer.clone(), floating.clone(), Vec::new()).unwrap_err();
        assert_eq!(
            left.identifier(),
            CROSS_INTEGER_A_EXTENSION.error_identifier
        );
        let right = cross_builtin(floating, integer, Vec::new()).unwrap_err();
        assert_eq!(
            right.identifier(),
            CROSS_INTEGER_B_EXTENSION.error_identifier
        );
    }

    #[test]
    fn cross_typed_integer_dimension_is_a_gated_extension() {
        let lhs = Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap());
        let rhs = Value::Tensor(Tensor::new(vec![0.0, 1.0, 0.0], vec![1, 3]).unwrap());
        let dim = Value::Int(IntValue::U8(2));
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = cross_builtin(lhs, rhs, vec![dim]).unwrap_err();
        assert_eq!(
            error.identifier(),
            CROSS_INTEGER_DIM_EXTENSION.error_identifier
        );
    }

    #[test]
    fn cross_malformed_typed_dimension_fails_before_extension_gate() {
        let lhs = Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap());
        let rhs = Value::Tensor(Tensor::new(vec![0.0, 1.0, 0.0], vec![1, 3]).unwrap());
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = cross_builtin(lhs, rhs, vec![Value::Int(IntValue::U8(0))]).unwrap_err();
        assert_eq!(error.identifier(), CROSS_ERROR_INVALID_ARGUMENT.identifier);
        assert!(error.message().contains("dimension must be >= 1"));
    }

    #[test]
    fn cross_logical_dimension_is_a_gated_extension() {
        let lhs = Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0], vec![3, 1]).unwrap());
        let rhs = Value::Tensor(Tensor::new(vec![0.0, 1.0, 0.0], vec![3, 1]).unwrap());
        let dim = Value::Bool(true);
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = cross_builtin(lhs, rhs, vec![dim]).unwrap_err();
        assert_eq!(
            error.identifier(),
            CROSS_LOGICAL_DIM_EXTENSION.error_identifier
        );
    }

    #[test]
    fn cross_logical_scalar_dimension_runs_when_extension_is_enabled() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let lhs = Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0], vec![3, 1]).unwrap());
        let rhs = Value::Tensor(Tensor::new(vec![0.0, 1.0, 0.0], vec![3, 1]).unwrap());
        let Value::Tensor(output) =
            cross_builtin(lhs, rhs, vec![Value::Bool(true)]).expect("logical dim extension")
        else {
            panic!("tensor")
        };
        assert_eq!(output.materialize_f64(), vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn cross_resident_typed_integer_dimension_gathers_exactly() {
        test_support::with_test_provider(|provider| {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            let dim = Tensor::new_integer(IntegerStorage::U64(vec![2]), vec![1, 1]).unwrap();
            let dim = gpu_helpers::upload_tensor(provider, &dim).unwrap();
            let lhs = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 3]).unwrap();
            let rhs = Tensor::new(vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0], vec![2, 3]).unwrap();
            let Value::Tensor(output) = cross_builtin(
                Value::Tensor(lhs),
                Value::Tensor(rhs),
                vec![Value::GpuTensor(dim)],
            )
            .unwrap() else {
                panic!("host tensor")
            };
            assert_eq!(output.materialize_f64(), vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0]);
        });
    }

    #[test]
    fn cross_logical_roles_are_independently_gated() {
        let logical = Value::LogicalArray(LogicalArray::new(vec![1, 0, 0], vec![1, 3]).unwrap());
        let floating = Value::Tensor(Tensor::new(vec![0.0, 1.0, 0.0], vec![1, 3]).unwrap());
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let left = cross_builtin(logical.clone(), floating.clone(), Vec::new()).unwrap_err();
        assert_eq!(
            left.identifier(),
            CROSS_LOGICAL_A_EXTENSION.error_identifier
        );
        let right = cross_builtin(floating, logical, Vec::new()).unwrap_err();
        assert_eq!(
            right.identifier(),
            CROSS_LOGICAL_B_EXTENSION.error_identifier
        );
    }

    #[test]
    fn cross_typed_complex_integer_returns_complex_double() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let lhs = runmat_builtins::IntegerComplexStorage::new(
            IntegerStorage::I16(vec![1, 0, 0]),
            IntegerStorage::I16(vec![1, 0, 0]),
        )
        .unwrap();
        let rhs = runmat_builtins::IntegerComplexStorage::new(
            IntegerStorage::U16(vec![0, 1, 0]),
            IntegerStorage::U16(vec![0, 1, 0]),
        )
        .unwrap();
        let lhs = ComplexTensor::from_complex_storage(
            runmat_builtins::ComplexStorage::Integer(lhs),
            vec![1, 3],
        )
        .unwrap();
        let rhs = ComplexTensor::from_complex_storage(
            runmat_builtins::ComplexStorage::Integer(rhs),
            vec![1, 3],
        )
        .unwrap();
        let Value::ComplexTensor(output) = cross_builtin(
            Value::ComplexTensor(lhs),
            Value::ComplexTensor(rhs),
            Vec::new(),
        )
        .expect("complex integer cross") else {
            panic!("complex tensor")
        };
        assert_eq!(output.numeric_dtype(), runmat_builtins::NumericDType::F64);
        assert_eq!(output.materialize_f64()[2], (0.0, 2.0));
    }

    #[test]
    fn cross_resident_integer_extension_gathers_and_restores_double() {
        test_support::with_test_provider(|provider| {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            let lhs = Tensor::new_integer(IntegerStorage::I64(vec![1, 0, 0]), vec![1, 3]).unwrap();
            let rhs = Tensor::new_integer(IntegerStorage::U64(vec![0, 1, 0]), vec![1, 3]).unwrap();
            let lhs = gpu_helpers::upload_tensor(provider, &lhs).unwrap();
            let rhs = gpu_helpers::upload_tensor(provider, &rhs).unwrap();
            let Value::GpuTensor(output) =
                cross_builtin(Value::GpuTensor(lhs), Value::GpuTensor(rhs), Vec::new()).unwrap()
            else {
                panic!("resident output")
            };
            assert_eq!(runmat_accelerate_api::handle_integer_type(&output), None);
            let gathered = test_support::gather(Value::GpuTensor(output)).unwrap();
            assert_eq!(gathered.numeric_dtype(), runmat_builtins::NumericDType::F64);
            assert_eq!(gathered.materialize_f64(), vec![0.0, 0.0, 1.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cross_nd_along_third_dimension() {
        let lhs = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![1, 2, 3]).unwrap();
        let rhs = Tensor::new(vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0], vec![1, 2, 3]).unwrap();
        let value =
            cross_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new()).expect("cross");
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2, 3]);
                assert_eq!(t.materialize_f64(), vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cross_complex_vectors() {
        let lhs = ComplexTensor::new(vec![(1.0, 1.0), (0.0, 0.0), (0.0, 0.0)], vec![1, 3]).unwrap();
        let rhs =
            ComplexTensor::new(vec![(0.0, 0.0), (1.0, -2.0), (0.0, 0.0)], vec![1, 3]).unwrap();
        let value = cross_builtin(
            Value::ComplexTensor(lhs),
            Value::ComplexTensor(rhs),
            Vec::new(),
        )
        .expect("cross");
        match value {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                assert_eq!(t.materialize_f64()[0], (0.0, 0.0));
                assert_eq!(t.materialize_f64()[1], (0.0, 0.0));
                assert_eq!(t.materialize_f64()[2], (3.0, -1.0));
            }
            other => panic!("expected complex tensor result, got {other:?}"),
        }
    }

    #[test]
    fn cross_complex_single_retains_class_and_explicit_complexity() {
        let lhs = ComplexTensor::from_complex_storage(
            ComplexStorage::F32(vec![(1.0, 0.0), (0.0, 0.0), (0.0, 0.0)]),
            vec![1, 3],
        )
        .unwrap();
        let rhs = ComplexTensor::from_complex_storage(
            ComplexStorage::F32(vec![(0.0, 0.0), (1.0, 0.0), (0.0, 0.0)]),
            vec![1, 3],
        )
        .unwrap();
        let Value::ComplexTensor(output) = cross_builtin(
            Value::ComplexTensor(lhs),
            Value::ComplexTensor(rhs),
            Vec::new(),
        )
        .unwrap() else {
            panic!("explicitly complex tensor")
        };
        assert_eq!(output.numeric_dtype(), runmat_builtins::NumericDType::F32);
        assert_eq!(
            output.materialize_f64(),
            vec![(0.0, 0.0), (0.0, 0.0), (1.0, 0.0)]
        );
    }

    #[test]
    fn cross_resident_complex_gathers_computes_and_restores() {
        test_support::with_test_provider(|provider| {
            let lhs =
                ComplexTensor::new(vec![(1.0, 1.0), (0.0, 0.0), (0.0, 0.0)], vec![1, 3]).unwrap();
            let rhs =
                ComplexTensor::new(vec![(0.0, 0.0), (1.0, -2.0), (0.0, 0.0)], vec![1, 3]).unwrap();
            let lhs = gpu_helpers::upload_complex_tensor(provider, &lhs).unwrap();
            let rhs = gpu_helpers::upload_complex_tensor(provider, &rhs).unwrap();
            let Value::GpuTensor(output) =
                cross_builtin(Value::GpuTensor(lhs), Value::GpuTensor(rhs), Vec::new()).unwrap()
            else {
                panic!("resident complex output")
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&output),
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            );
            let Value::ComplexTensor(gathered) =
                block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(output))).unwrap()
            else {
                panic!("complex tensor")
            };
            assert_eq!(
                gathered.materialize_f64(),
                vec![(0.0, 0.0), (0.0, 0.0), (3.0, -1.0)]
            );
        });
    }

    #[test]
    fn cross_mixed_provider_inputs_gather_independently() {
        let _guard = test_support::accel_test_lock();
        let provider_a: &'static runmat_accelerate::simple_provider::InProcessProvider = Box::leak(
            Box::new(runmat_accelerate::simple_provider::InProcessProvider::new()),
        );
        let provider_b: &'static runmat_accelerate::simple_provider::InProcessProvider = Box::leak(
            Box::new(runmat_accelerate::simple_provider::InProcessProvider::new()),
        );
        unsafe {
            runmat_accelerate_api::register_provider(provider_a);
            runmat_accelerate_api::register_provider(provider_b);
        }
        let lhs = provider_a
            .upload(&HostTensorView {
                data: &[1.0, 0.0, 0.0],
                shape: &[1, 3],
            })
            .unwrap();
        let rhs = provider_b
            .upload(&HostTensorView {
                data: &[0.0, 1.0, 0.0],
                shape: &[1, 3],
            })
            .unwrap();
        let Value::GpuTensor(output) =
            cross_builtin(Value::GpuTensor(lhs), Value::GpuTensor(rhs), Vec::new()).unwrap()
        else {
            panic!("owner-restored output")
        };
        assert_eq!(output.device_id, provider_a.device_id());
        let gathered = block_on(provider_a.download(&output)).unwrap();
        assert_eq!(gathered.data, vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn cross_rejects_native_output_with_wrong_single_precision() {
        test_support::with_test_provider(|provider| {
            let lhs = gpu_helpers::upload_tensor(
                provider,
                &runmat_builtins::Tensor::from_f32(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap(),
            )
            .unwrap();
            let rhs = provider
                .upload(&HostTensorView {
                    data: &[0.0, 1.0, 0.0],
                    shape: &[1, 3],
                })
                .unwrap();
            let Value::Tensor(output) =
                cross_builtin(Value::GpuTensor(lhs), Value::GpuTensor(rhs), Vec::new()).unwrap()
            else {
                panic!("class-preserving host fallback")
            };
            assert_eq!(output.numeric_dtype(), runmat_builtins::NumericDType::F32);
            assert_eq!(output.materialize_f64(), vec![0.0, 0.0, 1.0]);
        });
    }

    #[test]
    fn cross_rejects_native_result_owned_by_another_provider() {
        let _guard = test_support::accel_test_lock();
        let provider_a: &'static runmat_accelerate::simple_provider::InProcessProvider = Box::leak(
            Box::new(runmat_accelerate::simple_provider::InProcessProvider::new()),
        );
        let provider_b: &'static runmat_accelerate::simple_provider::InProcessProvider = Box::leak(
            Box::new(runmat_accelerate::simple_provider::InProcessProvider::new()),
        );
        unsafe {
            runmat_accelerate_api::register_provider(provider_a);
            runmat_accelerate_api::register_provider(provider_b);
        }
        let wrong_owner = provider_b
            .upload(&HostTensorView {
                data: &[1.0],
                shape: &[1, 1],
            })
            .unwrap();
        assert!(!native_result_matches_provider(
            &wrong_owner,
            provider_a.device_id(),
            provider_a,
            runmat_accelerate_api::ProviderPrecision::F64,
        ));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cross_promotes_logical_inputs() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let lhs = LogicalArray::new(vec![1, 0, 0], vec![1, 3]).unwrap();
        let rhs = Tensor::new(vec![0.0, 1.0, 0.0], vec![1, 3]).unwrap();
        let value =
            cross_builtin(Value::LogicalArray(lhs), Value::Tensor(rhs), Vec::new()).expect("cross");
        match value {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![0.0, 0.0, 1.0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cross_errors_when_shapes_mismatch() {
        let lhs = Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap();
        let rhs = Tensor::new(vec![0.0, 1.0, 0.0, 0.0], vec![2, 2]).unwrap();
        let err = unwrap_error(
            cross_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new()).expect_err("cross"),
        );
        assert_eq!(err.identifier(), CROSS_ERROR_INVALID_INPUT.identifier);
        assert!(err.message().contains("A and B must be the same size"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cross_errors_when_no_dimension_has_length_three() {
        let lhs = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let rhs = Tensor::new(vec![4.0, 3.0, 2.0, 1.0], vec![2, 2]).unwrap();
        let err = unwrap_error(
            cross_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new())
                .expect_err("expected cross error"),
        );
        assert_eq!(err.identifier(), CROSS_ERROR_INVALID_INPUT.identifier);
        assert!(err.message().contains("dimension of length 3"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cross_errors_when_dimension_exceeds_rank() {
        let lhs = Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap();
        let rhs = Tensor::new(vec![0.0, 1.0, 0.0], vec![1, 3]).unwrap();
        let err = unwrap_error(
            cross_builtin(
                Value::Tensor(lhs),
                Value::Tensor(rhs),
                vec![Value::Num(3.0)],
            )
            .expect_err("expected rank error"),
        );
        assert_eq!(err.identifier(), CROSS_ERROR_INVALID_INPUT.identifier);
        assert!(err
            .message()
            .contains("exceeds the number of array dimensions"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cross_errors_when_dimension_length_is_not_three() {
        let lhs = Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap();
        let rhs = Tensor::new(vec![0.0, 1.0, 0.0], vec![1, 3]).unwrap();
        let err = unwrap_error(
            cross_builtin(
                Value::Tensor(lhs),
                Value::Tensor(rhs),
                vec![Value::Num(1.0)],
            )
            .expect_err("expected length error"),
        );
        assert!(err.message().contains("must have length 3"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cross_dimension_zero_errors() {
        let lhs = Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap();
        let rhs = Tensor::new(vec![0.0, 1.0, 0.0], vec![1, 3]).unwrap();
        let err = unwrap_error(
            cross_builtin(
                Value::Tensor(lhs),
                Value::Tensor(rhs),
                vec![Value::Num(0.0)],
            )
            .expect_err("expected dimension error"),
        );
        assert_eq!(err.identifier(), CROSS_ERROR_INVALID_ARGUMENT.identifier);
        assert!(err.message().contains("dimension must be >= 1"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cross_dimension_non_integer_errors() {
        let lhs = Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap();
        let rhs = Tensor::new(vec![0.0, 1.0, 0.0], vec![1, 3]).unwrap();
        let err = unwrap_error(
            cross_builtin(
                Value::Tensor(lhs),
                Value::Tensor(rhs),
                vec![Value::Num(1.5)],
            )
            .expect_err("expected integer dimension error"),
        );
        assert_eq!(err.identifier(), CROSS_ERROR_INVALID_ARGUMENT.identifier);
        assert!(err.message().contains("dimension must be an integer"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cross_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let lhs = Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap();
            let rhs = Tensor::new(vec![0.0, 1.0, 0.0], vec![1, 3]).unwrap();
            let view_lhs = HostTensorView {
                data: &lhs.materialize_f64(),
                shape: &lhs.shape,
            };
            let view_rhs = HostTensorView {
                data: &rhs.materialize_f64(),
                shape: &rhs.shape,
            };
            let gpu_lhs = provider.upload(&view_lhs).expect("upload lhs");
            let gpu_rhs = provider.upload(&view_rhs).expect("upload rhs");
            let value = cross_builtin(
                Value::GpuTensor(gpu_lhs),
                Value::GpuTensor(gpu_rhs),
                Vec::new(),
            )
            .expect("cross");
            match value {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert_eq!(gathered.shape, vec![1, 3]);
                    assert_eq!(gathered.materialize_f64(), vec![0.0, 0.0, 1.0]);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[test]
    fn cross_frees_rejected_native_result_before_fallback() {
        test_support::with_rejecting_native_result_provider(|provider| {
            let lhs = provider
                .upload(&HostTensorView {
                    data: &[1.0, 0.0, 0.0],
                    shape: &[1, 3],
                })
                .unwrap();
            let rhs = provider
                .upload(&HostTensorView {
                    data: &[0.0, 1.0, 0.0],
                    shape: &[1, 3],
                })
                .unwrap();
            let result = cross_builtin(Value::GpuTensor(lhs), Value::GpuTensor(rhs), Vec::new())
                .expect("cross fallback");
            let result = test_support::gather(result).expect("gather fallback");
            assert_eq!(result.shape, vec![1, 3]);
            assert_eq!(result.materialize_f64(), vec![0.0, 0.0, 1.0]);
            assert_eq!(provider.free_count(), 1);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn cross_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let lhs = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![2, 3]).unwrap();
        let rhs = Tensor::new(vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0], vec![2, 3]).unwrap();
        let cpu = cross_real_tensor(&lhs, &rhs, Some(2), false).expect("cpu cross");
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let view_lhs = HostTensorView {
            data: &lhs.materialize_f64(),
            shape: &lhs.shape,
        };
        let view_rhs = HostTensorView {
            data: &rhs.materialize_f64(),
            shape: &rhs.shape,
        };
        let gpu_lhs = provider.upload(&view_lhs).expect("upload lhs");
        let gpu_rhs = provider.upload(&view_rhs).expect("upload rhs");
        let gpu_value = cross_builtin(
            Value::GpuTensor(gpu_lhs),
            Value::GpuTensor(gpu_rhs),
            vec![Value::Num(2.0)],
        )
        .expect("gpu cross");
        let gathered = test_support::gather(gpu_value).expect("gather");
        assert_eq!(gathered.shape, cpu.shape);
        assert_eq!(gathered.materialize_f64(), cpu.materialize_f64());
    }

    fn cross_builtin(lhs: Value, rhs: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::cross_builtin(lhs, rhs, rest))
    }
}
