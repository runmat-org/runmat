//! MATLAB-compatible `hypot` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage, ProviderPrecision};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexStorage, NumericStorage, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{
    broadcast::BroadcastPlan, gpu_helpers, map_control_flow_with_builtin, tensor,
};
use crate::builtins::math::type_resolvers::numeric_binary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::hypot")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "hypot",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Binary {
        name: "elem_hypot",
        commutative: true,
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers can execute hypot in a single binary kernel; the runtime gathers to host when the hook is unavailable or shapes require implicit expansion.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::hypot")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "hypot",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let a = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            let b = ctx.inputs.get(1).ok_or(FusionError::MissingInput(1))?;
            Ok(format!("hypot({a}, {b})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion emits WGSL hypot(a, b); providers may override via elem_hypot.",
};

const BUILTIN_NAME: &str = "hypot";

const HYPOT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "R",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elementwise Euclidean norm result.",
}];

const HYPOT_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Left operand.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right operand.",
    },
];

const HYPOT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "R = hypot(X, Y)",
    inputs: &HYPOT_INPUTS,
    outputs: &HYPOT_OUTPUT,
}];

const HYPOT_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HYPOT.INVALID_INPUT",
    identifier: Some("RunMat:hypot:InvalidInput"),
    when: "Input value cannot be converted to supported numeric form.",
    message: "hypot: invalid input",
};

const HYPOT_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HYPOT.SIZE_MISMATCH",
    identifier: Some("RunMat:hypot:SizeMismatch"),
    when: "Operands are not broadcast-compatible.",
    message: "hypot: size mismatch",
};

const HYPOT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HYPOT.INTERNAL",
    identifier: Some("RunMat:hypot:Internal"),
    when: "Internal gather/provider/tensor construction failed.",
    message: "hypot: internal error",
};

const HYPOT_ERRORS: [BuiltinErrorDescriptor; 3] = [
    HYPOT_ERROR_INVALID_INPUT,
    HYPOT_ERROR_SIZE_MISMATCH,
    HYPOT_ERROR_INTERNAL,
];

const HYPOT_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "hypot-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "hypot with integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:HypotIntegerInputExtension"),
};
const HYPOT_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "hypot-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "hypot with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:HypotLogicalInputExtension"),
};
const HYPOT_CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "hypot-character-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "hypot with character input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:HypotCharacterInputExtension"),
};
const HYPOT_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    HYPOT_INTEGER_INPUT_EXTENSION,
    HYPOT_LOGICAL_INPUT_EXTENSION,
    HYPOT_CHARACTER_INPUT_EXTENSION,
];

const HYPOT_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes are accepted only in RunMat extension mode after exact binary64 validation.",
    },
    BuiltinIntegerInputCapability {
        name: "Y",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes are accepted only in RunMat extension mode after exact binary64 validation.",
    },
];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "R = hypot(integer_X, Y) or hypot(X, integer_Y)",
        inputs: &HYPOT_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::BroadcastCompatible,
        notes: "Each admitted integer operand crosses an exact binary64 boundary before stable hypot evaluation. Resident integers gather exactly through their owners and never reach floating provider hooks.",
    }];

pub const HYPOT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &HYPOT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &HYPOT_ERRORS,
};

fn hypot_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("{}: {}", error.message, detail)).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn hypot_terminal_error(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {detail}", error.message))
        .with_builtin(BUILTIN_NAME)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "hypot",
    category = "math/elementwise",
    summary = "Compute element-wise Euclidean norms with hypot.",
    keywords = "hypot,euclidean norm,distance,gpu",
    accel = "binary",
    type_resolver(numeric_binary_type),
    descriptor(crate::builtins::math::elementwise::hypot::HYPOT_DESCRIPTOR),
    extensions(HYPOT_EXTENSIONS),
    integer_capabilities(crate::builtins::math::elementwise::hypot::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::elementwise::hypot"
)]
async fn hypot_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    ensure_hypot_extensions(&lhs, &rhs)?;
    crate::builtins::common::validation::reject_typed_complex_integer(&lhs, BUILTIN_NAME)?;
    crate::builtins::common::validation::reject_typed_complex_integer(&rhs, BUILTIN_NAME)?;
    crate::builtins::math::trigonometry::inverse_helpers::ensure_integer_exact_f64(
        &lhs,
        BUILTIN_NAME,
    )?;
    crate::builtins::math::trigonometry::inverse_helpers::ensure_integer_exact_f64(
        &rhs,
        BUILTIN_NAME,
    )?;
    match (lhs, rhs) {
        (Value::GpuTensor(a), Value::GpuTensor(b)) => hypot_gpu_pair(a, b).await,
        (Value::GpuTensor(a), other) => hypot_gpu_host(a, other, true).await,
        (other, Value::GpuTensor(b)) => hypot_gpu_host(b, other, false).await,
        (left, right) => hypot_host(left, right),
    }
}

async fn hypot_gpu_pair(a: GpuTensorHandle, b: GpuTensorHandle) -> BuiltinResult<Value> {
    let has_integer_input = runmat_accelerate_api::handle_integer_type(&a).is_some()
        || runmat_accelerate_api::handle_integer_type(&b).is_some();
    let provider = runmat_accelerate_api::provider_for_handle(&a).ok_or_else(|| {
        hypot_terminal_error(
            &HYPOT_ERROR_INTERNAL,
            "GPU provider unavailable for left input",
        )
    })?;
    let right_provider = runmat_accelerate_api::provider_for_handle(&b).ok_or_else(|| {
        hypot_terminal_error(
            &HYPOT_ERROR_INTERNAL,
            "GPU provider unavailable for right input",
        )
    })?;
    let same_owner = std::ptr::eq(provider, right_provider);
    let real_floating = !has_integer_input
        && !runmat_accelerate_api::handle_is_logical(&a)
        && !runmat_accelerate_api::handle_is_logical(&b)
        && runmat_accelerate_api::handle_storage(&a) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_storage(&b) == GpuTensorStorage::Real;
    if same_owner && real_floating && a.shape == b.shape {
        match provider.elem_hypot(&a, &b).await {
            Ok(handle) if valid_hypot_gpu_output(&handle, &a, &b, provider) => {
                return Ok(gpu_helpers::resident_gpu_value(handle));
            }
            Ok(handle) => {
                free_rejected_hypot_output(&handle, &[&a, &b]);
                return Err(hypot_terminal_error(
                    &HYPOT_ERROR_INTERNAL,
                    "provider elem_hypot returned malformed output",
                ));
            }
            Err(err) if hypot_provider_operation_unsupported(&err, "elem_hypot") => {}
            Err(err) => {
                return Err(hypot_terminal_error(
                    &HYPOT_ERROR_INTERNAL,
                    format!("provider elem_hypot failed: {err}"),
                ));
            }
        }
    }
    let left = gpu_helpers::download_value_preserving_residency_async(provider, &a)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    let right = gpu_helpers::download_value_preserving_residency_async(right_provider, &b)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    crate::builtins::math::trigonometry::inverse_helpers::ensure_integer_exact_f64(
        &left,
        BUILTIN_NAME,
    )?;
    crate::builtins::math::trigonometry::inverse_helpers::ensure_integer_exact_f64(
        &right,
        BUILTIN_NAME,
    )?;
    let output = hypot_host(left, right)?;
    restore_hypot_gpu_output(provider, &a, output)
}

async fn hypot_gpu_host(
    handle: GpuTensorHandle,
    host: Value,
    gpu_is_left: bool,
) -> BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider_for_handle(&handle).ok_or_else(|| {
        hypot_terminal_error(&HYPOT_ERROR_INTERNAL, "GPU provider unavailable for input")
    })?;
    let gathered = gpu_helpers::download_value_preserving_residency_async(provider, &handle)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    crate::builtins::math::trigonometry::inverse_helpers::ensure_integer_exact_f64(
        &gathered,
        BUILTIN_NAME,
    )?;
    let output = if gpu_is_left {
        hypot_host(gathered, host)?
    } else {
        hypot_host(host, gathered)?
    };
    restore_hypot_gpu_output(provider, &handle, output)
}

fn ensure_hypot_extensions(lhs: &Value, rhs: &Value) -> BuiltinResult<()> {
    for value in [lhs, rhs] {
        let integer = matches!(value, Value::Int(_))
            || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
            || matches!(value, Value::ComplexTensor(tensor) if tensor.integer_storage().is_some())
            || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some());
        if integer {
            crate::compatibility::ensure_builtin_extension_enabled(
                &HYPOT_INTEGER_INPUT_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
        let logical = matches!(value, Value::Bool(_) | Value::LogicalArray(_))
            || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle));
        if logical {
            crate::compatibility::ensure_builtin_extension_enabled(
                &HYPOT_LOGICAL_INPUT_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
        if matches!(value, Value::CharArray(_)) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &HYPOT_CHARACTER_INPUT_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
    }
    Ok(())
}

fn hypot_provider_operation_unsupported(error: &anyhow::Error, operation: &str) -> bool {
    error
        .chain()
        .any(|cause| cause.to_string() == format!("{operation} not supported by provider"))
}

fn restore_hypot_gpu_output(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    input: &GpuTensorHandle,
    value: Value,
) -> BuiltinResult<Value> {
    let tensor = match value {
        Value::Tensor(tensor) => tensor,
        Value::Num(value) => Tensor::new(vec![value], vec![1, 1])
            .map_err(|source| hypot_terminal_error(&HYPOT_ERROR_INTERNAL, source))?,
        other => {
            return Err(hypot_terminal_error(
                &HYPOT_ERROR_INTERNAL,
                format!("unexpected host fallback result {other:?}"),
            ));
        }
    };
    let expected_precision = match tensor.numeric_dtype() {
        runmat_value::NumericDType::F32 => ProviderPrecision::F32,
        _ => ProviderPrecision::F64,
    };
    if provider.precision() != expected_precision {
        return Ok(tensor::tensor_into_value(tensor));
    }
    let expected_shape = tensor.shape.clone();
    let output = gpu_helpers::upload_tensor(provider, &tensor).map_err(|source| {
        hypot_terminal_error(
            &HYPOT_ERROR_INTERNAL,
            format!("failed to restore fallback result to input provider: {source}"),
        )
    })?;
    if !valid_restored_hypot_output(
        &output,
        input,
        provider,
        &expected_shape,
        expected_precision,
    ) {
        free_rejected_hypot_output(&output, &[input]);
        return Err(hypot_terminal_error(
            &HYPOT_ERROR_INTERNAL,
            "provider upload returned malformed fallback output",
        ));
    }
    Ok(gpu_helpers::resident_gpu_value(output))
}

fn valid_hypot_gpu_output(
    output: &GpuTensorHandle,
    lhs: &GpuTensorHandle,
    rhs: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
) -> bool {
    let expected_precision = if runmat_accelerate_api::handle_precision(lhs)
        == Some(ProviderPrecision::F32)
        && runmat_accelerate_api::handle_precision(rhs) == Some(ProviderPrecision::F32)
    {
        ProviderPrecision::F32
    } else {
        ProviderPrecision::F64
    };
    valid_restored_hypot_output(output, lhs, provider, &lhs.shape, expected_precision)
        && !hypot_gpu_handles_alias(output, rhs)
}

fn valid_restored_hypot_output(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    expected_shape: &[usize],
    expected_precision: ProviderPrecision,
) -> bool {
    output.shape == expected_shape
        && output.device_id == input.device_id
        && !hypot_gpu_handles_alias(output, input)
        && runmat_accelerate_api::handle_storage(output) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_integer_type(output).is_none()
        && !runmat_accelerate_api::handle_is_logical(output)
        && runmat_accelerate_api::handle_precision(output) == Some(expected_precision)
        && runmat_accelerate_api::provider_for_handle(output)
            .is_some_and(|owner| std::ptr::eq(owner, provider))
}

fn hypot_gpu_handles_alias(lhs: &GpuTensorHandle, rhs: &GpuTensorHandle) -> bool {
    lhs.device_id == rhs.device_id && lhs.buffer_id == rhs.buffer_id
}

fn free_rejected_hypot_output(output: &GpuTensorHandle, inputs: &[&GpuTensorHandle]) {
    if inputs
        .iter()
        .any(|input| hypot_gpu_handles_alias(output, input))
    {
        return;
    }
    if let Some(owner) = runmat_accelerate_api::provider_for_handle(output) {
        if owner.free(output).is_ok() {
            runmat_accelerate_api::clear_residency(output);
        }
    }
}

fn hypot_host(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    if let (Some(left), Some(right)) = (scalar_hypot_value(&lhs), scalar_hypot_value(&rhs)) {
        return Ok(Value::Num(matlab_hypot_f64(left, right)));
    }
    let tensor_a = value_into_hypot_tensor(lhs)?;
    let tensor_b = value_into_hypot_tensor(rhs)?;
    compute_hypot_tensor(tensor_a, tensor_b)
}

fn compute_hypot_tensor(a: Tensor, b: Tensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&a.shape, &b.shape)
        .map_err(|err| hypot_error_with_detail(&HYPOT_ERROR_SIZE_MISMATCH, err))?;
    let output_shape = plan.output_shape().to_vec();
    let a_storage = a
        .into_numeric_storage()
        .map_err(|e| hypot_error_with_detail(&HYPOT_ERROR_INTERNAL, e))?;
    let b_storage = b
        .into_numeric_storage()
        .map_err(|e| hypot_error_with_detail(&HYPOT_ERROR_INTERNAL, e))?;
    let use_single =
        matches!(a_storage, NumericStorage::F32(_)) && matches!(b_storage, NumericStorage::F32(_));
    let storage = if use_single {
        let a_values = promote_hypot_operand_to_single_domain(a_storage);
        let b_values = promote_hypot_operand_to_single_domain(b_storage);
        NumericStorage::F32(
            plan.iter()
                .map(|(_, idx_a, idx_b)| matlab_hypot_f32(a_values[idx_a], b_values[idx_b]))
                .collect(),
        )
    } else {
        let a_values = promote_hypot_operand_to_double_domain(a_storage);
        let b_values = promote_hypot_operand_to_double_domain(b_storage);
        NumericStorage::F64(
            plan.iter()
                .map(|(_, idx_a, idx_b)| matlab_hypot_f64(a_values[idx_a], b_values[idx_b]))
                .collect(),
        )
    };
    let tensor = Tensor::from_numeric_storage(storage, output_shape)
        .map_err(|e| hypot_error_with_detail(&HYPOT_ERROR_INTERNAL, e))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn promote_hypot_operand_to_single_domain(storage: NumericStorage) -> Vec<f32> {
    storage.materialize_f32()
}

fn promote_hypot_operand_to_double_domain(storage: NumericStorage) -> Vec<f64> {
    match storage {
        NumericStorage::F64(values) => values,
        NumericStorage::F32(values) => values.into_iter().map(f64::from).collect(),
        storage => storage
            .into_integer_storage()
            .expect("hypot double-domain promotion received unsupported storage")
            .to_f64_vec(),
    }
}

fn value_into_hypot_tensor(value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::CharArray(ca) => {
            let data: Vec<f64> = ca.data.iter().map(|&ch| ch as u32 as f64).collect();
            Tensor::new(data, vec![ca.rows, ca.cols])
                .map_err(|e| hypot_error_with_detail(&HYPOT_ERROR_INTERNAL, e))
        }
        Value::Complex(re, im) => Tensor::new(vec![complex_magnitude(re, im)], vec![1, 1])
            .map_err(|e| hypot_error_with_detail(&HYPOT_ERROR_INTERNAL, e)),
        Value::ComplexTensor(ct) => {
            let shape = ct.shape.clone();
            let storage = match ct.into_complex_storage() {
                ComplexStorage::F64(values) => NumericStorage::F64(
                    values
                        .into_iter()
                        .map(|(real, imag)| matlab_hypot_f64(real, imag))
                        .collect(),
                ),
                ComplexStorage::F32(values) => NumericStorage::F32(
                    values
                        .into_iter()
                        .map(|(real, imag)| matlab_hypot_f32(real, imag))
                        .collect(),
                ),
                ComplexStorage::Integer(_) => {
                    return Err(hypot_error_with_detail(
                        &HYPOT_ERROR_INVALID_INPUT,
                        "typed complex integer input is not supported",
                    ))
                }
            };
            Tensor::from_numeric_storage(storage, shape)
                .map_err(|e| hypot_error_with_detail(&HYPOT_ERROR_INTERNAL, e))
        }
        other => {
            if let Value::GpuTensor(_) = other {
                return Err(hypot_error_with_detail(
                    &HYPOT_ERROR_INTERNAL,
                    "internal error converting GPU tensor",
                ));
            }
            tensor::value_into_tensor_for("hypot", other)
                .map_err(|e| hypot_error_with_detail(&HYPOT_ERROR_INVALID_INPUT, e))
        }
    }
}

fn complex_magnitude(re: f64, im: f64) -> f64 {
    matlab_hypot_f64(re, im)
}

fn matlab_hypot_f64(lhs: f64, rhs: f64) -> f64 {
    if lhs.is_nan() || rhs.is_nan() {
        f64::NAN
    } else {
        lhs.hypot(rhs)
    }
}

fn matlab_hypot_f32(lhs: f32, rhs: f32) -> f32 {
    if lhs.is_nan() || rhs.is_nan() {
        f32::NAN
    } else {
        lhs.hypot(rhs)
    }
}

fn scalar_hypot_value(value: &Value) -> Option<f64> {
    match value {
        Value::Num(n) => Some(*n),
        Value::Int(i) => Some(i.to_f64()),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::LogicalArray(l) if l.data.len() == 1 => Some(if l.data[0] != 0 { 1.0 } else { 0.0 }),
        Value::CharArray(ca) if ca.rows * ca.cols == 1 => {
            Some(ca.data.first().map(|&ch| ch as u32 as f64).unwrap_or(0.0))
        }
        Value::Complex(re, im) => Some(complex_magnitude(*re, *im)),
        _ => None,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::AccelProvider;
    use runmat_builtins::{ResolveContext, Type};
    use runmat_value::{
        CharArray, ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, LogicalArray,
        Tensor, Value,
    };

    fn hypot_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
        block_on(super::hypot_builtin(lhs, rhs))
    }

    #[test]
    fn scalar_hypot_value_leaves_typed_integer_tensor_for_storage_dispatch() {
        let tensor =
            Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
                .expect("integer tensor");

        assert_eq!(scalar_hypot_value(&Value::Tensor(tensor)), None);
    }

    #[test]
    fn scalar_hypot_value_leaves_complex_tensor_for_storage_dispatch() {
        let storage =
            IntegerComplexStorage::new(IntegerStorage::I16(vec![3]), IntegerStorage::I16(vec![4]))
                .expect("complex integer storage");
        let tensor = ComplexTensor::new_integer(storage, vec![1, 1]).expect("complex tensor");

        assert_eq!(scalar_hypot_value(&Value::ComplexTensor(tensor)), None);
    }

    #[test]
    fn hypot_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = HYPOT_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"R = hypot(X, Y)"));
    }

    #[test]
    fn hypot_type_preserves_tensor_shape() {
        let out = numeric_binary_type(
            &[
                Type::Tensor {
                    shape: Some(vec![Some(2), Some(3)]),
                },
                Type::Tensor {
                    shape: Some(vec![Some(2), Some(3)]),
                },
            ],
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
    fn hypot_type_scalar_returns_num() {
        let out = numeric_binary_type(&[Type::Num, Type::Int], &ResolveContext::new(Vec::new()));
        assert_eq!(out, Type::Num);
    }

    #[test]
    fn hypot_rejects_typed_complex_integer_inputs() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let complex = ComplexTensor::new_integer(
            IntegerComplexStorage::new(IntegerStorage::I64(vec![1]), IntegerStorage::I64(vec![-2]))
                .expect("storage"),
            vec![1, 1],
        )
        .expect("tensor");

        let left = hypot_builtin(Value::ComplexTensor(complex.clone()), Value::Num(1.0))
            .expect_err("typed complex integer input must reject");
        assert!(left
            .message()
            .contains("complex numbers with integer types"));

        let right = hypot_builtin(Value::Num(1.0), Value::ComplexTensor(complex))
            .expect_err("typed complex integer input must reject");
        assert!(right
            .message()
            .contains("complex numbers with integer types"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hypot_scalar_pair() {
        let result = hypot_builtin(Value::Num(3.0), Value::Num(4.0)).expect("hypot");
        match result {
            Value::Num(v) => assert!((v - 5.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hypot_matrix_elements() {
        let lhs = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let rhs = Tensor::new(vec![0.0, 0.0, 1.0, 1.0], vec![2, 2]).unwrap();
        let result =
            hypot_builtin(Value::Tensor(lhs), Value::Tensor(rhs)).expect("element-wise hypot");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = [1.0, 3.0, (5.0f64).sqrt(), (17.0f64).sqrt()];
                for (actual, expect) in t.materialize_f64().iter().zip(expected.iter()) {
                    assert!((actual - expect).abs() < 1e-12, "{actual} vs {expect}");
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hypot_scalar_broadcast() {
        let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = hypot_builtin(Value::Tensor(matrix), Value::Num(4.0)).expect("broadcast");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = [4.123105625617661, 4.47213595499958, 5.0, 5.656854249492381];
                for (actual, expect) in t.materialize_f64().iter().zip(expected.iter()) {
                    assert!((actual - expect).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hypot_row_vector_broadcasts_over_matrix() {
        let matrix = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let row = Tensor::new(vec![3.0, 4.0, 5.0], vec![1, 3]).unwrap();
        let result =
            hypot_builtin(Value::Tensor(matrix), Value::Tensor(row)).expect("row broadcast");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 3]);
                let expected = [
                    (1.0f64).hypot(3.0),
                    (4.0f64).hypot(3.0),
                    (2.0f64).hypot(4.0),
                    (5.0f64).hypot(4.0),
                    (3.0f64).hypot(5.0),
                    (6.0f64).hypot(5.0),
                ];
                for (actual, expect) in t.materialize_f64().iter().zip(expected.iter()) {
                    assert!((actual - expect).abs() < 1e-12, "{actual} vs {expect}");
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hypot_typed_integer_tensor_broadcast_reads_integer_storage() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let matrix =
            Tensor::new_integer(IntegerStorage::I16(vec![3, 4, 5, 12]), vec![2, 2]).unwrap();
        let row = Tensor::new_integer(IntegerStorage::I16(vec![4, 3]), vec![1, 2]).unwrap();

        let result =
            hypot_builtin(Value::Tensor(matrix), Value::Tensor(row)).expect("integer hypot");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = [
                    3.0_f64.hypot(4.0),
                    4.0_f64.hypot(4.0),
                    5.0_f64.hypot(3.0),
                    12.0_f64.hypot(3.0),
                ];
                for (actual, expect) in t.materialize_f64().iter().zip(expected.iter()) {
                    assert!((actual - expect).abs() < 1e-12, "{actual} vs {expect}");
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn hypot_preserves_native_single_real_complex_mixed_and_empty_storage() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let left = Tensor::from_f32(vec![3.0, 5.0], vec![2, 1]).unwrap();
        let right = Tensor::from_f32(vec![4.0, 12.0], vec![2, 1]).unwrap();
        let Value::Tensor(output) =
            hypot_builtin(Value::Tensor(left), Value::Tensor(right)).expect("single hypot")
        else {
            panic!("expected single tensor");
        };
        assert_eq!(
            output.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![5.0, 13.0])
        );

        let left = Tensor::new(vec![3.0, 5.0], vec![2, 1]).unwrap();
        let right = Tensor::from_f32(vec![4.0, 12.0], vec![2, 1]).unwrap();
        let Value::Tensor(output) =
            hypot_builtin(Value::Tensor(left), Value::Tensor(right)).expect("mixed hypot")
        else {
            panic!("expected mixed result to use double");
        };
        assert_eq!(
            output.into_numeric_storage().unwrap(),
            NumericStorage::F64(vec![5.0, 13.0])
        );

        let left = Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_992, 3]),
            vec![1, 2],
        )
        .unwrap();
        let right = Tensor::from_f32(vec![1.0, 4.0], vec![1, 2]).unwrap();
        let Value::Tensor(output) =
            hypot_builtin(Value::Tensor(left), Value::Tensor(right)).expect("integer/single hypot")
        else {
            panic!("expected integer/single result to use double");
        };
        assert_eq!(output.numeric_dtype(), runmat_value::NumericDType::F64);
        assert_eq!(
            output.materialize_f64(),
            vec![(9_007_199_254_740_992_f64).hypot(1.0), 5.0]
        );

        let complex = ComplexTensor::from_f32(vec![(3.0, 4.0)], vec![1, 1]).unwrap();
        let real = Tensor::from_f32(vec![12.0], vec![1, 1]).unwrap();
        let Value::Tensor(output) =
            hypot_builtin(Value::ComplexTensor(complex), Value::Tensor(real))
                .expect("complex single hypot")
        else {
            panic!("one-element single result must retain tensor class");
        };
        assert_eq!(
            output.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![13.0])
        );

        let left = Tensor::from_f32(Vec::new(), vec![0, 3]).unwrap();
        let right = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let Value::Tensor(output) =
            hypot_builtin(Value::Tensor(left), Value::Tensor(right)).expect("empty hypot")
        else {
            panic!("expected empty mixed tensor");
        };
        assert_eq!(output.shape, vec![0, 3]);
        assert_eq!(
            output.into_numeric_storage().unwrap(),
            NumericStorage::F64(Vec::new())
        );
    }

    #[test]
    fn hypot_integer_gpu_pair_gathers_exact_storage_before_floating_provider_hook() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let wide = 9_007_199_254_740_992_u64;
            let left = Tensor::new_integer(IntegerStorage::U64(vec![wide, 3]), vec![1, 2]).unwrap();
            let right = Tensor::new_integer(IntegerStorage::U64(vec![1, 4]), vec![1, 2]).unwrap();
            let left = gpu_helpers::upload_tensor(provider, &left).expect("upload left");
            let right = gpu_helpers::upload_tensor(provider, &right).expect("upload right");
            let left_type = runmat_accelerate_api::handle_integer_type(&left);
            let right_type = runmat_accelerate_api::handle_integer_type(&right);
            let output = hypot_builtin(
                Value::GpuTensor(left.clone()),
                Value::GpuTensor(right.clone()),
            )
            .expect("integer gpu hypot");
            assert!(runmat_accelerate_api::provider_for_handle(&left).is_some());
            assert!(runmat_accelerate_api::provider_for_handle(&right).is_some());
            assert_eq!(runmat_accelerate_api::handle_integer_type(&left), left_type);
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&right),
                right_type
            );
            let output = test_support::gather(output).expect("gather restored hypot");
            assert_eq!(
                output.into_numeric_storage().unwrap(),
                NumericStorage::F64(vec![(wide as f64).hypot(1.0), 5.0])
            );
        });
    }

    #[test]
    fn hypot_rejects_inexact_wide_integer_extension_values() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let value = Value::Int(IntValue::U64(9_007_199_254_740_993));
        let err = hypot_builtin(value, Value::Num(1.0)).expect_err("inexact integer rejects");
        assert_eq!(err.identifier(), Some("RunMat:hypot:InvalidInput"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hypot_complex_scalars() {
        let left = (3.0, 4.0);
        let right = (-1.0, 2.0);
        let result = hypot_builtin(
            Value::Complex(left.0, left.1),
            Value::Complex(right.0, right.1),
        )
        .expect("complex hypot");
        let expected = complex_magnitude(left.0, left.1).hypot(complex_magnitude(right.0, right.1));
        match result {
            Value::Num(v) => assert!((v - expected).abs() < 1e-12),
            other => panic!("expected scalar norm, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hypot_complex_tensor_with_real() {
        let complex = ComplexTensor::new(vec![(3.0, 4.0), (5.0, 12.0)], vec![2, 1]).unwrap();
        let real = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
        let result =
            hypot_builtin(Value::ComplexTensor(complex), Value::Tensor(real)).expect("mixed");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                let expected = [
                    complex_magnitude(3.0, 4.0).hypot(0.0),
                    complex_magnitude(5.0, 12.0).hypot(1.0),
                ];
                for (actual, expect) in t.materialize_f64().iter().zip(expected.iter()) {
                    assert!((actual - expect).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hypot_char_array_inputs() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let chars = CharArray::new("AB".chars().collect(), 1, 2).unwrap();
        let result = hypot_builtin(Value::CharArray(chars), Value::Int(IntValue::I32(1)))
            .expect("char hypot");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                let expected = [
                    (65.0f64.powi(2) + 1.0).sqrt(),
                    (66.0f64.powi(2) + 1.0).sqrt(),
                ];
                for (actual, expect) in t.materialize_f64().iter().zip(expected.iter()) {
                    assert!((actual - expect).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hypot_logical_inputs() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let logical = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).expect("logical array");
        let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
        let result =
            hypot_builtin(Value::LogicalArray(logical), Value::Tensor(tensor)).expect("logical");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                let expected = [
                    1.0_f64.hypot(0.0),
                    0.0_f64.hypot(1.0),
                    0.0_f64.hypot(2.0),
                    1.0_f64.hypot(3.0),
                ];
                for (actual, expect) in out.materialize_f64().iter().zip(expected.iter()) {
                    assert!((actual - expect).abs() < 1e-12, "{actual} vs {expect}");
                }
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hypot_string_input_has_stable_identifier() {
        let err = hypot_builtin(Value::from("bad"), Value::Num(1.0)).expect_err("expected error");
        assert_eq!(err.identifier(), HYPOT_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hypot_dimension_mismatch_errors() {
        let lhs = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![2, 2]).unwrap();
        let rhs = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let err = hypot_builtin(Value::Tensor(lhs), Value::Tensor(rhs)).unwrap_err();
        assert!(
            err.message().contains("dimension"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hypot_nan_propagates() {
        let result = hypot_builtin(Value::Num(f64::NAN), Value::Num(1.0)).expect("nan propagation");
        match result {
            Value::Num(v) => assert!(v.is_nan()),
            other => panic!("expected NaN scalar, got {other:?}"),
        }
    }

    #[test]
    fn hypot_nan_takes_precedence_over_infinity() {
        for (lhs, rhs) in [
            (f64::NAN, f64::INFINITY),
            (f64::INFINITY, f64::NAN),
            (f64::NAN, f64::NEG_INFINITY),
            (f64::NEG_INFINITY, f64::NAN),
        ] {
            let result = hypot_builtin(Value::Num(lhs), Value::Num(rhs)).unwrap();
            assert!(matches!(result, Value::Num(value) if value.is_nan()));
        }
    }

    #[test]
    fn hypot_runmat_extensions_follow_compatibility_mode() {
        for (value, identifier) in [
            (
                Value::Int(IntValue::I32(3)),
                "RunMat:compatibility:HypotIntegerInputExtension",
            ),
            (
                Value::Bool(true),
                "RunMat:compatibility:HypotLogicalInputExtension",
            ),
            (
                Value::CharArray(CharArray::new(vec!['A'], 1, 1).unwrap()),
                "RunMat:compatibility:HypotCharacterInputExtension",
            ),
        ] {
            let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let err =
                hypot_builtin(value, Value::Num(4.0)).expect_err("strict mode rejects extension");
            assert_eq!(err.identifier(), Some(identifier));
            assert_eq!(err.gpu_gather_retry(), crate::GpuGatherRetry::Never);
        }
    }

    /// IEEE 754 / MATLAB require hypot(Inf, Inf) = Inf.
    /// The scaling form lo/hi = Inf/Inf = NaN, so the host path must not regress.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hypot_both_infinite_is_inf() {
        let result =
            hypot_builtin(Value::Num(f64::INFINITY), Value::Num(f64::INFINITY)).expect("hypot inf");
        match result {
            Value::Num(v) => assert!(v.is_infinite() && v > 0.0, "expected +Inf, got {v}"),
            other => panic!("expected +Inf scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hypot_one_infinite_is_inf() {
        let result =
            hypot_builtin(Value::Num(f64::INFINITY), Value::Num(3.0)).expect("hypot inf/finite");
        match result {
            Value::Num(v) => assert!(v.is_infinite() && v > 0.0, "expected +Inf, got {v}"),
            other => panic!("expected +Inf scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hypot_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let lhs = Tensor::new(vec![3.0, 5.0, 8.0, 7.0], vec![2, 2]).unwrap();
            let rhs = Tensor::new(vec![4.0, 12.0, 15.0, 24.0], vec![2, 2]).unwrap();
            let h_lhs = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &lhs.materialize_f64(),
                    shape: &lhs.shape,
                })
                .expect("upload lhs");
            let h_rhs = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &rhs.materialize_f64(),
                    shape: &rhs.shape,
                })
                .expect("upload rhs");
            let result =
                hypot_builtin(Value::GpuTensor(h_lhs), Value::GpuTensor(h_rhs)).expect("gpu hypot");
            let gathered = test_support::gather(result).expect("gathered result");
            let expected = [5.0, 13.0, 17.0, 25.0];
            assert_eq!(gathered.shape, vec![2, 2]);
            for (actual, expect) in gathered.materialize_f64().iter().zip(expected.iter()) {
                assert!((actual - expect).abs() < 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hypot_gpu_and_host_mix_falls_back() {
        test_support::with_test_provider(|provider| {
            let lhs = Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap();
            let handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &lhs.materialize_f64(),
                    shape: &lhs.shape,
                })
                .expect("upload");
            let result =
                hypot_builtin(Value::GpuTensor(handle), Value::Num(4.0)).expect("gpu + host hypot");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 1]);
            let expected: Vec<f64> = lhs
                .materialize_f64()
                .iter()
                .map(|&x| x.hypot(4.0))
                .collect();
            for (actual, expect) in gathered.materialize_f64().iter().zip(expected.iter()) {
                assert!((actual - expect).abs() < 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hypot_gpu_left_host_integer_right_fallback_reads_integer_storage() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let lhs = Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap();
            let handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &lhs.materialize_f64(),
                    shape: &lhs.shape,
                })
                .expect("upload");
            let rhs = Tensor::new_integer(IntegerStorage::I16(vec![4, 3]), vec![2, 1]).unwrap();

            let precision = runmat_accelerate_api::handle_precision(&handle);
            let result = hypot_builtin(Value::GpuTensor(handle.clone()), Value::Tensor(rhs))
                .expect("gpu + integer host hypot");
            assert!(runmat_accelerate_api::provider_for_handle(&handle).is_some());
            assert_eq!(runmat_accelerate_api::handle_precision(&handle), precision);
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 1]);
            let expected = [5.0, 5.0];
            for (actual, expect) in gathered.materialize_f64().iter().zip(expected.iter()) {
                assert!((actual - expect).abs() < 1e-12, "{actual} vs {expect}");
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hypot_host_integer_left_gpu_right_fallback_reads_integer_storage() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let rhs = Tensor::new(vec![4.0, 3.0], vec![2, 1]).unwrap();
            let handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &rhs.materialize_f64(),
                    shape: &rhs.shape,
                })
                .expect("upload");
            let lhs = Tensor::new_integer(IntegerStorage::I16(vec![3, 4]), vec![2, 1]).unwrap();

            let result = hypot_builtin(Value::Tensor(lhs), Value::GpuTensor(handle))
                .expect("integer host + gpu hypot");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 1]);
            let expected = [5.0, 5.0];
            for (actual, expect) in gathered.materialize_f64().iter().zip(expected.iter()) {
                assert!((actual - expect).abs() < 1e-12, "{actual} vs {expect}");
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hypot_empty_tensor_result() {
        let lhs = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let rhs = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let result =
            hypot_builtin(Value::Tensor(lhs), Value::Tensor(rhs)).expect("empty hypot result");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![0, 3]);
                assert!(out.materialize_f64().is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn hypot_wgpu_matches_cpu_elementwise() {
        let _guard = test_support::accel_test_lock();
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        let lhs = Tensor::new(vec![3.0, 4.0, 5.0, 12.0], vec![2, 2]).unwrap();
        let rhs = Tensor::new(vec![4.0, 3.0, 12.0, 5.0], vec![2, 2]).unwrap();

        let cpu_value = compute_hypot_tensor(lhs.clone(), rhs.clone()).expect("cpu hypot");
        let expected = test_support::gather(cpu_value).expect("gather cpu result");

        let h_lhs = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &lhs.materialize_f64(),
                shape: &lhs.shape,
            })
            .expect("upload lhs");
        let h_rhs = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &rhs.materialize_f64(),
                shape: &rhs.shape,
            })
            .expect("upload rhs");

        let gpu_value =
            hypot_builtin(Value::GpuTensor(h_lhs), Value::GpuTensor(h_rhs)).expect("gpu hypot");
        let gathered = test_support::gather(gpu_value).expect("gather gpu result");

        assert_eq!(gathered.shape, expected.shape);
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (actual, expect) in gathered
            .materialize_f64()
            .iter()
            .zip(expected.materialize_f64().iter())
        {
            assert!(
                (actual - expect).abs() < tol,
                "|{actual} - {expect}| >= {tol}"
            );
        }
    }
}
