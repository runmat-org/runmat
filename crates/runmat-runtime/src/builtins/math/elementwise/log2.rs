//! MATLAB-compatible base-2 logarithm (`log2`) builtin with GPU-aware semantics for RunMat.
//!
//! Provides element-wise base-2 logarithms for real, logical, character, and complex inputs while
//! preserving MATLAB semantics. Negative real values promote to complex outputs and GPU execution
//! falls back to the host whenever complex numbers are required or the provider lacks a dedicated
//! kernel.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, ComplexStorage, ComplexTensor, NumericStorage, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use super::log::{detect_gpu_requires_complex, log_complex_parts, log_complex_parts_f32};
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const IMAG_EPS: f64 = 1e-12;
const LOG2_E: f64 = std::f64::consts::LOG2_E;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::log2")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "log2",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_log2" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may execute log2 directly on device buffers; runtimes fall back to the host when complex outputs are required or the hook is unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::log2")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "log2",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx
                .inputs
                .first()
                .ok_or(FusionError::MissingInput(0))?;
            Ok(format!("log2({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL `log2` calls; providers can override with fused kernels when available.",
};

const BUILTIN_NAME: &str = "log2";

const LOG2_INTEGER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "log2-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "log2 with integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Log2IntegerInputExtension"),
};
const LOG2_LOGICAL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "log2-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "log2 with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Log2LogicalInputExtension"),
};
const LOG2_CHARACTER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "log2-character-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "log2 with character input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Log2CharacterInputExtension"),
};
pub const LOG2_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    LOG2_INTEGER_EXTENSION,
    LOG2_LOGICAL_EXTENSION,
    LOG2_CHARACTER_EXTENSION,
];
const LOG2_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "RunMat mode promotes all eight integer classes to the binary64 logarithm domain.",
}];
pub const LOG2_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "Y = log2(integer_X)", inputs: &LOG2_INTEGER_INPUTS,
    computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
    output_class: BuiltinIntegerOutputClassRule::Double,
    overflow: BuiltinIntegerOverflowRule::NotApplicable,
    backend: BuiltinIntegerBackendRule::GatherFallback,
    overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
    notes: "This is a gated RunMat extension; resident integer inputs gather exactly from their owning provider.",
}];

const LOG2_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elementwise base-2 logarithm result.",
}];
const LOG2_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric, logical, char, or complex input.",
}];
const LOG2_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = log2(X)",
    inputs: &LOG2_INPUTS,
    outputs: &LOG2_OUTPUT,
}];
const LOG2_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOG2.INVALID_INPUT",
    identifier: Some("RunMat:log2:InvalidInput"),
    when: "Input cannot be interpreted as numeric, logical, char, or complex data.",
    message: "log2: invalid input",
};
const LOG2_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOG2.INTERNAL",
    identifier: Some("RunMat:log2:Internal"),
    when: "Internal tensor construction or provider interaction failed.",
    message: "log2: internal error",
};
const LOG2_ERRORS: [BuiltinErrorDescriptor; 2] = [LOG2_ERROR_INVALID_INPUT, LOG2_ERROR_INTERNAL];
pub const LOG2_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LOG2_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LOG2_ERRORS,
};

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn log2_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "log2",
    category = "math/elementwise",
    summary = "Base-2 logarithm of scalars, vectors, matrices, or N-D tensors.",
    keywords = "log2,base-2 logarithm,elementwise,gpu,complex",
    accel = "unary",
    type_resolver(numeric_unary_type),
    descriptor(crate::builtins::math::elementwise::log2::LOG2_DESCRIPTOR),
    extensions(crate::builtins::math::elementwise::log2::LOG2_EXTENSIONS),
    integer_capabilities(crate::builtins::math::elementwise::log2::LOG2_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::elementwise::log2"
)]
async fn log2_builtin(value: Value) -> BuiltinResult<Value> {
    ensure_log2_extensions(&value).await?;
    match value {
        Value::GpuTensor(handle) => log2_gpu(handle).await,
        Value::Complex(re, im) => {
            let (r, i) = log2_complex_parts(re, im);
            Ok(Value::Complex(r, i))
        }
        Value::ComplexTensor(ct) => {
            crate::builtins::common::validation::reject_typed_complex_integer_tensor(&ct, "log2")?;
            log2_complex_tensor(ct)
        }
        Value::CharArray(ca) => log2_char_array(ca),
        Value::String(_) | Value::StringArray(_) => Err(log2_error_with_detail(
            &LOG2_ERROR_INVALID_INPUT,
            "expected numeric input",
        )),
        other => log2_real(other),
    }
}

async fn ensure_log2_extensions(value: &Value) -> BuiltinResult<()> {
    let extension = match value {
        Value::Int(_) => Some(&LOG2_INTEGER_EXTENSION),
        Value::Tensor(t) if t.integer_storage().is_some() => Some(&LOG2_INTEGER_EXTENSION),
        Value::GpuTensor(h) if runmat_accelerate_api::handle_integer_type(h).is_some() => {
            Some(&LOG2_INTEGER_EXTENSION)
        }
        Value::Bool(_) | Value::LogicalArray(_) => Some(&LOG2_LOGICAL_EXTENSION),
        Value::GpuTensor(h) if runmat_accelerate_api::handle_is_logical(h) => {
            Some(&LOG2_LOGICAL_EXTENSION)
        }
        Value::CharArray(_) => Some(&LOG2_CHARACTER_EXTENSION),
        _ => None,
    };
    if let Some(extension) = extension {
        crate::compatibility::ensure_builtin_extension_enabled(extension, BUILTIN_NAME)?;
    }
    if crate::builtins::common::validation::value_has_native_integer_class(value)
        && !crate::builtins::common::validation::native_integer_value_is_exact_f64_async(value)
            .await?
    {
        return Err(log2_error_with_detail(
            &LOG2_ERROR_INVALID_INPUT,
            "integer input lies outside the exact binary64 interval",
        ));
    }
    Ok(())
}

async fn log2_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if runmat_accelerate_api::handle_integer_type(&handle).is_some() {
        let owner = gpu_helpers::exact_provider_for_handle(&handle);
        let tensor = gpu_helpers::gather_tensor_async(&handle)
            .await
            .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
        let result = log2_tensor(tensor)?;
        return restore_explicit_log2_result(result, &handle, owner);
    }
    let owner = gpu_helpers::exact_provider_for_handle(&handle).ok_or_else(|| {
        build_runtime_error("log2: no exact owner for GPU input")
            .with_builtin(BUILTIN_NAME)
            .with_identifier("RunMat:gpu:ProviderOwnershipMismatch")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build()
    })?;
    {
        let provider = owner;
        match detect_gpu_requires_complex(provider, &handle).await {
            Ok(false) => {
                if let Ok(out) = provider.unary_log2(&handle).await {
                    if valid_log2_output(&out, &handle, provider) {
                        runmat_accelerate_api::set_handle_provenance(
                            &out,
                            runmat_accelerate_api::handle_provenance(&handle)
                                .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic),
                        );
                        return Ok(gpu_helpers::resident_gpu_value(out));
                    }
                    gpu_helpers::free_unprotected_exact_owner(&out, &[&handle]);
                }
            }
            Ok(true) => {
                if runmat_accelerate_api::handle_is_explicit(&handle) {
                    return Err(build_runtime_error(
                        "log2: real gpuArray input must be explicitly complex when the result can be complex",
                    ).with_builtin(BUILTIN_NAME)
                    .with_identifier("RunMat:log2:GpuComplexInputRequired")
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never).build());
                }
                let tensor = gpu_helpers::gather_tensor_async(&handle)
                    .await
                    .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
                return log2_tensor(tensor);
            }
            Err(err) => {
                if err.message() == "interaction pending..." {
                    return Err(err);
                }
                // Fall through to host fallback below.
            }
        }
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    let result = log2_tensor(tensor)?;
    restore_explicit_log2_result(result, &handle, Some(owner))
}

fn restore_explicit_log2_result(
    result: Value,
    input: &GpuTensorHandle,
    owner: Option<&'static dyn runmat_accelerate_api::AccelProvider>,
) -> BuiltinResult<Value> {
    if !runmat_accelerate_api::handle_is_explicit(input) {
        return Ok(result);
    }
    let owner = owner.ok_or_else(|| {
        build_runtime_error("log2: no exact owner for explicit gpuArray input")
            .with_builtin(BUILTIN_NAME)
            .with_identifier("RunMat:gpu:ProviderOwnershipMismatch")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build()
    })?;
    let output = match &result {
        Value::Tensor(tensor) => gpu_helpers::upload_tensor(owner, tensor).map_err(|error| {
            builtin_error(format!("log2: failed to restore GPU result: {error}"))
        })?,
        Value::ComplexTensor(tensor) => gpu_helpers::upload_complex_tensor(owner, tensor)?,
        _ => return Ok(result),
    };
    runmat_accelerate_api::mark_handle_explicit(&output);
    Ok(gpu_helpers::resident_gpu_value(output))
}

fn valid_log2_output(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    owner: &'static dyn runmat_accelerate_api::AccelProvider,
) -> bool {
    output.shape == input.shape
        && output.device_id == input.device_id
        && !gpu_helpers::same_gpu_handle(output, input)
        && runmat_accelerate_api::handle_storage(output)
            == runmat_accelerate_api::GpuTensorStorage::Real
        && runmat_accelerate_api::handle_precision(output)
            == runmat_accelerate_api::handle_precision(input)
        && runmat_accelerate_api::handle_integer_type(output).is_none()
        && !runmat_accelerate_api::handle_is_logical(output)
        && gpu_helpers::exact_provider_for_handle(output)
            .is_some_and(|candidate| std::ptr::eq(candidate, owner))
}

fn log2_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("log2", value)
        .map_err(|e| builtin_error(format!("log2: {e}")))?;
    log2_tensor(tensor)
}

fn log2_tensor(tensor: Tensor) -> BuiltinResult<Value> {
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|e| builtin_error(format!("log2: {e}")))?;
    match storage {
        NumericStorage::F64(values) => log2_real_f64_values(values, shape),
        NumericStorage::F32(values) => log2_real_f32_values(values, shape),
        storage => log2_real_f64_values(promote_integer_storage_to_log2_domain(storage), shape),
    }
}

fn log2_real_f64_values(values: Vec<f64>, shape: Vec<usize>) -> BuiltinResult<Value> {
    let mut complex_values = Vec::with_capacity(values.len());
    let mut has_imag = false;
    for v in values {
        let (re_part, im_part) = log2_complex_parts(v, 0.0);
        if im_part != 0.0 {
            has_imag = true;
        }
        complex_values.push((re_part, im_part));
    }
    if has_imag {
        let tensor =
            ComplexTensor::from_complex_storage(ComplexStorage::F64(complex_values), shape)
                .map_err(|e| builtin_error(format!("log2: {e}")))?;
        Ok(complex_tensor_into_value(tensor))
    } else {
        let data: Vec<f64> = complex_values
            .into_iter()
            .map(|(mut re, _)| {
                if re.is_finite() && re.abs() < IMAG_EPS {
                    re = 0.0;
                }
                re
            })
            .collect();
        let tensor = Tensor::from_numeric_storage(NumericStorage::F64(data), shape)
            .map_err(|e| builtin_error(format!("log2: {e}")))?;
        Ok(tensor::tensor_into_value(tensor))
    }
}

fn log2_real_f32_values(values: Vec<f32>, shape: Vec<usize>) -> BuiltinResult<Value> {
    let mut complex_values = Vec::with_capacity(values.len());
    let mut has_imag = false;
    for value in values {
        let (real, imag) = log2_complex_parts_f32(value, 0.0);
        has_imag |= imag != 0.0;
        complex_values.push((real, imag));
    }
    if has_imag {
        let tensor =
            ComplexTensor::from_complex_storage(ComplexStorage::F32(complex_values), shape)
                .map_err(|e| builtin_error(format!("log2: {e}")))?;
        Ok(complex_tensor_into_value(tensor))
    } else {
        let values = complex_values
            .into_iter()
            .map(|(mut real, _)| {
                if real.is_finite() && real.abs() < IMAG_EPS as f32 {
                    real = 0.0;
                }
                real
            })
            .collect();
        let tensor = Tensor::from_numeric_storage(NumericStorage::F32(values), shape)
            .map_err(|e| builtin_error(format!("log2: {e}")))?;
        Ok(tensor::tensor_into_value(tensor))
    }
}

fn log2_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let shape = ct.shape.clone();
    let storage = match ct.into_complex_storage() {
        ComplexStorage::F64(values) => ComplexStorage::F64(
            values
                .into_iter()
                .map(|(real, imag)| log2_complex_parts(real, imag))
                .collect(),
        ),
        ComplexStorage::F32(values) => ComplexStorage::F32(
            values
                .into_iter()
                .map(|(real, imag)| log2_complex_parts_f32(real, imag))
                .collect(),
        ),
        ComplexStorage::Integer(_) => {
            return Err(log2_error_with_detail(
                &LOG2_ERROR_INVALID_INPUT,
                "typed complex integer input is not supported",
            ))
        }
    };
    let tensor = ComplexTensor::from_complex_storage(storage, shape)
        .map_err(|e| builtin_error(format!("log2: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

fn promote_integer_storage_to_log2_domain(storage: NumericStorage) -> Vec<f64> {
    storage
        .into_integer_storage()
        .expect("log2 integer-promotion boundary received floating storage")
        .to_f64_vec()
}

fn log2_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data: Vec<f64> = ca.data.iter().map(|&ch| ch as u32 as f64).collect();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| builtin_error(format!("log2: {e}")))?;
    log2_tensor(tensor)
}

fn log2_complex_parts(re: f64, im: f64) -> (f64, f64) {
    let (real_ln, imag_ln) = log_complex_parts(re, im);
    let mut real_part = real_ln * LOG2_E;
    let mut imag_part = imag_ln * LOG2_E;

    if real_part.is_finite() && real_part.abs() < IMAG_EPS {
        real_part = 0.0;
    }
    if imag_part.abs() < IMAG_EPS {
        imag_part = 0.0;
    }

    (real_part, imag_part)
}

fn log2_complex_parts_f32(re: f32, im: f32) -> (f32, f32) {
    let (real_ln, imag_ln) = log_complex_parts_f32(re, im);
    let mut real_part = real_ln * std::f32::consts::LOG2_E;
    let mut imag_part = imag_ln * std::f32::consts::LOG2_E;
    if real_part.is_finite() && real_part.abs() < IMAG_EPS as f32 {
        real_part = 0.0;
    }
    if imag_part.abs() < IMAG_EPS as f32 {
        imag_part = 0.0;
    }
    (real_part, imag_part)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::AccelProvider;
    use runmat_builtins::{
        IntegerStorage, LogicalArray, ResolveContext, StringArray, Tensor, Type, Value,
    };

    fn log2_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::log2_builtin(value))
    }

    #[test]
    fn log2_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = LOG2_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = log2(X)"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log2_reads_typed_integer_tensor_storage_exactly() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new_integer(IntegerStorage::U16(vec![1, 2, 4]), vec![3, 1])
            .expect("integer tensor");

        let result = log2_builtin(Value::Tensor(tensor)).expect("log2");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                assert_eq!(out.materialize_f64(), vec![0.0, 1.0, 2.0]);
                assert!(out.integer_storage().is_none());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn log2_rejects_integer_values_outside_exact_binary64_interval() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor =
            Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
                .unwrap();
        let error = log2_builtin(Value::Tensor(tensor)).expect_err("wide integer must reject");
        assert_eq!(error.identifier(), LOG2_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log2_negative_typed_integer_tensor_promotes_to_complex_from_storage() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new_integer(IntegerStorage::I32(vec![-4, 4]), vec![1, 2])
            .expect("integer tensor");

        let result = log2_builtin(Value::Tensor(tensor)).expect("log2");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert!((out.materialize_f64()[0].0 - 2.0).abs() < 1e-12);
                assert!((out.materialize_f64()[0].1 - std::f64::consts::PI * LOG2_E).abs() < 1e-12);
                assert_eq!(out.materialize_f64()[1], (2.0, 0.0));
            }
            other => panic!("expected complex tensor result, got {other:?}"),
        }
    }

    #[test]
    fn log2_string_rejected_with_stable_identifier() {
        let err = log2_builtin(Value::from("bad")).expect_err("expected input error");
        assert_eq!(err.identifier(), LOG2_ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn log2_compatibility_mode_gates_integer_logical_and_character_extensions() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let integer = Tensor::new_integer(IntegerStorage::I8(vec![2]), vec![1, 1]).unwrap();
        assert_eq!(
            log2_builtin(Value::Tensor(integer))
                .unwrap_err()
                .identifier(),
            Some("RunMat:compatibility:Log2IntegerInputExtension")
        );
        assert_eq!(
            log2_builtin(Value::Bool(true)).unwrap_err().identifier(),
            Some("RunMat:compatibility:Log2LogicalInputExtension")
        );
        let chars = CharArray::new(vec!['A'], 1, 1).unwrap();
        assert_eq!(
            log2_builtin(Value::CharArray(chars))
                .unwrap_err()
                .identifier(),
            Some("RunMat:compatibility:Log2CharacterInputExtension")
        );
    }

    #[test]
    fn log2_type_preserves_tensor_shape() {
        let out = numeric_unary_type(
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
    fn log2_type_scalar_tensor_returns_num() {
        let out = numeric_unary_type(
            &[Type::Tensor {
                shape: Some(vec![Some(1), Some(1)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(out, Type::Num);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log2_scalar_one() {
        let result = log2_builtin(Value::Num(1.0)).expect("log2");
        match result {
            Value::Num(v) => assert!((v - 0.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log2_scalar_two() {
        let result = log2_builtin(Value::Num(2.0)).expect("log2");
        match result {
            Value::Num(v) => assert!((v - 1.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log2_scalar_zero() {
        let result = log2_builtin(Value::Num(0.0)).expect("log2");
        match result {
            Value::Num(v) => assert!(v.is_infinite() && v.is_sign_negative()),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log2_scalar_negative() {
        let result = log2_builtin(Value::Num(-1.0)).expect("log2");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 0.0).abs() < 1e-12);
                assert!((im - (std::f64::consts::PI * LOG2_E)).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log2_bool_true() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = log2_builtin(Value::Bool(true)).expect("log2");
        match result {
            Value::Num(v) => assert!((v - 0.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log2_logical_array_inputs() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let logical = LogicalArray::new(vec![1u8, 0, 1, 0], vec![2, 2]).expect("logical");
        let result = log2_builtin(Value::LogicalArray(logical)).expect("log2");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert!((t.materialize_f64()[0] - 0.0).abs() < 1e-12);
                assert!(
                    t.materialize_f64()[1].is_infinite()
                        && t.materialize_f64()[1].is_sign_negative()
                );
                assert!((t.materialize_f64()[2] - 0.0).abs() < 1e-12);
                assert!(
                    t.materialize_f64()[3].is_infinite()
                        && t.materialize_f64()[3].is_sign_negative()
                );
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log2_string_input_errors() {
        let err = log2_builtin(Value::from("hello")).unwrap_err();
        assert_eq!(err.identifier(), LOG2_ERROR_INVALID_INPUT.identifier);
        assert!(err.message().contains("expected numeric input"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log2_string_array_errors() {
        let array = StringArray::new(vec!["hello".to_string()], vec![1, 1]).unwrap();
        let err = log2_builtin(Value::StringArray(array)).unwrap_err();
        assert_eq!(err.identifier(), LOG2_ERROR_INVALID_INPUT.identifier);
        assert!(err.message().contains("expected numeric input"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log2_tensor_with_negatives() {
        let tensor = Tensor::new(vec![-1.0, 1.0], vec![1, 2]).unwrap();
        let result = log2_builtin(Value::Tensor(tensor)).expect("log2");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![1, 2]);
                assert!((ct.materialize_f64()[0].0 - 0.0).abs() < 1e-12);
                assert!(
                    (ct.materialize_f64()[0].1 - (std::f64::consts::PI * LOG2_E)).abs() < 1e-12
                );
                assert!((ct.materialize_f64()[1].0 - 0.0).abs() < 1e-12);
                assert!((ct.materialize_f64()[1].1).abs() < 1e-12);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn log2_preserves_native_single_real_complex_negative_and_empty_storage() {
        let tensor = Tensor::from_f32(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let Value::Tensor(output) = log2_builtin(Value::Tensor(tensor)).expect("log2") else {
            panic!("expected single real tensor");
        };
        assert_eq!(
            output.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![0.0, 1.0])
        );

        let tensor = Tensor::from_f32(vec![-2.0, 2.0], vec![1, 2]).unwrap();
        let Value::ComplexTensor(output) = log2_builtin(Value::Tensor(tensor)).expect("log2")
        else {
            panic!("expected complex single tensor");
        };
        assert_eq!(
            output.as_f32_slice(),
            Some(
                &[
                    log2_complex_parts_f32(-2.0, 0.0),
                    log2_complex_parts_f32(2.0, 0.0),
                ][..]
            )
        );

        let complex = ComplexTensor::from_f32(vec![(1.0, 1.0)], vec![1, 1]).unwrap();
        let Value::ComplexTensor(output) =
            log2_builtin(Value::ComplexTensor(complex)).expect("log2")
        else {
            panic!("one-element complex single must retain class");
        };
        assert_eq!(
            output.as_f32_slice(),
            Some(&[log2_complex_parts_f32(1.0, 1.0)][..])
        );

        let empty = ComplexTensor::from_f32(Vec::new(), vec![0, 2]).unwrap();
        let Value::ComplexTensor(output) = log2_builtin(Value::ComplexTensor(empty)).expect("log2")
        else {
            panic!("expected empty complex single tensor");
        };
        assert_eq!(output.shape, vec![0, 2]);
        assert_eq!(output.as_f32_slice(), Some(&[][..]));
    }

    #[test]
    fn log2_integer_gpu_gathers_exact_storage_before_floating_domain() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let wide = 9_007_199_254_740_992_u64;
            let tensor =
                Tensor::new_integer(IntegerStorage::U64(vec![1, wide]), vec![1, 2]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let Value::Tensor(output) = log2_builtin(Value::GpuTensor(handle)).expect("log2")
            else {
                panic!("expected host double tensor");
            };
            assert_eq!(
                output.into_numeric_storage().unwrap(),
                NumericStorage::F64(vec![0.0, (wide as f64).log2()])
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log2_complex_scalar() {
        let result = log2_builtin(Value::Complex(1.0, 2.0)).expect("log2");
        match result {
            Value::Complex(re, im) => {
                let (expected_re, expected_im) = log2_complex_parts(1.0, 2.0);
                assert!((re - expected_re).abs() < 1e-12);
                assert!((im - expected_im).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log2_char_array_inputs() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let chars = CharArray::new("AZ".chars().collect(), 1, 2).unwrap();
        let result = log2_builtin(Value::CharArray(chars)).expect("log2");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert!((t.materialize_f64()[0] - (65.0f64).log2()).abs() < 1e-12);
                assert!((t.materialize_f64()[1] - (90.0f64).log2()).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log2_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 4.0, 8.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = log2_builtin(Value::GpuTensor(handle)).expect("log2");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![4, 1]);
            let expected: Vec<f64> = tensor.materialize_f64().iter().map(|&v| v.log2()).collect();
            for (a, b) in gathered.materialize_f64().iter().zip(expected.iter()) {
                assert!((a - b).abs() < 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log2_gpu_negative_falls_back_to_complex() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![-2.0, 2.0], vec![1, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = log2_builtin(Value::GpuTensor(handle)).expect("log2");
            match result {
                Value::ComplexTensor(ct) => {
                    assert_eq!(ct.shape, vec![1, 2]);
                    assert!((ct.materialize_f64()[0].0 - 1.0).abs() < 1e-12);
                    assert!(
                        (ct.materialize_f64()[0].1 - (std::f64::consts::PI * LOG2_E)).abs() < 1e-12
                    );
                    assert!((ct.materialize_f64()[1].0 - 1.0).abs() < 1e-12);
                    assert!((ct.materialize_f64()[1].1).abs() < 1e-12);
                }
                other => panic!("expected complex tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn log2_wgpu_matches_cpu_elementwise() {
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        let tensor = Tensor::new(vec![1.0, 2.0, 4.0, 8.0], vec![4, 1]).unwrap();
        let cpu = log2_real(Value::Tensor(tensor.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu = block_on(log2_gpu(handle)).expect("log2 gpu");
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(gt.shape, ct.shape);
                let tol = match provider.precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                    runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                };
                for (a, b) in gt.materialize_f64().iter().zip(ct.materialize_f64().iter()) {
                    assert!((a - b).abs() < tol, "|{} - {}| >= {}", a, b, tol);
                }
            }
            _ => panic!("unexpected shapes"),
        }
    }
}
