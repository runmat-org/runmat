//! MATLAB-compatible natural logarithm (`log`) builtin with GPU-aware semantics for RunMat.
//!
//! Provides element-wise natural logarithms for real, logical, character, and complex inputs while
//! preserving MATLAB semantics, including promotion of negative real values to complex outputs.
//! GPU execution uses provider hooks when available and falls back to host computation whenever
//! complex results are required or the provider lacks a dedicated kernel.

use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, ComplexStorage, ComplexTensor, NumericStorage, Tensor, Value};

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::symbolic::symbolic_function;
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_value::SymbolicFunction;

const IMAG_EPS: f64 = 1e-12;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::log")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "log",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_log" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may execute log directly on device buffers; runtimes gather to host when complex outputs are required or the hook is unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::log")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "log",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Fusion is disabled because real inputs can require complex promotion and explicit gpuArray inputs have a distinct complex-domain contract.",
};

const BUILTIN_NAME: &str = "log";

const LOG_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elementwise natural-log result.",
}];
const LOG_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric, logical, char, or complex input.",
}];
const LOG_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = log(X)",
    inputs: &LOG_INPUTS,
    outputs: &LOG_OUTPUT,
}];
const LOG_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOG.INVALID_INPUT",
    identifier: Some("RunMat:log:InvalidInput"),
    when: "Input cannot be interpreted as numeric, logical, char, or complex data.",
    message: "log: invalid input",
};
const LOG_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOG.INTERNAL",
    identifier: Some("RunMat:log:Internal"),
    when: "Internal tensor construction or provider interaction failed.",
    message: "log: internal error",
};
const LOG_ERRORS: [BuiltinErrorDescriptor; 2] = [LOG_ERROR_INVALID_INPUT, LOG_ERROR_INTERNAL];
pub(super) const LOG_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "log-integer-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "log with integer input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:LogIntegerInputExtension"),
    };
pub(super) const LOG_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "log-logical-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "log with logical input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:LogLogicalInputExtension"),
    };
pub(super) const LOG_CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "log-character-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "log with character input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:LogCharacterInputExtension"),
    };
pub(super) const LOG_EXPLICIT_GPU_COMPLEX_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "log-explicit-real-gpu-complex-promotion",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "complex promotion from an explicit real gpuArray is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:LogExplicitGpuComplexExtension"),
    };
pub const LOG_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    LOG_INTEGER_INPUT_EXTENSION,
    LOG_LOGICAL_INPUT_EXTENSION,
    LOG_CHARACTER_INPUT_EXTENSION,
    LOG_EXPLICIT_GPU_COMPLEX_EXTENSION,
];
const LOG_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Accepted only in RunMat mode and only inside the exact binary64 integer interval.",
}];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "Y = log(integer_X)", inputs: &LOG_INTEGER_INPUTS,
    computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
    output_class: BuiltinIntegerOutputClassRule::Double,
    overflow: BuiltinIntegerOverflowRule::NotApplicable,
    backend: BuiltinIntegerBackendRule::GatherFallback,
    overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
    notes: "Resident integer input is downloaded exactly through its owner before entering the double logarithm domain.",
}];
pub const LOG_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LOG_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LOG_ERRORS,
};

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn log_error_with_detail(
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
    name = "log",
    category = "math/elementwise",
    summary = "Natural logarithm of scalars, vectors, matrices, or N-D tensors.",
    keywords = "log,natural logarithm,elementwise,gpu,complex",
    accel = "unary",
    type_resolver(numeric_unary_type),
    descriptor(crate::builtins::math::elementwise::log::LOG_DESCRIPTOR),
    extensions(LOG_EXTENSIONS),
    integer_capabilities(crate::builtins::math::elementwise::log::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::elementwise::log"
)]
async fn log_builtin(value: Value) -> BuiltinResult<Value> {
    if let Some(symbolic) = symbolic_function(&value, SymbolicFunction::Log) {
        return Ok(symbolic);
    }
    ensure_log_extensions(&value).await?;
    match value {
        Value::GpuTensor(handle) => log_gpu(handle).await,
        Value::Complex(re, im) => {
            let (r, i) = log_complex_parts(re, im);
            Ok(Value::Complex(r, i))
        }
        Value::ComplexTensor(ct) => {
            crate::builtins::common::validation::reject_typed_complex_integer_tensor(&ct, "log")?;
            log_complex_tensor(ct)
        }
        Value::CharArray(ca) => log_char_array(ca),
        Value::String(_) | Value::StringArray(_) => Err(log_error_with_detail(
            &LOG_ERROR_INVALID_INPUT,
            "expected numeric input",
        )),
        other => log_real(other),
    }
}

async fn ensure_log_extensions(value: &Value) -> BuiltinResult<()> {
    let integer = matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(t) if t.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(h) if runmat_accelerate_api::handle_integer_type(h).is_some());
    if integer {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LOG_INTEGER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
        if !crate::builtins::common::validation::native_integer_value_is_exact_f64_async(value)
            .await?
        {
            return Err(log_error_with_detail(
                &LOG_ERROR_INVALID_INPUT,
                "integer input lies outside the exact binary64 interval",
            ));
        }
    }
    if crate::builtins::common::validation::value_has_logical_class(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LOG_LOGICAL_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if matches!(value, Value::CharArray(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LOG_CHARACTER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

async fn log_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = gpu_helpers::exact_provider_for_handle(&handle).ok_or_else(|| {
        log_error_with_detail(
            &LOG_ERROR_INTERNAL,
            "GPU provider unavailable for input owner",
        )
    })?;
    if runmat_accelerate_api::handle_integer_type(&handle).is_some() {
        let gathered =
            gpu_helpers::download_value_preserving_residency_async(provider, &handle).await?;
        let result = log_real(gathered)?;
        return gpu_helpers::restore_class_preserving_value(&handle, result, BUILTIN_NAME);
    }
    if runmat_accelerate_api::handle_is_logical(&handle)
        || runmat_accelerate_api::handle_storage(&handle) == GpuTensorStorage::ComplexInterleaved
    {
        let gathered =
            gpu_helpers::download_value_preserving_residency_async(provider, &handle).await?;
        let result = match gathered {
            Value::ComplexTensor(ct) => log_complex_tensor(ct)?,
            other => log_real(other)?,
        };
        return gpu_helpers::restore_class_preserving_value(&handle, result, BUILTIN_NAME);
    }
    {
        match detect_gpu_requires_complex(provider, &handle).await {
            Ok(false) => {
                if let Ok(out) = provider.unary_log(&handle).await {
                    return validate_log_gpu_output(provider, &handle, out);
                }
            }
            Ok(true) => {
                if runmat_accelerate_api::handle_is_explicit(&handle) {
                    crate::compatibility::ensure_builtin_extension_enabled(
                        &LOG_EXPLICIT_GPU_COMPLEX_EXTENSION,
                        BUILTIN_NAME,
                    )?;
                }
                let gathered =
                    gpu_helpers::download_value_preserving_residency_async(provider, &handle)
                        .await?;
                let result = log_real(gathered)?;
                return gpu_helpers::restore_class_preserving_value(&handle, result, BUILTIN_NAME);
            }
            Err(err) => {
                if err.message() == "interaction pending..." {
                    return Err(err);
                }
                // Fall through to host fallback below.
            }
        }
    }
    let gathered =
        gpu_helpers::download_value_preserving_residency_async(provider, &handle).await?;
    let result = log_real(gathered)?;
    if matches!(result, Value::Complex(_, _) | Value::ComplexTensor(_))
        && runmat_accelerate_api::handle_is_explicit(&handle)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LOG_EXPLICIT_GPU_COMPLEX_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    gpu_helpers::restore_class_preserving_value(&handle, result, BUILTIN_NAME)
}

fn validate_log_gpu_output(
    provider: &'static dyn AccelProvider,
    source: &GpuTensorHandle,
    out: GpuTensorHandle,
) -> BuiltinResult<Value> {
    let valid = !gpu_helpers::same_gpu_handle(source, &out)
        && out.shape == source.shape
        && out.device_id == source.device_id
        && gpu_helpers::exact_provider_for_handle(&out)
            .is_some_and(|owner| std::ptr::eq(owner, provider))
        && runmat_accelerate_api::handle_storage(&out) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_precision(&out)
            == runmat_accelerate_api::handle_precision(source)
        && runmat_accelerate_api::handle_integer_type(&out).is_none()
        && !runmat_accelerate_api::handle_is_logical(&out);
    if !valid {
        gpu_helpers::free_unprotected_exact_owner(&out, &[source]);
        return Err(log_error_with_detail(
            &LOG_ERROR_INTERNAL,
            "provider returned malformed log output",
        ));
    }
    let mut out = out;
    runmat_accelerate_api::set_handle_provenance(
        &mut out,
        runmat_accelerate_api::handle_provenance(source)
            .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic),
    );
    Ok(gpu_helpers::resident_gpu_value(out))
}

pub(super) async fn detect_gpu_requires_complex(
    provider: &'static dyn AccelProvider,
    handle: &GpuTensorHandle,
) -> BuiltinResult<bool> {
    let min_handle = provider
        .reduce_min(handle)
        .await
        .map_err(|e| builtin_error(format!("log: reduce_min failed: {e}")))?;
    if gpu_helpers::same_gpu_handle(&min_handle, handle) {
        return Err(builtin_error("log: reduce_min aliased its input"));
    }
    if min_handle.device_id != handle.device_id
        || gpu_helpers::exact_provider_for_handle(&min_handle)
            .is_none_or(|owner| !std::ptr::eq(owner, provider))
        || runmat_accelerate_api::handle_storage(&min_handle) != GpuTensorStorage::Real
    {
        gpu_helpers::free_unprotected_exact_owner(&min_handle, &[handle]);
        return Err(builtin_error("log: reduce_min returned a malformed handle"));
    }
    let host = provider
        .download(&min_handle)
        .await
        .map_err(|e| builtin_error(format!("log: reduce_min download failed: {e}")));
    gpu_helpers::free_unprotected_exact_owner(&min_handle, &[handle]);
    let host = host?;
    if host.shape.iter().product::<usize>() != 1 {
        return Err(builtin_error("log: reduce_min returned a non-scalar"));
    }
    if host.data.iter().any(|&v| v.is_nan()) {
        return Err(builtin_error("log: reduce_min result contained NaN"));
    }
    Ok(host.data.iter().any(|&v| v.is_sign_negative()))
}

fn log_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("log", value)
        .map_err(|e| builtin_error(format!("log: {e}")))?;
    log_tensor_real(tensor)
}

fn log_tensor_real(tensor: Tensor) -> BuiltinResult<Value> {
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|e| builtin_error(format!("log: {e}")))?;
    match storage {
        NumericStorage::F64(values) => log_real_f64_values(values, shape),
        NumericStorage::F32(values) => log_real_f32_values(values, shape),
        storage => log_real_f64_values(promote_integer_storage_to_log_domain(storage), shape),
    }
}

fn log_real_f64_values(values: Vec<f64>, shape: Vec<usize>) -> BuiltinResult<Value> {
    let len = values.len();
    let mut complex_values = Vec::with_capacity(len);
    let mut has_imag = false;

    for v in values {
        let (mut real_part, mut imag_part) = log_complex_parts(v, 0.0);
        if real_part.is_finite() && real_part.abs() < IMAG_EPS {
            real_part = 0.0;
        }
        if !imag_part.is_finite() || imag_part.abs() < IMAG_EPS {
            imag_part = 0.0;
        }
        if imag_part != 0.0 {
            has_imag = true;
        }
        complex_values.push((real_part, imag_part));
    }

    if has_imag {
        let tensor =
            ComplexTensor::from_complex_storage(ComplexStorage::F64(complex_values), shape)
                .map_err(|e| builtin_error(format!("log: {e}")))?;
        Ok(complex_tensor_into_value(tensor))
    } else {
        let data: Vec<f64> = complex_values.into_iter().map(|(re, _)| re).collect();
        let tensor = Tensor::from_numeric_storage(NumericStorage::F64(data), shape)
            .map_err(|e| builtin_error(format!("log: {e}")))?;
        Ok(tensor::tensor_into_value(tensor))
    }
}

fn log_real_f32_values(values: Vec<f32>, shape: Vec<usize>) -> BuiltinResult<Value> {
    let mut complex_values = Vec::with_capacity(values.len());
    let mut has_imag = false;
    for value in values {
        let (mut real_part, mut imag_part) = log_complex_parts_f32(value, 0.0);
        if real_part.is_finite() && real_part.abs() < IMAG_EPS as f32 {
            real_part = 0.0;
        }
        if !imag_part.is_finite() || imag_part.abs() < IMAG_EPS as f32 {
            imag_part = 0.0;
        }
        has_imag |= imag_part != 0.0;
        complex_values.push((real_part, imag_part));
    }
    if has_imag {
        let tensor =
            ComplexTensor::from_complex_storage(ComplexStorage::F32(complex_values), shape)
                .map_err(|e| builtin_error(format!("log: {e}")))?;
        Ok(complex_tensor_into_value(tensor))
    } else {
        let values = complex_values.into_iter().map(|(real, _)| real).collect();
        let tensor = Tensor::from_numeric_storage(NumericStorage::F32(values), shape)
            .map_err(|e| builtin_error(format!("log: {e}")))?;
        Ok(tensor::tensor_into_value(tensor))
    }
}

fn log_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let shape = ct.shape.clone();
    let storage = match ct.into_complex_storage() {
        ComplexStorage::F64(values) => ComplexStorage::F64(
            values
                .into_iter()
                .map(|(real, imag)| {
                    let (mut real, mut imag) = log_complex_parts(real, imag);
                    if real.abs() < IMAG_EPS {
                        real = 0.0;
                    }
                    if imag.abs() < IMAG_EPS {
                        imag = 0.0;
                    }
                    (real, imag)
                })
                .collect(),
        ),
        ComplexStorage::F32(values) => ComplexStorage::F32(
            values
                .into_iter()
                .map(|(real, imag)| {
                    let (mut real, mut imag) = log_complex_parts_f32(real, imag);
                    if real.abs() < IMAG_EPS as f32 {
                        real = 0.0;
                    }
                    if imag.abs() < IMAG_EPS as f32 {
                        imag = 0.0;
                    }
                    (real, imag)
                })
                .collect(),
        ),
        ComplexStorage::Integer(_) => {
            return Err(log_error_with_detail(
                &LOG_ERROR_INVALID_INPUT,
                "typed complex integer input is not supported",
            ))
        }
    };
    let tensor = ComplexTensor::from_complex_storage(storage, shape)
        .map_err(|e| builtin_error(format!("log: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

fn promote_integer_storage_to_log_domain(storage: NumericStorage) -> Vec<f64> {
    storage
        .into_integer_storage()
        .expect("log integer-promotion boundary received floating storage")
        .to_f64_vec()
}

fn log_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data: Vec<f64> = ca.data.iter().map(|&ch| ch as u32 as f64).collect();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| builtin_error(format!("log: {e}")))?;
    log_tensor_real(tensor)
}

pub(super) fn log_complex_parts(re: f64, im: f64) -> (f64, f64) {
    let magnitude = re.hypot(im);
    if magnitude == 0.0 {
        (f64::NEG_INFINITY, 0.0)
    } else {
        let real_part = magnitude.ln();
        let imag_part = im.atan2(re);
        (real_part, imag_part)
    }
}

pub(super) fn log_complex_parts_f32(re: f32, im: f32) -> (f32, f32) {
    let magnitude = re.hypot(im);
    if magnitude == 0.0 {
        (f32::NEG_INFINITY, 0.0)
    } else {
        (magnitude.ln(), im.atan2(re))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{ResolveContext, Type};
    use runmat_value::{IntValue, IntegerStorage, LogicalArray, Tensor, Value};

    fn log_builtin(value: Value) -> BuiltinResult<Value> {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        block_on(super::log_builtin(value))
    }

    #[test]
    fn log_integer_extension_is_rejected_in_matlab_mode() {
        let _matlab = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(super::log_builtin(Value::Int(IntValue::I64(2))))
            .expect_err("integer log is a RunMat extension");
        assert_eq!(
            error.identifier(),
            LOG_INTEGER_INPUT_EXTENSION.error_identifier
        );
    }

    #[test]
    fn log_fusion_is_disabled_until_complex_domain_is_representable() {
        assert!(FUSION_SPEC.elementwise.is_none());
    }

    #[test]
    fn log_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = LOG_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = log(X)"));
    }

    #[test]
    fn log_string_rejected_with_stable_identifier() {
        let err = log_builtin(Value::from("bad")).expect_err("expected input error");
        assert_eq!(err.identifier(), LOG_ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn log_type_preserves_tensor_shape() {
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
    fn log_type_scalar_tensor_returns_num() {
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
    fn log_scalar_one() {
        let result = log_builtin(Value::Num(1.0)).expect("log");
        match result {
            Value::Num(v) => assert!((v - 0.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_scalar_zero() {
        let result = log_builtin(Value::Num(0.0)).expect("log");
        match result {
            Value::Num(v) => assert!(v.is_infinite() && v.is_sign_negative()),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_scalar_negative() {
        let result = log_builtin(Value::Num(-1.0)).expect("log");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 0.0).abs() < 1e-12);
                assert!((im - std::f64::consts::PI).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_scalar_nan_remains_real() {
        let result = log_builtin(Value::Num(f64::NAN)).expect("log");
        match result {
            Value::Num(v) => assert!(v.is_nan()),
            other => panic!("expected real NaN, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_bool_true() {
        let result = log_builtin(Value::Bool(true)).expect("log");
        match result {
            Value::Num(v) => assert!((v - 0.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_logical_array_inputs() {
        let logical = LogicalArray::new(vec![1u8, 0, 1, 0], vec![2, 2]).expect("logical");
        let result = log_builtin(Value::LogicalArray(logical)).expect("log");
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
    fn log_string_input_errors() {
        let err = log_builtin(Value::from("hello")).unwrap_err();
        assert_eq!(err.identifier(), LOG_ERROR_INVALID_INPUT.identifier);
        assert!(err.message().contains("expected numeric input"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_tensor_with_negatives() {
        let tensor = Tensor::new(vec![-1.0, 1.0], vec![1, 2]).unwrap();
        let result = log_builtin(Value::Tensor(tensor)).expect("log");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![1, 2]);
                assert!((ct.materialize_f64()[0].0 - 0.0).abs() < 1e-12);
                assert!((ct.materialize_f64()[0].1 - std::f64::consts::PI).abs() < 1e-12);
                assert!((ct.materialize_f64()[1].0 - 0.0).abs() < 1e-12);
                assert!((ct.materialize_f64()[1].1).abs() < 1e-12);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn log_preserves_native_single_real_complex_negative_and_empty_storage() {
        let tensor = Tensor::from_f32(vec![1.0, std::f32::consts::E], vec![2, 1]).unwrap();
        let Value::Tensor(output) = log_builtin(Value::Tensor(tensor)).expect("log") else {
            panic!("expected single real tensor");
        };
        assert_eq!(
            output.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![0.0, std::f32::consts::E.ln()])
        );

        let tensor = Tensor::from_f32(vec![-1.0, 4.0], vec![1, 2]).unwrap();
        let Value::ComplexTensor(output) = log_builtin(Value::Tensor(tensor)).expect("log") else {
            panic!("expected complex single tensor");
        };
        assert_eq!(
            output.as_f32_slice(),
            Some(
                &[
                    log_complex_parts_f32(-1.0, 0.0),
                    log_complex_parts_f32(4.0, 0.0),
                ][..]
            )
        );

        let complex = ComplexTensor::from_f32(vec![(1.0, 1.0)], vec![1, 1]).unwrap();
        let Value::ComplexTensor(output) = log_builtin(Value::ComplexTensor(complex)).expect("log")
        else {
            panic!("one-element complex single must retain class");
        };
        assert_eq!(
            output.as_f32_slice(),
            Some(&[log_complex_parts_f32(1.0, 1.0)][..])
        );

        let empty = ComplexTensor::from_f32(Vec::new(), vec![0, 3]).unwrap();
        let Value::ComplexTensor(output) = log_builtin(Value::ComplexTensor(empty)).expect("log")
        else {
            panic!("expected empty complex single tensor");
        };
        assert_eq!(output.shape, vec![0, 3]);
        assert_eq!(output.as_f32_slice(), Some(&[][..]));
    }

    #[test]
    fn log_integer_gpu_gathers_exact_storage_before_floating_domain() {
        test_support::with_test_provider(|provider| {
            let wide = 9_007_199_254_740_992_u64;
            let tensor =
                Tensor::new_integer(IntegerStorage::U64(vec![1, wide]), vec![1, 2]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let Value::GpuTensor(output) = log_builtin(Value::GpuTensor(handle)).expect("log")
            else {
                panic!("expected restored GPU tensor");
            };
            let Value::Tensor(output) = futures::executor::block_on(
                gpu_helpers::download_value_preserving_residency_async(provider, &output),
            )
            .expect("download log result") else {
                panic!("expected real tensor result");
            };
            assert_eq!(
                output.into_numeric_storage().unwrap(),
                NumericStorage::F64(vec![0.0, (wide as f64).ln()])
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_complex_scalar() {
        let result = log_builtin(Value::Complex(1.0, 2.0)).expect("log");
        match result {
            Value::Complex(re, im) => {
                let expected_re = (1.0_f64.hypot(2.0)).ln();
                let expected_im = 2.0_f64.atan2(1.0);
                assert!((re - expected_re).abs() < 1e-12);
                assert!((im - expected_im).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_char_array_inputs() {
        let chars = CharArray::new("AZ".chars().collect(), 1, 2).unwrap();
        let result = log_builtin(Value::CharArray(chars)).expect("log");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert!((t.materialize_f64()[0] - (65.0f64).ln()).abs() < 1e-12);
                assert!((t.materialize_f64()[1] - (90.0f64).ln()).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 4.0], vec![3, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = log_builtin(Value::GpuTensor(handle)).expect("log");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![3, 1]);
            let expected: Vec<f64> = tensor.materialize_f64().iter().map(|&v| v.ln()).collect();
            for (a, b) in gathered.materialize_f64().iter().zip(expected.iter()) {
                assert!((a - b).abs() < 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_gpu_negative_falls_back_to_complex() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![-1.0, 1.0], vec![1, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let Value::GpuTensor(result) = log_builtin(Value::GpuTensor(handle)).expect("log")
            else {
                panic!("expected restored GPU result");
            };
            let Value::ComplexTensor(ct) = futures::executor::block_on(
                gpu_helpers::download_value_preserving_residency_async(provider, &result),
            )
            .expect("download complex result") else {
                panic!("expected complex tensor");
            };
            assert_eq!(ct.shape, vec![1, 2]);
            assert!((ct.materialize_f64()[0].0 - 0.0).abs() < 1e-12);
            assert!((ct.materialize_f64()[0].1 - std::f64::consts::PI).abs() < 1e-12);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_with_integer_argument() {
        let result = log_builtin(Value::Int(IntValue::I32(4))).expect("log");
        match result {
            Value::Num(v) => assert!((v - (4.0f64).ln()).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_reads_typed_integer_tensor_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![1, 2, 4]), vec![3, 1])
            .expect("integer tensor");

        let result = log_builtin(Value::Tensor(tensor)).expect("log");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                let expected = [0.0, 2.0f64.ln(), 4.0f64.ln()];
                for (actual, expected) in out.materialize_f64().iter().zip(expected.iter()) {
                    assert!((actual - expected).abs() < 1e-12);
                }
                assert!(out.integer_storage().is_none());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_negative_typed_integer_tensor_promotes_to_complex_from_storage() {
        let tensor = Tensor::new_integer(IntegerStorage::I16(vec![-1, 1]), vec![1, 2])
            .expect("integer tensor");

        let result = log_builtin(Value::Tensor(tensor)).expect("log");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert_eq!(out.materialize_f64()[0].0, 0.0);
                assert!((out.materialize_f64()[0].1 - std::f64::consts::PI).abs() < 1e-12);
                assert_eq!(out.materialize_f64()[1], (0.0, 0.0));
            }
            other => panic!("expected complex tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn log_wgpu_matches_cpu_elementwise() {
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        let tensor = Tensor::new(vec![1.0, 2.0, 4.0, 8.0], vec![4, 1]).unwrap();
        let cpu = log_real(Value::Tensor(tensor.clone())).expect("cpu log");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_value = block_on(log_gpu(handle)).expect("gpu log");
        let gathered = test_support::gather(gpu_value).expect("gather");
        match cpu {
            Value::Tensor(ct) => {
                assert_eq!(gathered.shape, ct.shape);
                for (gpu, cpu) in gathered
                    .materialize_f64()
                    .iter()
                    .zip(ct.materialize_f64().iter())
                {
                    let tol = match runmat_accelerate_api::provider().unwrap().precision() {
                        runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                        runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                    };
                    assert!((gpu - cpu).abs() < tol, "|{gpu} - {cpu}| >= {tol}");
                }
            }
            Value::Num(_) => panic!("expected tensor result from cpu path"),
            other => panic!("unexpected cpu result {other:?}"),
        }
    }
}
