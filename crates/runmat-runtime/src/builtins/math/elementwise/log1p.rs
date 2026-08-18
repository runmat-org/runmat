//! MATLAB-compatible `log1p` builtin with GPU-aware semantics for RunMat.
//!
//! Provides an element-wise `log(1 + x)` with improved accuracy for small magnitudes, covering
//! real, logical, character, and complex inputs. GPU execution uses provider hooks when available
//! and falls back to host computation whenever complex results are required or device support is
//! missing, mirroring MATLAB behavior.

use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};
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

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const IMAG_EPS: f64 = 1e-12;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::log1p")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "log1p",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_log1p" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers should supply unary_log1p and reduce_min; runtimes gather to host when complex outputs are required or either hook is unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::log1p")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "log1p",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Fusion is disabled because raw log(x+1) loses log1p accuracy and real inputs can require complex promotion.",
};

const BUILTIN_NAME: &str = "log1p";

const LOG1P_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elementwise log(1+x) result.",
}];
const LOG1P_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric, logical, char, or complex input.",
}];
const LOG1P_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = log1p(X)",
    inputs: &LOG1P_INPUTS,
    outputs: &LOG1P_OUTPUT,
}];
const LOG1P_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOG1P.INVALID_INPUT",
    identifier: Some("RunMat:log1p:InvalidInput"),
    when: "Input cannot be interpreted as numeric, logical, char, or complex data.",
    message: "log1p: invalid input",
};
const LOG1P_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOG1P.INTERNAL",
    identifier: Some("RunMat:log1p:Internal"),
    when: "Internal tensor construction or provider interaction failed.",
    message: "log1p: internal error",
};
const LOG1P_ERRORS: [BuiltinErrorDescriptor; 2] = [LOG1P_ERROR_INVALID_INPUT, LOG1P_ERROR_INTERNAL];
const LOG1P_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "log1p-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "log1p with integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Log1pIntegerInputExtension"),
};
const LOG1P_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "log1p-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "log1p with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Log1pLogicalInputExtension"),
};
const LOG1P_CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "log1p-character-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "log1p with character input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Log1pCharacterInputExtension"),
};
const LOG1P_EXPLICIT_GPU_COMPLEX_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "log1p-explicit-real-gpu-complex-promotion",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "complex promotion from an explicit real gpuArray is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Log1pExplicitGpuComplexExtension"),
    };
pub const LOG1P_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    LOG1P_INTEGER_INPUT_EXTENSION,
    LOG1P_LOGICAL_INPUT_EXTENSION,
    LOG1P_CHARACTER_INPUT_EXTENSION,
    LOG1P_EXPLICIT_GPU_COMPLEX_EXTENSION,
];
const LOG1P_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Accepted only in RunMat mode and inside the exact binary64 integer interval.",
}];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor { form: "Y = log1p(integer_X)", inputs: &LOG1P_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving, notes: "Resident integer input is downloaded exactly through its owner before double-domain computation." }];
pub const LOG1P_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LOG1P_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LOG1P_ERRORS,
};

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn log1p_error_with_detail(
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
    name = "log1p",
    category = "math/elementwise",
    summary = "Accurate element-wise computation of log(1 + x).",
    keywords = "log1p,log(1+x),natural logarithm,elementwise,gpu,precision",
    accel = "unary",
    type_resolver(numeric_unary_type),
    descriptor(crate::builtins::math::elementwise::log1p::LOG1P_DESCRIPTOR),
    extensions(LOG1P_EXTENSIONS),
    integer_capabilities(crate::builtins::math::elementwise::log1p::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::elementwise::log1p"
)]
async fn log1p_builtin(value: Value) -> BuiltinResult<Value> {
    ensure_log1p_extensions(&value).await?;
    match value {
        Value::GpuTensor(handle) => log1p_gpu(handle).await,
        Value::Complex(re, im) => {
            let (real, imag) = log1p_complex_parts(re, im);
            Ok(Value::Complex(real, imag))
        }
        Value::ComplexTensor(ct) => {
            crate::builtins::common::validation::reject_typed_complex_integer_tensor(&ct, "log1p")?;
            log1p_complex_tensor(ct)
        }
        Value::CharArray(ca) => log1p_char_array(ca),
        Value::String(_) | Value::StringArray(_) => Err(log1p_error_with_detail(
            &LOG1P_ERROR_INVALID_INPUT,
            "expected numeric input",
        )),
        other => log1p_real(other),
    }
}

async fn ensure_log1p_extensions(value: &Value) -> BuiltinResult<()> {
    let integer = matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(t) if t.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(h) if runmat_accelerate_api::handle_integer_type(h).is_some());
    if integer {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LOG1P_INTEGER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
        if !crate::builtins::common::validation::native_integer_value_is_exact_f64_async(value)
            .await?
        {
            return Err(log1p_error_with_detail(
                &LOG1P_ERROR_INVALID_INPUT,
                "integer input lies outside the exact binary64 interval",
            ));
        }
    }
    if crate::builtins::common::validation::value_has_logical_class(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LOG1P_LOGICAL_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if matches!(value, Value::CharArray(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LOG1P_CHARACTER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

async fn log1p_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = gpu_helpers::exact_provider_for_handle(&handle).ok_or_else(|| {
        log1p_error_with_detail(
            &LOG1P_ERROR_INTERNAL,
            "GPU provider unavailable for input owner",
        )
    })?;
    if runmat_accelerate_api::handle_integer_type(&handle).is_some() {
        let gathered =
            gpu_helpers::download_value_preserving_residency_async(provider, &handle).await?;
        let result = log1p_real(gathered)?;
        return gpu_helpers::restore_class_preserving_value(&handle, result, BUILTIN_NAME);
    }
    if runmat_accelerate_api::handle_is_logical(&handle)
        || runmat_accelerate_api::handle_storage(&handle) == GpuTensorStorage::ComplexInterleaved
    {
        let gathered =
            gpu_helpers::download_value_preserving_residency_async(provider, &handle).await?;
        let result = match gathered {
            Value::ComplexTensor(ct) => log1p_complex_tensor(ct)?,
            other => log1p_real(other)?,
        };
        return gpu_helpers::restore_class_preserving_value(&handle, result, BUILTIN_NAME);
    }
    {
        match detect_gpu_requires_complex(provider, &handle).await {
            Ok(true) => {
                if runmat_accelerate_api::handle_is_explicit(&handle) {
                    crate::compatibility::ensure_builtin_extension_enabled(
                        &LOG1P_EXPLICIT_GPU_COMPLEX_EXTENSION,
                        BUILTIN_NAME,
                    )?;
                }
                let gathered =
                    gpu_helpers::download_value_preserving_residency_async(provider, &handle)
                        .await?;
                let result = log1p_real(gathered)?;
                return gpu_helpers::restore_class_preserving_value(&handle, result, BUILTIN_NAME);
            }
            Ok(false) => {
                if let Ok(out) = provider.unary_log1p(&handle).await {
                    return validate_log1p_gpu_output(provider, &handle, out);
                }
            }
            Err(err) => {
                if err.message() == "interaction pending..." {
                    return Err(builtin_error("interaction pending..."));
                }
            }
        }
    }
    let gathered =
        gpu_helpers::download_value_preserving_residency_async(provider, &handle).await?;
    let result = log1p_real(gathered)?;
    if matches!(result, Value::Complex(_, _) | Value::ComplexTensor(_))
        && runmat_accelerate_api::handle_is_explicit(&handle)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LOG1P_EXPLICIT_GPU_COMPLEX_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    gpu_helpers::restore_class_preserving_value(&handle, result, BUILTIN_NAME)
}

fn validate_log1p_gpu_output(
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
        return Err(log1p_error_with_detail(
            &LOG1P_ERROR_INTERNAL,
            "provider returned malformed log1p output",
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

async fn detect_gpu_requires_complex(
    provider: &'static dyn AccelProvider,
    handle: &GpuTensorHandle,
) -> BuiltinResult<bool> {
    let min_handle = provider
        .reduce_min(handle)
        .await
        .map_err(|e| builtin_error(format!("log1p: reduce_min failed: {e}")))?;
    if gpu_helpers::same_gpu_handle(&min_handle, handle) {
        return Err(builtin_error("log1p: reduce_min aliased its input"));
    }
    if min_handle.device_id != handle.device_id
        || gpu_helpers::exact_provider_for_handle(&min_handle)
            .is_none_or(|owner| !std::ptr::eq(owner, provider))
        || runmat_accelerate_api::handle_storage(&min_handle) != GpuTensorStorage::Real
    {
        gpu_helpers::free_unprotected_exact_owner(&min_handle, &[handle]);
        return Err(builtin_error(
            "log1p: reduce_min returned a malformed handle",
        ));
    }
    let host = provider
        .download(&min_handle)
        .await
        .map_err(|e| builtin_error(format!("log1p: reduce_min download failed: {e}")));
    gpu_helpers::free_unprotected_exact_owner(&min_handle, &[handle]);
    let host = host?;
    if host.shape.iter().product::<usize>() != 1 {
        return Err(builtin_error("log1p: reduce_min returned a non-scalar"));
    }
    if host.data.iter().any(|&v| v.is_nan()) {
        return Err(builtin_error("log1p: reduce_min result contained NaN"));
    }
    Ok(host.data.iter().any(|&v| v < -1.0))
}

fn log1p_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("log1p", value)
        .map_err(|e| builtin_error(format!("log1p: {e}")))?;
    log1p_tensor(tensor)
}

fn log1p_tensor(tensor: Tensor) -> BuiltinResult<Value> {
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|e| builtin_error(format!("log1p: {e}")))?;
    match storage {
        NumericStorage::F64(values) => log1p_real_f64_values(values, shape),
        NumericStorage::F32(values) => log1p_real_f32_values(values, shape),
        storage => log1p_real_f64_values(promote_integer_storage_to_log1p_domain(storage), shape),
    }
}

fn log1p_real_f64_values(values: Vec<f64>, shape: Vec<usize>) -> BuiltinResult<Value> {
    let mut entries = Vec::with_capacity(values.len());
    let mut has_imag = false;

    for v in values {
        let sum = 1.0 + v;
        if sum.is_nan() {
            entries.push((f64::NAN, 0.0));
            continue;
        }
        if sum < 0.0 {
            let (mut real_part, mut imag_part) = log1p_complex_parts(v, 0.0);
            if real_part.abs() < IMAG_EPS {
                real_part = 0.0;
            }
            if imag_part.abs() < IMAG_EPS {
                imag_part = 0.0;
            }
            if imag_part != 0.0 {
                has_imag = true;
            }
            entries.push((real_part, imag_part));
        } else {
            entries.push((v.ln_1p(), 0.0));
        }
    }

    if has_imag {
        let tensor = ComplexTensor::from_complex_storage(ComplexStorage::F64(entries), shape)
            .map_err(|e| builtin_error(format!("log1p: {e}")))?;
        Ok(complex_tensor_into_value(tensor))
    } else {
        let data: Vec<f64> = entries.into_iter().map(|(re, _)| re).collect();
        let tensor = Tensor::from_numeric_storage(NumericStorage::F64(data), shape)
            .map_err(|e| builtin_error(format!("log1p: {e}")))?;
        Ok(tensor::tensor_into_value(tensor))
    }
}

fn log1p_real_f32_values(values: Vec<f32>, shape: Vec<usize>) -> BuiltinResult<Value> {
    let mut entries = Vec::with_capacity(values.len());
    let mut has_imag = false;
    for value in values {
        let sum = 1.0 + value;
        if sum.is_nan() {
            entries.push((f32::NAN, 0.0));
        } else if sum < 0.0 {
            let (mut real, mut imag) = log1p_complex_parts_f32(value, 0.0);
            if real.abs() < IMAG_EPS as f32 {
                real = 0.0;
            }
            if imag.abs() < IMAG_EPS as f32 {
                imag = 0.0;
            }
            has_imag |= imag != 0.0;
            entries.push((real, imag));
        } else {
            entries.push((value.ln_1p(), 0.0));
        }
    }
    if has_imag {
        let tensor = ComplexTensor::from_complex_storage(ComplexStorage::F32(entries), shape)
            .map_err(|e| builtin_error(format!("log1p: {e}")))?;
        Ok(complex_tensor_into_value(tensor))
    } else {
        let values = entries.into_iter().map(|(real, _)| real).collect();
        let tensor = Tensor::from_numeric_storage(NumericStorage::F32(values), shape)
            .map_err(|e| builtin_error(format!("log1p: {e}")))?;
        Ok(tensor::tensor_into_value(tensor))
    }
}

fn log1p_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let shape = ct.shape.clone();
    let storage = match ct.into_complex_storage() {
        ComplexStorage::F64(values) => ComplexStorage::F64(
            values
                .into_iter()
                .map(|(real, imag)| {
                    let (mut real, mut imag) = log1p_complex_parts(real, imag);
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
                    let (mut real, mut imag) = log1p_complex_parts_f32(real, imag);
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
            return Err(log1p_error_with_detail(
                &LOG1P_ERROR_INVALID_INPUT,
                "typed complex integer input is not supported",
            ))
        }
    };
    let tensor = ComplexTensor::from_complex_storage(storage, shape)
        .map_err(|e| builtin_error(format!("log1p: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

fn promote_integer_storage_to_log1p_domain(storage: NumericStorage) -> Vec<f64> {
    storage
        .into_integer_storage()
        .expect("log1p integer-promotion boundary received floating storage")
        .to_f64_vec()
}

fn log1p_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data: Vec<f64> = ca.data.iter().map(|&ch| ch as u32 as f64).collect();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| builtin_error(format!("log1p: {e}")))?;
    log1p_tensor(tensor)
}

fn log1p_complex_parts(re: f64, im: f64) -> (f64, f64) {
    let shifted_re = re + 1.0;
    let magnitude = shifted_re.hypot(im);
    if magnitude == 0.0 {
        (f64::NEG_INFINITY, 0.0)
    } else {
        let real_part = magnitude.ln();
        let imag_part = im.atan2(shifted_re);
        (real_part, imag_part)
    }
}

fn log1p_complex_parts_f32(re: f32, im: f32) -> (f32, f32) {
    let shifted_re = re + 1.0;
    let magnitude = shifted_re.hypot(im);
    if magnitude == 0.0 {
        (f32::NEG_INFINITY, 0.0)
    } else {
        (magnitude.ln(), im.atan2(shifted_re))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, IntegerStorage, LogicalArray, ResolveContext, Tensor, Type};
    use std::f64::consts::PI;

    fn log1p_builtin(value: Value) -> BuiltinResult<Value> {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        block_on(super::log1p_builtin(value))
    }

    #[test]
    fn log1p_integer_extension_is_rejected_in_matlab_mode() {
        let _matlab = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(super::log1p_builtin(Value::Int(IntValue::I64(1))))
            .expect_err("integer log1p is a RunMat extension");
        assert_eq!(
            error.identifier(),
            LOG1P_INTEGER_INPUT_EXTENSION.error_identifier
        );
    }

    #[test]
    fn log1p_fusion_is_disabled_to_preserve_near_zero_accuracy() {
        assert!(FUSION_SPEC.elementwise.is_none());
    }

    #[test]
    fn log1p_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = LOG1P_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = log1p(X)"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log1p_reads_typed_integer_tensor_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::I16(vec![0, 1, 3]), vec![3, 1])
            .expect("integer tensor");

        let result = log1p_builtin(Value::Tensor(tensor)).expect("log1p");
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
    fn log1p_less_than_negative_one_typed_integer_promotes_to_complex_from_storage() {
        let tensor = Tensor::new_integer(IntegerStorage::I16(vec![-2, 0]), vec![1, 2])
            .expect("integer tensor");

        let result = log1p_builtin(Value::Tensor(tensor)).expect("log1p");
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

    #[test]
    fn log1p_preserves_native_single_real_complex_negative_and_empty_storage() {
        let tensor = Tensor::from_f32(vec![0.0, 0.5], vec![2, 1]).unwrap();
        let Value::Tensor(output) = log1p_builtin(Value::Tensor(tensor)).expect("log1p") else {
            panic!("expected single real tensor");
        };
        assert_eq!(
            output.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![0.0, 0.5_f32.ln_1p()])
        );

        let tensor = Tensor::from_f32(vec![-2.0, 1.0], vec![1, 2]).unwrap();
        let Value::ComplexTensor(output) = log1p_builtin(Value::Tensor(tensor)).expect("log1p")
        else {
            panic!("expected complex single tensor");
        };
        assert_eq!(
            output.as_f32_slice(),
            Some(
                &[
                    log1p_complex_parts_f32(-2.0, 0.0),
                    log1p_complex_parts_f32(1.0, 0.0),
                ][..]
            )
        );

        let complex = ComplexTensor::from_f32(vec![(1.0, 1.0)], vec![1, 1]).unwrap();
        let Value::ComplexTensor(output) =
            log1p_builtin(Value::ComplexTensor(complex)).expect("log1p")
        else {
            panic!("one-element complex single must retain class");
        };
        assert_eq!(
            output.as_f32_slice(),
            Some(&[log1p_complex_parts_f32(1.0, 1.0)][..])
        );

        let empty = ComplexTensor::from_f32(Vec::new(), vec![0, 3]).unwrap();
        let Value::ComplexTensor(output) =
            log1p_builtin(Value::ComplexTensor(empty)).expect("log1p")
        else {
            panic!("expected empty complex single tensor");
        };
        assert_eq!(output.shape, vec![0, 3]);
        assert_eq!(output.as_f32_slice(), Some(&[][..]));
    }

    #[test]
    fn log1p_integer_gpu_gathers_exact_storage_before_floating_domain() {
        test_support::with_test_provider(|provider| {
            let wide = 9_007_199_254_740_992_u64;
            let tensor =
                Tensor::new_integer(IntegerStorage::U64(vec![0, wide]), vec![1, 2]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let Value::GpuTensor(output) = log1p_builtin(Value::GpuTensor(handle)).expect("log1p")
            else {
                panic!("expected restored GPU tensor");
            };
            let Value::Tensor(output) = futures::executor::block_on(
                gpu_helpers::download_value_preserving_residency_async(provider, &output),
            )
            .expect("download log1p result") else {
                panic!("expected real tensor result");
            };
            assert_eq!(
                output.into_numeric_storage().unwrap(),
                NumericStorage::F64(vec![0.0, (wide as f64).ln_1p()])
            );
        });
    }

    #[test]
    fn log1p_string_rejected_with_stable_identifier() {
        let err = log1p_builtin(Value::from("bad")).expect_err("expected input error");
        assert_eq!(err.identifier(), LOG1P_ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn log1p_type_preserves_tensor_shape() {
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
    fn log1p_type_scalar_tensor_returns_num() {
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
    fn log1p_scalar_zero() {
        let result = log1p_builtin(Value::Num(0.0)).expect("log1p");
        match result {
            Value::Num(v) => assert!((v - 0.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log1p_scalar_negative_one() {
        let result = log1p_builtin(Value::Num(-1.0)).expect("log1p");
        match result {
            Value::Num(v) => assert!(v.is_infinite() && v.is_sign_negative()),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log1p_scalar_less_than_negative_one_complex() {
        let result = log1p_builtin(Value::Num(-2.0)).expect("log1p");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 0.0).abs() < 1e-12);
                assert!((im - PI).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log1p_tensor_mixed_values() {
        let tensor = Tensor::new(vec![0.0, -0.5, -2.0, 3.0], vec![2, 2]).unwrap();
        let result = log1p_builtin(Value::Tensor(tensor)).expect("log1p");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![2, 2]);
                let expected = [
                    (0.0, 0.0),
                    ((0.5f64).ln(), 0.0),
                    (0.0, PI),
                    ((4.0f64).ln(), 0.0),
                ];
                for ((re, im), (er, ei)) in ct.materialize_f64().iter().zip(expected.iter()) {
                    assert!((re - er).abs() < 1e-12);
                    assert!((im - ei).abs() < 1e-12);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log1p_complex_input() {
        let result = log1p_builtin(Value::Complex(0.5, 1.0)).expect("log1p");
        match result {
            Value::Complex(re, im) => {
                let expected = (1.5f64.hypot(1.0).ln(), 1.0f64.atan2(1.5));
                assert!((re - expected.0).abs() < 1e-12);
                assert!((im - expected.1).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log1p_char_array_roundtrip() {
        let chars = CharArray::new("ABC".chars().collect(), 1, 3).unwrap();
        let result = log1p_builtin(Value::CharArray(chars)).expect("log1p");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                for (idx, ch) in ['A', 'B', 'C'].into_iter().enumerate() {
                    let expected = (ch as u32 as f64).ln_1p();
                    assert!((t.materialize_f64()[idx] - expected).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log1p_string_rejects() {
        let err = log1p_builtin(Value::from("not numeric")).expect_err("should fail");
        assert!(
            err.message().contains("expected numeric input"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log1p_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, -0.25, 0.5, 2.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = log1p_builtin(Value::GpuTensor(handle)).expect("log1p");
            let gathered = test_support::gather(result).expect("gather");
            let expected: Vec<f64> = tensor
                .materialize_f64()
                .iter()
                .map(|&v| v.ln_1p())
                .collect();
            assert_eq!(gathered.shape, vec![4, 1]);
            for (out, exp) in gathered.materialize_f64().iter().zip(expected.iter()) {
                assert!((out - exp).abs() < 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log1p_bool_promotes() {
        let result = log1p_builtin(Value::Bool(true)).expect("log1p");
        match result {
            Value::Num(v) => assert!((v - 2.0f64.ln()).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log1p_logical_array_converts() {
        let logical = LogicalArray::new(vec![0, 1], vec![2, 1]).unwrap();
        let result = log1p_builtin(Value::LogicalArray(logical)).expect("log1p");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                assert!((t.materialize_f64()[0] - 0.0).abs() < 1e-12);
                assert!((t.materialize_f64()[1] - 2.0f64.ln()).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log1p_gpu_complex_falls_back() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![-2.0, -3.0], vec![2, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let Value::GpuTensor(result) = log1p_builtin(Value::GpuTensor(handle)).expect("log1p")
            else {
                panic!("expected restored GPU result");
            };
            let Value::ComplexTensor(ct) = futures::executor::block_on(
                gpu_helpers::download_value_preserving_residency_async(provider, &result),
            )
            .expect("download complex result") else {
                panic!("expected complex tensor result");
            };
            assert_eq!(ct.shape, vec![2, 1]);
            let expected = [(0.0, PI), ((2.0f64).ln(), PI)];
            for ((re, im), (er, ei)) in ct.materialize_f64().iter().zip(expected.iter()) {
                assert!((re - er).abs() < 1e-12);
                assert!((im - ei).abs() < 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn log1p_wgpu_matches_cpu() {
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        let tensor = Tensor::new(vec![0.0, -0.25, 0.25, 1.0], vec![4, 1]).unwrap();
        let cpu = log1p_real(Value::Tensor(tensor.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).unwrap();
        let gpu = block_on(log1p_gpu(handle)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(ct.shape, gt.shape);
                let tol = match runmat_accelerate_api::provider().unwrap().precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                    runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                };
                for (a, b) in gt.materialize_f64().iter().zip(ct.materialize_f64().iter()) {
                    assert!((a - b).abs() < tol, "|{} - {}| >= {}", a, b, tol);
                }
            }
            _ => panic!("unexpected value kinds"),
        }
    }
}
