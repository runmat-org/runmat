//! MATLAB-compatible `gammaln` builtin with GPU-aware semantics for RunMat.
//!
//! `gammaln` evaluates the natural logarithm of the gamma function for real,
//! nonnegative inputs. The CPU implementation uses a log-Lanczos form so large
//! arguments do not overflow through `log(gamma(x))`.

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
use runmat_value::{CharArray, NumericDType, NumericScalar, NumericStorage, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::dispatcher::download_handle_async;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "gammaln";
const PI: f64 = std::f64::consts::PI;
const LN_SQRT_TWO_PI: f64 = 0.918_938_533_204_672_7;
const LANCZOS_G: f64 = 7.0;
const SMALL_REFLECTION_CUTOFF: f64 = 1.0e-305;

const LANCZOS_COEFFS: [f64; 8] = [
    676.5203681218851,
    -1259.1392167224028,
    771.3234287776531,
    -176.6150291621406,
    12.507343278686905,
    -0.13857109526572012,
    9.984_369_578_019_572e-6,
    1.5056327351493116e-7,
];

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Natural logarithm of the gamma function.",
}];

const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real nonnegative numeric input.",
}];

const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = gammaln(A)",
    inputs: &INPUTS,
    outputs: &OUTPUT,
}];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GAMMALN.INVALID_INPUT",
    identifier: Some("RunMat:gammaln:InvalidInput"),
    when: "Input cannot be interpreted as real, nonsparse numeric data.",
    message: "gammaln: invalid input",
};

const ERROR_DOMAIN: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GAMMALN.DOMAIN",
    identifier: Some("RunMat:gammaln:Domain"),
    when: "At least one real input value is negative.",
    message: "gammaln: input must be nonnegative",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GAMMALN.INTERNAL",
    identifier: Some("RunMat:gammaln:Internal"),
    when: "Internal tensor construction or provider interaction failed.",
    message: "gammaln: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 3] = [ERROR_INVALID_INPUT, ERROR_DOMAIN, ERROR_INTERNAL];

const INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "gammaln-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "gammaln with typed-integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GammalnIntegerInputExtension"),
};
const LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "gammaln-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "gammaln with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GammalnLogicalInputExtension"),
};
const CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "gammaln-character-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "gammaln with character input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GammalnCharacterInputExtension"),
};
const EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    INTEGER_INPUT_EXTENSION,
    LOGICAL_INPUT_EXTENSION,
    CHARACTER_INPUT_EXTENSION,
];
const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "RunMat mode admits all eight integer classes only when every value is exactly representable at the binary64 log-gamma boundary.",
}];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = gammaln(integer_A)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Integer admission is checked before provider access; resident values gather through authoritative integer storage and produce floating output restored to the owning provider.",
    }];

pub const GAMMALN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::gammaln")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "gammaln",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Reduction { name: "reduce_min" },
        ProviderHook::Unary {
            name: "unary_gammaln",
        },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "RunMat uses provider gammaln kernels only after proving gpuArray inputs are nonnegative; otherwise it gathers to enforce MATLAB's real-domain input rule.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::gammaln")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "gammaln",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Acts as a fusion sink because negative inputs must raise a domain error instead of producing an elementwise NaN.",
};

#[runtime_builtin(
    name = "gammaln",
    category = "math/elementwise",
    summary = "Compute the natural logarithm of the gamma function.",
    keywords = "gammaln,gamma,log gamma,special,elementwise,gpu",
    accel = "unary",
    type_resolver(numeric_unary_type),
    descriptor(crate::builtins::math::elementwise::gammaln::GAMMALN_DESCRIPTOR),
    extensions(EXTENSIONS),
    integer_capabilities(crate::builtins::math::elementwise::gammaln::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::elementwise::gammaln"
)]
async fn gammaln_builtin(value: Value) -> BuiltinResult<Value> {
    ensure_gammaln_extensions(&value)?;
    match value {
        Value::GpuTensor(handle) => gammaln_gpu(handle).await,
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(error_with_detail(
            &ERROR_INVALID_INPUT,
            "complex input is not supported",
        )),
        Value::String(_) | Value::StringArray(_) => Err(error_with_detail(
            &ERROR_INVALID_INPUT,
            "expected real nonnegative numeric input",
        )),
        Value::SparseTensor(_) => Err(error_with_detail(
            &ERROR_INVALID_INPUT,
            "sparse input is not supported",
        )),
        Value::CharArray(chars) => gammaln_char_array(chars),
        other => gammaln_real(other),
    }
}

fn ensure_gammaln_extensions(value: &Value) -> BuiltinResult<()> {
    let integer = matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some());
    if integer {
        crate::compatibility::ensure_builtin_extension_enabled(
            &INTEGER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let logical = matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle));
    if logical {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LOGICAL_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if matches!(value, Value::CharArray(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CHARACTER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

async fn gammaln_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if runmat_accelerate_api::handle_storage(&handle) == GpuTensorStorage::ComplexInterleaved {
        return Err(error_with_detail(
            &ERROR_INVALID_INPUT,
            "complex gpuArray input is not supported",
        ));
    }

    let provider = runmat_accelerate_api::provider_for_handle(&handle);
    let requires_authoritative_host_path = runmat_accelerate_api::handle_integer_type(&handle)
        .is_some()
        || runmat_accelerate_api::handle_is_logical(&handle);
    if !requires_authoritative_host_path {
        if let Some(provider) = provider {
            match gpu_has_negative_input(provider, &handle).await {
                Ok(true) => {
                    return Err(error_with_detail(
                        &ERROR_DOMAIN,
                        "gpuArray contains negative values",
                    ))
                }
                Ok(false) => match provider.unary_gammaln(&handle).await {
                    Ok(out) if gammaln_native_output_matches(&handle, &out, provider) => {
                        return Ok(gpu_helpers::resident_gpu_value(out))
                    }
                    Ok(out) => {
                        free_rejected_gammaln_output(&out, &handle, provider);
                    }
                    Err(err) if is_unsupported_provider_hook(&err) => {}
                    Err(err) => {
                        return Err(error_with_detail(
                            &ERROR_INTERNAL,
                            format!("provider unary_gammaln failed: {err}"),
                        ))
                    }
                },
                Err(err) => {
                    if err.message() == "interaction pending..." {
                        return Err(err);
                    }
                    // Fall back to host evaluation when reduction proof is unavailable.
                }
            }
        }
    }

    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    let output = gammaln_tensor(tensor)?;
    if let Some(provider) = provider {
        return restore_gammaln_gpu_output(provider, &handle, output);
    }
    Ok(output)
}

fn gammaln_native_output_matches(
    input: &GpuTensorHandle,
    output: &GpuTensorHandle,
    provider: &dyn AccelProvider,
) -> bool {
    output.shape == input.shape
        && output.device_id == input.device_id
        && !gpu_handles_alias(output, input)
        && runmat_accelerate_api::handle_storage(output) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_integer_type(output).is_none()
        && !runmat_accelerate_api::handle_is_logical(output)
        && runmat_accelerate_api::handle_precision(output)
            == Some(
                runmat_accelerate_api::handle_precision(input)
                    .unwrap_or_else(|| provider.precision()),
            )
        && runmat_accelerate_api::provider_for_handle(output)
            .is_some_and(|owner| std::ptr::eq(owner, provider))
}

fn gpu_handles_alias(lhs: &GpuTensorHandle, rhs: &GpuTensorHandle) -> bool {
    lhs.device_id == rhs.device_id && lhs.buffer_id == rhs.buffer_id
}

fn free_rejected_gammaln_output(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &dyn AccelProvider,
) {
    if !gpu_handles_alias(output, input) {
        let owner = runmat_accelerate_api::provider_for_handle(output).unwrap_or(provider);
        let _ = owner.free(output);
    }
}

fn restore_gammaln_gpu_output(
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    value: Value,
) -> BuiltinResult<Value> {
    let tensor = match value {
        Value::Tensor(tensor) => tensor,
        Value::Num(value) => Tensor::new(vec![value], input.shape.clone())
            .map_err(|detail| error_with_detail(&ERROR_INTERNAL, detail))?,
        other => {
            return Err(error_with_detail(
                &ERROR_INTERNAL,
                format!("unexpected host fallback result {other:?}"),
            ))
        }
    };
    let dtype = tensor.numeric_dtype();
    let output = gpu_helpers::upload_tensor(provider, &tensor)
        .map_err(|detail| error_with_detail(&ERROR_INTERNAL, detail))?;
    if dtype == NumericDType::F32 {
        runmat_accelerate_api::set_handle_precision(
            &output,
            runmat_accelerate_api::ProviderPrecision::F32,
        );
    }
    if output.shape != input.shape
        || runmat_accelerate_api::provider_for_handle(&output)
            .is_none_or(|owner| !std::ptr::eq(owner, provider))
    {
        let _ = provider.free(&output);
        return Err(error_with_detail(
            &ERROR_INTERNAL,
            "provider upload returned malformed fallback output",
        ));
    }
    Ok(gpu_helpers::resident_gpu_value(output))
}

async fn gpu_has_negative_input(
    provider: &'static dyn AccelProvider,
    handle: &GpuTensorHandle,
) -> BuiltinResult<bool> {
    let min_handle = provider
        .reduce_min(handle)
        .await
        .map_err(|e| internal_error(format!("gammaln: reduce_min failed: {e}")))?;
    let download = download_handle_async(provider, &min_handle)
        .await
        .map_err(|e| internal_error(format!("gammaln: reduce_min download failed: {e}")));
    let _ = provider.free(&min_handle);
    let host = download?;
    Ok(host.data.iter().any(|&value| value < 0.0))
}

fn gammaln_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value)
        .map_err(|detail| error_with_detail(&ERROR_INVALID_INPUT, detail))?;
    gammaln_tensor(tensor)
}

fn gammaln_tensor(tensor: Tensor) -> BuiltinResult<Value> {
    if let Some(storage) = tensor.integer_storage() {
        for integer in storage.exact_values() {
            if !crate::builtins::math::trigonometry::cos::integer_is_exact_f64(&integer) {
                return Err(error_with_detail(
                    &ERROR_INVALID_INPUT,
                    "integer values must be exactly representable as double",
                ));
            }
        }
    }
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|detail| error_with_detail(&ERROR_INTERNAL, detail))?;
    let output = match storage {
        NumericStorage::F32(values) => {
            if values.iter().any(|&value| value < 0.0) {
                return Err(error_with_detail(
                    &ERROR_DOMAIN,
                    "input values must be nonnegative",
                ));
            }
            NumericStorage::F32(
                values
                    .into_iter()
                    .map(|value| gammaln_nonnegative_scalar(f64::from(value)) as f32)
                    .collect(),
            )
        }
        storage => {
            let values = storage.materialize_f64();
            ensure_nonnegative(&values)?;
            NumericStorage::F64(values.into_iter().map(gammaln_nonnegative_scalar).collect())
        }
    };
    let out = Tensor::from_numeric_storage(output, shape)
        .map_err(|detail| error_with_detail(&ERROR_INTERNAL, detail))?;
    Ok(gammaln_tensor_into_value(out))
}

fn gammaln_tensor_into_value(tensor: Tensor) -> Value {
    if tensor.len() == 1 && tensor.numeric_dtype() == NumericDType::F64 {
        if let Some(NumericScalar::F64(value)) = tensor.numeric_value_at(0) {
            return Value::Num(value);
        }
    }
    Value::Tensor(tensor)
}

fn gammaln_char_array(chars: CharArray) -> BuiltinResult<Value> {
    let data = chars
        .data
        .iter()
        .map(|&ch| gammaln_nonnegative_scalar(ch as u32 as f64))
        .collect::<Vec<_>>();
    let out = Tensor::new(data, vec![chars.rows, chars.cols])
        .map_err(|detail| error_with_detail(&ERROR_INTERNAL, detail))?;
    Ok(gammaln_tensor_into_value(out))
}

pub(crate) fn gammaln_nonnegative_scalar(value: f64) -> f64 {
    if value.is_nan() {
        return f64::NAN;
    }
    if value == 0.0 || value == f64::INFINITY {
        return f64::INFINITY;
    }
    if value < 0.0 {
        return f64::NAN;
    }
    if value < SMALL_REFLECTION_CUTOFF {
        return -value.ln();
    }
    if value < 0.5 {
        return PI.ln() - (PI * value).sin().ln() - lanczos_gammaln(1.0 - value);
    }
    lanczos_gammaln(value)
}

fn lanczos_gammaln(value: f64) -> f64 {
    let z_minus_one = value - 1.0;
    let mut sum = 0.999_999_999_999_809_9;
    for (idx, coeff) in LANCZOS_COEFFS.iter().enumerate() {
        sum += coeff / (z_minus_one + (idx + 1) as f64);
    }
    let t = z_minus_one + LANCZOS_G + 0.5;
    LN_SQRT_TWO_PI + (z_minus_one + 0.5) * t.ln() - t + sum.ln()
}

fn ensure_nonnegative(data: &[f64]) -> BuiltinResult<()> {
    if data.iter().any(|&value| value < 0.0) {
        Err(error_with_detail(
            &ERROR_DOMAIN,
            "input values must be nonnegative",
        ))
    } else {
        Ok(())
    }
}

fn is_unsupported_provider_hook(err: &anyhow::Error) -> bool {
    err.to_string().contains("unary_gammaln not supported")
}

fn internal_error(detail: impl std::fmt::Display) -> RuntimeError {
    error_with_detail(&ERROR_INTERNAL, detail)
}

fn error_with_detail(
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

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{ResolveContext, Type};
    use runmat_value::{ComplexTensor, IntValue, IntegerStorage, LogicalArray, SparseTensor};

    fn call(value: Value) -> BuiltinResult<Value> {
        block_on(gammaln_builtin(value))
    }

    fn approx_eq(got: f64, expected: f64, tol: f64) {
        assert!(
            (got - expected).abs() <= tol,
            "got {got}, expected {expected}, tol {tol}"
        );
    }

    fn values_f64(tensor: &Tensor) -> Vec<f64> {
        tensor.materialize_f64()
    }

    #[test]
    fn gammaln_descriptor_signature_covers_core_form() {
        let labels = GAMMALN_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect::<Vec<_>>();
        assert!(labels.contains(&"Y = gammaln(A)"));
    }

    #[test]
    fn gammaln_type_preserves_tensor_shape() {
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gammaln_scalar_values() {
        match call(Value::Num(1.0)).expect("gammaln") {
            Value::Num(v) => approx_eq(v, 0.0, 1e-14),
            other => panic!("expected scalar result, got {other:?}"),
        }
        match call(Value::Num(5.0)).expect("gammaln") {
            Value::Num(v) => approx_eq(v, 24.0_f64.ln(), 1e-13),
            other => panic!("expected scalar result, got {other:?}"),
        }
        match call(Value::Num(0.5)).expect("gammaln") {
            Value::Num(v) => approx_eq(v, std::f64::consts::PI.sqrt().ln(), 1e-14),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gammaln_avoids_overflow_for_large_values() {
        match call(Value::Num(171.0)).expect("gammaln") {
            Value::Num(v) => {
                assert!(v.is_finite());
                approx_eq(v, 706.573_062_245_787_5, 1e-10);
            }
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gammaln_tiny_positive_values_use_log_asymptote() {
        let tiny = f64::MIN_POSITIVE / 2.0;
        match call(Value::Num(tiny)).expect("gammaln") {
            Value::Num(v) => approx_eq(v, -tiny.ln(), 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gammaln_tensor_shape_and_single_dtype() {
        let tensor =
            Tensor::new_with_dtype(vec![0.5, 1.0, 2.0, 5.0], vec![2, 2], NumericDType::F32)
                .unwrap();
        let result = call(Value::Tensor(tensor)).expect("gammaln");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.numeric_dtype(), NumericDType::F32);
                let values = values_f64(&t);
                approx_eq(values[0], std::f32::consts::PI.sqrt().ln() as f64, 1e-7);
                approx_eq(values[1], 0.0, 1e-7);
                approx_eq(values[2], 0.0, 1e-7);
                approx_eq(values[3], 24.0_f32.ln() as f64, 1e-6);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gammaln_integer_bool_logical_and_char_promote() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        match call(Value::Int(IntValue::I32(5))).expect("gammaln") {
            Value::Num(v) => approx_eq(v, 24.0_f64.ln(), 1e-13),
            other => panic!("expected scalar result, got {other:?}"),
        }
        match call(Value::Bool(true)).expect("gammaln") {
            Value::Num(v) => approx_eq(v, 0.0, 1e-14),
            other => panic!("expected scalar result, got {other:?}"),
        }

        let logical = LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap();
        match call(Value::LogicalArray(logical)).expect("gammaln") {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                let values = values_f64(&t);
                approx_eq(values[0], 0.0, 1e-14);
                assert_eq!(values[1], f64::INFINITY);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }

        let chars = CharArray::new(vec!['\0', '\u{1}'], 1, 2).unwrap();
        match call(Value::CharArray(chars)).expect("gammaln") {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                let values = values_f64(&t);
                assert_eq!(values[0], f64::INFINITY);
                approx_eq(values[1], 0.0, 1e-14);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gammaln_reads_typed_integer_storage() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let scalar =
            Tensor::new_integer(IntegerStorage::U16(vec![5]), vec![1, 1]).expect("int tensor");
        match call(Value::Tensor(scalar)).expect("gammaln") {
            Value::Num(v) => approx_eq(v, 24.0_f64.ln(), 1e-13),
            other => panic!("expected scalar result, got {other:?}"),
        }

        let tensor =
            Tensor::new_integer(IntegerStorage::U16(vec![5, 1]), vec![1, 2]).expect("int tensor");
        match call(Value::Tensor(tensor)).expect("gammaln") {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.numeric_dtype(), NumericDType::F64);
                let values = values_f64(&t);
                approx_eq(values[0], 24.0_f64.ln(), 1e-13);
                approx_eq(values[1], 0.0, 1e-14);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn gammaln_integer_and_logical_extensions_are_independently_gated() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let integer = call(Value::Int(IntValue::I16(5))).unwrap_err();
        assert_eq!(
            integer.identifier(),
            Some("RunMat:compatibility:GammalnIntegerInputExtension")
        );
        let logical = call(Value::Bool(true)).unwrap_err();
        assert_eq!(
            logical.identifier(),
            Some("RunMat:compatibility:GammalnLogicalInputExtension")
        );
        let character = call(Value::CharArray(
            CharArray::new(vec!['A'], 1, 1).expect("character"),
        ))
        .unwrap_err();
        assert_eq!(
            character.identifier(),
            Some("RunMat:compatibility:GammalnCharacterInputExtension")
        );
    }

    #[test]
    fn gammaln_integer_extension_covers_all_classes_and_exact_binary64_boundary() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let storages = [
            IntegerStorage::I8(vec![1, 5]),
            IntegerStorage::I16(vec![1, 5]),
            IntegerStorage::I32(vec![1, 5]),
            IntegerStorage::I64(vec![1, 5]),
            IntegerStorage::U8(vec![1, 5]),
            IntegerStorage::U16(vec![1, 5]),
            IntegerStorage::U32(vec![1, 5]),
            IntegerStorage::U64(vec![1, 5]),
        ];
        for storage in storages {
            let tensor = Tensor::new_integer(storage, vec![1, 2]).unwrap();
            let Value::Tensor(output) = call(Value::Tensor(tensor)).expect("integer gammaln")
            else {
                panic!("expected tensor output")
            };
            assert_eq!(output.numeric_dtype(), NumericDType::F64);
        }

        let exact =
            Tensor::new_integer(IntegerStorage::U64(vec![1_u64 << 54]), vec![1, 1]).unwrap();
        call(Value::Tensor(exact)).expect("exact wide power of two");
        let inexact =
            Tensor::new_integer(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]).unwrap();
        let error = call(Value::Tensor(inexact)).unwrap_err();
        assert!(error.message().contains("exactly representable as double"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gammaln_nan_zero_and_infinity() {
        match call(Value::Num(0.0)).expect("gammaln") {
            Value::Num(v) => assert_eq!(v, f64::INFINITY),
            other => panic!("expected scalar result, got {other:?}"),
        }
        match call(Value::Num(f64::INFINITY)).expect("gammaln") {
            Value::Num(v) => assert_eq!(v, f64::INFINITY),
            other => panic!("expected scalar result, got {other:?}"),
        }
        match call(Value::Num(f64::NAN)).expect("gammaln") {
            Value::Num(v) => assert!(v.is_nan()),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gammaln_rejects_negative_complex_string_and_sparse_inputs() {
        let err = call(Value::Num(-0.5)).expect_err("negative should error");
        assert_eq!(err.identifier(), ERROR_DOMAIN.identifier);

        let err = call(Value::Complex(1.0, 1.0)).expect_err("complex should error");
        assert_eq!(err.identifier(), ERROR_INVALID_INPUT.identifier);

        let complex = ComplexTensor::new(vec![(1.0, 0.0)], vec![1, 1]).unwrap();
        let err = call(Value::ComplexTensor(complex)).expect_err("complex should error");
        assert_eq!(err.identifier(), ERROR_INVALID_INPUT.identifier);

        let err = call(Value::from("1")).expect_err("string should error");
        assert_eq!(err.identifier(), ERROR_INVALID_INPUT.identifier);

        let sparse = SparseTensor::zeros(2, 2);
        let err = call(Value::SparseTensor(sparse)).expect_err("sparse should error");
        assert_eq!(err.identifier(), ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gammaln_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.5, 1.0, 2.0, 5.0, 171.0], vec![1, 5]).unwrap();
            let view = HostTensorView {
                data: tensor.as_f64_slice().expect("double input"),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = call(Value::GpuTensor(handle)).expect("gammaln");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![1, 5]);
            for (got, input) in gathered
                .materialize_f64()
                .iter()
                .zip(tensor.as_f64_slice().expect("double input"))
            {
                approx_eq(*got, gammaln_nonnegative_scalar(*input), 1e-10);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gammaln_gpu_negative_errors() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, -0.5], vec![1, 2]).unwrap();
            let view = HostTensorView {
                data: tensor.as_f64_slice().expect("double input"),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let err = call(Value::GpuTensor(handle)).expect_err("negative gpu should error");
            assert_eq!(err.identifier(), ERROR_DOMAIN.identifier);
        });
    }

    #[test]
    fn gammaln_resident_integer_gate_precedes_provider_and_restores_double_output() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new_integer(IntegerStorage::U16(vec![1, 5]), vec![1, 2])
                .expect("integer input");
            let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
                .expect("integer upload");
            {
                let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
                let error = call(Value::GpuTensor(handle.clone())).unwrap_err();
                assert_eq!(
                    error.identifier(),
                    Some("RunMat:compatibility:GammalnIntegerInputExtension")
                );
                assert!(runmat_accelerate_api::provider_for_handle(&handle)
                    .is_some_and(|owner| std::ptr::eq(owner, provider)));
            }
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            let Value::GpuTensor(output) = call(Value::GpuTensor(handle)).expect("gammaln") else {
                panic!("expected resident output")
            };
            assert!(runmat_accelerate_api::handle_integer_type(&output).is_none());
            assert!(runmat_accelerate_api::provider_for_handle(&output)
                .is_some_and(|owner| std::ptr::eq(owner, provider)));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn gammaln_wgpu_matches_cpu_elementwise() {
        if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_err()
        {
            return;
        }
        let tensor = Tensor::new(vec![0.25, 0.5, 1.0, 2.0, 5.0, 32.0, 171.0], vec![1, 7]).unwrap();
        let cpu = match gammaln_tensor(tensor.clone()).expect("cpu gammaln") {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected tensor result, got {other:?}"),
        };
        let Some(provider) = runmat_accelerate_api::provider() else {
            return;
        };
        let view = HostTensorView {
            data: tensor.as_f64_slice().expect("double input"),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_value = block_on(gammaln_gpu(handle)).expect("gpu gammaln");
        let gathered = test_support::gather(gpu_value).expect("gather");
        assert_eq!(gathered.shape, cpu.shape);
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-9,
            runmat_accelerate_api::ProviderPrecision::F32 => 2e-4,
        };
        for (got, expected) in gathered
            .materialize_f64()
            .iter()
            .zip(cpu.as_f64_slice().expect("double cpu result"))
        {
            approx_eq(*got, *expected, tol);
        }
    }
}
