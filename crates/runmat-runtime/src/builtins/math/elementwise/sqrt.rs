//! MATLAB-compatible `sqrt` builtin with GPU-aware semantics for RunMat.
//!
//! Provides element-wise square roots for real, logical, character, and complex inputs while
//! preserving MATLAB semantics. Negative real values promote to complex outputs. GPU execution
//! utilises provider hooks when available and falls back to host computation whenever complex
//! results are required or the provider lacks the dedicated kernel.

use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, ComplexStorage, ComplexTensor, NumericStorage, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::builtins::math::symbolic::symbolic_function;
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::dispatcher::download_handle_async;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::SymbolicFunction;

const ZERO_EPS: f64 = 1e-12;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::sqrt")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "sqrt",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_sqrt" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers execute sqrt directly on device buffers when inputs are non-negative; runtime gathers to host when complex promotion is required.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::sqrt")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "sqrt",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx
                .inputs
                .first()
                .ok_or(FusionError::MissingInput(0))?;
            Ok(format!("sqrt({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL sqrt calls; providers may replace them with fused elementwise kernels.",
};

const BUILTIN_NAME: &str = "sqrt";

const SQRT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elementwise square-root result.",
}];
const SQRT_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric, logical, char, or complex input.",
}];
const SQRT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = sqrt(X)",
    inputs: &SQRT_INPUTS,
    outputs: &SQRT_OUTPUT,
}];
const SQRT_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SQRT.INVALID_INPUT",
    identifier: Some("RunMat:sqrt:InvalidInput"),
    when: "Input cannot be interpreted as numeric, logical, char, or complex data.",
    message: "sqrt: invalid input",
};
const SQRT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SQRT.INTERNAL",
    identifier: Some("RunMat:sqrt:Internal"),
    when: "Internal tensor construction or provider interaction failed.",
    message: "sqrt: internal error",
};
const SQRT_ERRORS: [BuiltinErrorDescriptor; 2] = [SQRT_ERROR_INVALID_INPUT, SQRT_ERROR_INTERNAL];
pub const SQRT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SQRT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SQRT_ERRORS,
};

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn sqrt_error_with_detail(
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
    name = "sqrt",
    category = "math/elementwise",
    summary = "Compute principal square roots element-wise across array inputs.",
    keywords = "sqrt,square root,elementwise,gpu,complex",
    accel = "unary",
    type_resolver(numeric_unary_type),
    descriptor(crate::builtins::math::elementwise::sqrt::SQRT_DESCRIPTOR),
    builtin_path = "crate::builtins::math::elementwise::sqrt"
)]
async fn sqrt_builtin(value: Value) -> BuiltinResult<Value> {
    if let Some(symbolic) = symbolic_function(&value, SymbolicFunction::Sqrt) {
        return Ok(symbolic);
    }
    match value {
        Value::GpuTensor(handle) => sqrt_gpu(handle).await,
        Value::Complex(re, im) => Ok(sqrt_complex_value(re, im)),
        Value::ComplexTensor(ct) => {
            crate::builtins::common::validation::reject_typed_complex_integer_tensor(&ct, "sqrt")?;
            sqrt_complex_tensor(ct)
        }
        Value::CharArray(ca) => sqrt_char_array(ca),
        Value::String(_) | Value::StringArray(_) => Err(sqrt_error_with_detail(
            &SQRT_ERROR_INVALID_INPUT,
            "expected numeric input",
        )),
        other => sqrt_real(other),
    }
}

async fn sqrt_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if runmat_accelerate_api::handle_integer_type(&handle).is_some() {
        let tensor = gpu_helpers::gather_tensor_async(&handle)
            .await
            .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
        return sqrt_tensor_real(tensor);
    }
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        match detect_gpu_requires_complex(provider, &handle).await {
            Ok(false) => {
                if let Ok(out) = provider.unary_sqrt(&handle).await {
                    return Ok(gpu_helpers::resident_gpu_value(out));
                }
            }
            Ok(true) => {
                let tensor = gpu_helpers::gather_tensor_async(&handle)
                    .await
                    .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
                return sqrt_tensor_real(tensor);
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
    sqrt_tensor_real(tensor)
}

async fn detect_gpu_requires_complex(
    provider: &'static dyn AccelProvider,
    handle: &GpuTensorHandle,
) -> BuiltinResult<bool> {
    let min_handle = provider
        .reduce_min(handle)
        .await
        .map_err(|e| builtin_error(format!("sqrt: reduce_min failed: {e}")))?;
    let download = download_handle_async(provider, &min_handle)
        .await
        .map_err(|e| builtin_error(format!("sqrt: reduce_min download failed: {e}")));
    let _ = provider.free(&min_handle);
    let host = download?;
    if host.data.iter().any(|&v| v.is_nan()) {
        return Err(builtin_error("sqrt: reduce_min result contained NaN"));
    }
    Ok(host.data.iter().any(|&v| v < 0.0))
}

fn sqrt_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("sqrt", value)
        .map_err(|e| builtin_error(format!("sqrt: {e}")))?;
    sqrt_tensor_real(tensor)
}

fn sqrt_tensor_real(tensor: Tensor) -> BuiltinResult<Value> {
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|e| builtin_error(format!("sqrt: {e}")))?;
    match storage {
        NumericStorage::F64(values) => sqrt_real_f64_values(values, shape),
        NumericStorage::F32(values) => sqrt_real_f32_values(values, shape),
        storage => sqrt_real_f64_values(promote_integer_storage_to_sqrt_domain(storage), shape),
    }
}

fn sqrt_real_f64_values(values: Vec<f64>, shape: Vec<usize>) -> BuiltinResult<Value> {
    let len = values.len();
    let requires_complex = values.iter().any(|&value| value < 0.0);
    if !requires_complex {
        let values = values
            .into_iter()
            .map(|value| zero_small(value.sqrt()))
            .collect();
        let tensor = Tensor::from_numeric_storage(NumericStorage::F64(values), shape)
            .map_err(|e| builtin_error(format!("sqrt: {e}")))?;
        Ok(tensor::tensor_into_value(tensor))
    } else {
        let mut data = Vec::with_capacity(len);
        for v in values {
            if v < 0.0 {
                let imag = zero_small((-v).sqrt());
                data.push((0.0, imag));
            } else {
                let real = zero_small(v.sqrt());
                data.push((real, 0.0));
            }
        }
        let tensor = ComplexTensor::from_complex_storage(ComplexStorage::F64(data), shape)
            .map_err(|e| builtin_error(format!("sqrt: {e}")))?;
        Ok(complex_tensor_into_value(tensor))
    }
}

fn sqrt_real_f32_values(values: Vec<f32>, shape: Vec<usize>) -> BuiltinResult<Value> {
    let requires_complex = values.iter().any(|&value| value < 0.0);
    if !requires_complex {
        let values = values
            .into_iter()
            .map(|value| zero_small_f32(value.sqrt()))
            .collect();
        let tensor = Tensor::from_numeric_storage(NumericStorage::F32(values), shape)
            .map_err(|e| builtin_error(format!("sqrt: {e}")))?;
        Ok(tensor::tensor_into_value(tensor))
    } else {
        let values = values
            .into_iter()
            .map(|value| {
                if value < 0.0 {
                    (0.0, zero_small_f32((-value).sqrt()))
                } else {
                    (zero_small_f32(value.sqrt()), 0.0)
                }
            })
            .collect();
        let tensor = ComplexTensor::from_complex_storage(ComplexStorage::F32(values), shape)
            .map_err(|e| builtin_error(format!("sqrt: {e}")))?;
        Ok(complex_tensor_into_value(tensor))
    }
}

fn sqrt_complex_value(re: f64, im: f64) -> Value {
    let (mut real_part, mut imag_part) = sqrt_complex_parts(re, im);
    real_part = zero_small(real_part);
    imag_part = zero_small(imag_part);
    Value::Complex(real_part, imag_part)
}

fn sqrt_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let shape = ct.shape.clone();
    let storage = match ct.into_complex_storage() {
        ComplexStorage::F64(values) => ComplexStorage::F64(
            values
                .into_iter()
                .map(|(real, imag)| {
                    let (real, imag) = sqrt_complex_parts(real, imag);
                    (zero_small(real), zero_small(imag))
                })
                .collect(),
        ),
        ComplexStorage::F32(values) => ComplexStorage::F32(
            values
                .into_iter()
                .map(|(real, imag)| {
                    let (real, imag) = sqrt_complex_parts_f32(real, imag);
                    (zero_small_f32(real), zero_small_f32(imag))
                })
                .collect(),
        ),
        ComplexStorage::Integer(_) => {
            return Err(sqrt_error_with_detail(
                &SQRT_ERROR_INVALID_INPUT,
                "typed complex integer input is not supported",
            ))
        }
    };
    let tensor = ComplexTensor::from_complex_storage(storage, shape)
        .map_err(|e| builtin_error(format!("sqrt: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

fn promote_integer_storage_to_sqrt_domain(storage: NumericStorage) -> Vec<f64> {
    storage
        .into_integer_storage()
        .expect("sqrt integer-promotion boundary received floating storage")
        .to_f64_vec()
}

fn sqrt_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let mut data = Vec::with_capacity(ca.data.len());
    for &ch in &ca.data {
        let code = ch as u32 as f64;
        data.push(zero_small(code.sqrt()));
    }
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| builtin_error(format!("sqrt: {e}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn sqrt_complex_parts(re: f64, im: f64) -> (f64, f64) {
    if im == 0.0 {
        if re < 0.0 {
            (0.0, (-re).sqrt())
        } else {
            (re.sqrt(), 0.0)
        }
    } else {
        let magnitude = re.hypot(im);
        if magnitude == 0.0 {
            (0.0, 0.0)
        } else {
            let real_part = ((magnitude + re) / 2.0).sqrt();
            let imag_part_raw = ((magnitude - re) / 2.0).sqrt();
            let imag_part = if im >= 0.0 {
                imag_part_raw
            } else {
                -imag_part_raw
            };
            (real_part, imag_part)
        }
    }
}

fn sqrt_complex_parts_f32(re: f32, im: f32) -> (f32, f32) {
    if im == 0.0 {
        if re < 0.0 {
            (0.0, (-re).sqrt())
        } else {
            (re.sqrt(), 0.0)
        }
    } else {
        let magnitude = re.hypot(im);
        if magnitude == 0.0 {
            (0.0, 0.0)
        } else {
            let real_part = ((magnitude + re) / 2.0).sqrt();
            let imag_part_raw = ((magnitude - re) / 2.0).sqrt();
            let imag_part = if im >= 0.0 {
                imag_part_raw
            } else {
                -imag_part_raw
            };
            (real_part, imag_part)
        }
    }
}

fn zero_small(value: f64) -> f64 {
    if value.abs() < ZERO_EPS {
        0.0
    } else {
        value
    }
}

fn zero_small_f32(value: f32) -> f32 {
    if value.abs() < ZERO_EPS as f32 {
        0.0
    } else {
        value
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{
        CharArray, IntValue, IntegerStorage, LogicalArray, ResolveContext, Tensor, Type,
    };

    fn sqrt_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::sqrt_builtin(value))
    }

    #[test]
    fn sqrt_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = SQRT_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = sqrt(X)"));
    }

    #[test]
    fn sqrt_string_rejected_with_stable_identifier() {
        let err = sqrt_builtin(Value::from("bad")).expect_err("expected input error");
        assert_eq!(err.identifier(), SQRT_ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn sqrt_type_preserves_tensor_shape() {
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
    fn sqrt_type_scalar_tensor_returns_num() {
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
    fn sqrt_scalar_positive() {
        let result = sqrt_builtin(Value::Num(9.0)).expect("sqrt");
        match result {
            Value::Num(v) => assert!((v - 3.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sqrt_scalar_negative() {
        let result = sqrt_builtin(Value::Num(-4.0)).expect("sqrt");
        match result {
            Value::Complex(re, im) => {
                assert!(re.abs() < 1e-12);
                assert!((im - 2.0).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sqrt_bool_true() {
        let result = sqrt_builtin(Value::Bool(true)).expect("sqrt");
        match result {
            Value::Num(v) => assert!((v - 1.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sqrt_logical_array_inputs() {
        let logical = LogicalArray::new(vec![1u8, 0, 1, 0], vec![2, 2]).expect("logical");
        let result = sqrt_builtin(Value::LogicalArray(logical)).expect("sqrt");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert!((t.materialize_f64()[0] - 1.0).abs() < 1e-12);
                assert!(t.materialize_f64()[1].abs() < 1e-12);
                assert!((t.materialize_f64()[2] - 1.0).abs() < 1e-12);
                assert!(t.materialize_f64()[3].abs() < 1e-12);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sqrt_tensor_with_negatives() {
        let tensor = Tensor::new(vec![-1.0, 4.0], vec![1, 2]).unwrap();
        let result = sqrt_builtin(Value::Tensor(tensor)).expect("sqrt");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![1, 2]);
                assert!(ct.materialize_f64()[0].0.abs() < 1e-12);
                assert!((ct.materialize_f64()[0].1 - 1.0).abs() < 1e-12);
                assert!((ct.materialize_f64()[1].0 - 2.0).abs() < 1e-12);
                assert!(ct.materialize_f64()[1].1.abs() < 1e-12);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sqrt_char_array_inputs() {
        let chars = CharArray::new("AZ".chars().collect(), 1, 2).unwrap();
        let result = sqrt_builtin(Value::CharArray(chars)).expect("sqrt");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert!((t.materialize_f64()[0] - (65.0f64).sqrt()).abs() < 1e-12);
                assert!((t.materialize_f64()[1] - (90.0f64).sqrt()).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sqrt_string_input_errors() {
        let err = sqrt_builtin(Value::from("hello")).unwrap_err();
        assert_eq!(err.identifier(), SQRT_ERROR_INVALID_INPUT.identifier);
        assert!(err.message().contains("expected numeric input"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sqrt_complex_scalar() {
        let result = sqrt_builtin(Value::Complex(3.0, 4.0)).expect("sqrt");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 2.0).abs() < 1e-12);
                assert!((im - 1.0).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sqrt_integer_argument() {
        let result = sqrt_builtin(Value::Int(IntValue::I32(9))).expect("sqrt");
        match result {
            Value::Num(v) => assert!((v - 3.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sqrt_reads_typed_integer_tensor_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::U32(vec![0, 4, 9]), vec![3, 1])
            .expect("integer tensor");

        let result = sqrt_builtin(Value::Tensor(tensor)).expect("sqrt");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                assert_eq!(out.materialize_f64(), vec![0.0, 2.0, 3.0]);
                assert!(out.integer_storage().is_none());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sqrt_negative_typed_integer_tensor_promotes_to_complex_from_storage() {
        let tensor = Tensor::new_integer(IntegerStorage::I32(vec![-4, 9]), vec![1, 2])
            .expect("integer tensor");

        let result = sqrt_builtin(Value::Tensor(tensor)).expect("sqrt");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert_eq!(out.materialize_f64()[0], (0.0, 2.0));
                assert_eq!(out.materialize_f64()[1], (3.0, 0.0));
            }
            other => panic!("expected complex tensor result, got {other:?}"),
        }
    }

    #[test]
    fn sqrt_preserves_native_single_real_complex_negative_and_empty_storage() {
        let tensor = Tensor::from_f32(vec![0.0, 4.0], vec![2, 1]).unwrap();
        let Value::Tensor(output) = sqrt_builtin(Value::Tensor(tensor)).expect("sqrt") else {
            panic!("expected single real tensor");
        };
        assert_eq!(
            output.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![0.0, 2.0])
        );

        let tensor = Tensor::from_f32(vec![-4.0, 9.0], vec![1, 2]).unwrap();
        let Value::ComplexTensor(output) = sqrt_builtin(Value::Tensor(tensor)).expect("sqrt")
        else {
            panic!("expected complex single tensor");
        };
        assert_eq!(output.as_f32_slice(), Some(&[(0.0, 2.0), (3.0, 0.0)][..]));

        let complex = ComplexTensor::from_f32(vec![(3.0, 4.0)], vec![1, 1]).unwrap();
        let Value::ComplexTensor(output) =
            sqrt_builtin(Value::ComplexTensor(complex)).expect("sqrt")
        else {
            panic!("one-element complex single must retain class");
        };
        assert_eq!(
            output.as_f32_slice(),
            Some(&[sqrt_complex_parts_f32(3.0, 4.0)][..])
        );

        let empty = ComplexTensor::from_f32(Vec::new(), vec![0, 2]).unwrap();
        let Value::ComplexTensor(output) = sqrt_builtin(Value::ComplexTensor(empty)).expect("sqrt")
        else {
            panic!("expected empty complex single tensor");
        };
        assert_eq!(output.shape, vec![0, 2]);
        assert_eq!(output.as_f32_slice(), Some(&[][..]));
    }

    #[test]
    fn sqrt_integer_gpu_gathers_exact_storage_before_floating_domain() {
        test_support::with_test_provider(|provider| {
            let wide = 9_007_199_254_740_993_u64;
            let tensor =
                Tensor::new_integer(IntegerStorage::U64(vec![0, wide]), vec![1, 2]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let Value::Tensor(output) = sqrt_builtin(Value::GpuTensor(handle)).expect("sqrt")
            else {
                panic!("expected host double tensor");
            };
            assert_eq!(
                output.into_numeric_storage().unwrap(),
                NumericStorage::F64(vec![0.0, (wide as f64).sqrt()])
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sqrt_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, 4.0, 9.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = sqrt_builtin(Value::GpuTensor(handle)).expect("sqrt");
            let gathered = test_support::gather(result).expect("gather");
            let expected: Vec<f64> = tensor.materialize_f64().iter().map(|&v| v.sqrt()).collect();
            assert_eq!(gathered.shape, vec![4, 1]);
            for (gpu, cpu) in gathered.materialize_f64().iter().zip(expected.iter()) {
                assert!((gpu - cpu).abs() < 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sqrt_gpu_negative_falls_back_to_complex() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![-1.0, 9.0], vec![1, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = sqrt_builtin(Value::GpuTensor(handle)).expect("sqrt");
            match result {
                Value::ComplexTensor(ct) => {
                    assert_eq!(ct.shape, vec![1, 2]);
                    assert!(ct.materialize_f64()[0].0.abs() < 1e-12);
                    assert!((ct.materialize_f64()[0].1 - 1.0).abs() < 1e-12);
                }
                other => panic!("expected complex tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn sqrt_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![0.0, 1.0, 4.0, 9.0], vec![4, 1]).unwrap();
        let cpu = sqrt_real(Value::Tensor(tensor.clone())).expect("cpu sqrt");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .expect("upload");
        let gpu_value = block_on(sqrt_gpu(handle)).expect("gpu sqrt");
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
