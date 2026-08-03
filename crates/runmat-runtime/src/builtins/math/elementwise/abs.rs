//! MATLAB-compatible `abs` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, ComplexStorage, ComplexTensor, IntValue, IntegerStorage, NumericStorage, Tensor,
    Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::abs")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "abs",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_abs" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may execute abs on-device for real tensors and produce real magnitudes from complex-interleaved GPU tensors; the runtime gathers only when unary_abs is unavailable or host-only conversions are required.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::abs")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "abs",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!("abs({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL abs; providers can swap in specialised kernels.",
};

const BUILTIN_NAME: &str = "abs";

const ABS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Absolute value or magnitude.",
}];
const ABS_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric, logical, char, or complex input.",
}];
const ABS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = abs(X)",
    inputs: &ABS_INPUTS,
    outputs: &ABS_OUTPUT,
}];
const ABS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ABS.INVALID_INPUT",
    identifier: Some("RunMat:abs:InvalidInput"),
    when: "Input cannot be interpreted as numeric, logical, char, or complex data.",
    message: "abs: invalid input",
};
const ABS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ABS.INTERNAL",
    identifier: Some("RunMat:abs:Internal"),
    when: "Internal tensor conversion/allocation/provider interaction failed.",
    message: "abs: internal error",
};
const ABS_ERRORS: [BuiltinErrorDescriptor; 2] = [ABS_ERROR_INVALID_INPUT, ABS_ERROR_INTERNAL];
pub const ABS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ABS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ABS_ERRORS,
};

fn builtin_error_with_detail(
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
    name = "abs",
    category = "math/elementwise",
    summary = "Absolute value and complex magnitude for scalars and arrays.",
    keywords = "abs,absolute value,magnitude,complex,gpu",
    accel = "unary",
    type_resolver(numeric_unary_type),
    descriptor(crate::builtins::math::elementwise::abs::ABS_DESCRIPTOR),
    builtin_path = "crate::builtins::math::elementwise::abs"
)]
async fn abs_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => abs_gpu(handle).await,
        Value::Int(value) => Ok(Value::Int(abs_integer_scalar(value))),
        Value::Complex(re, im) => Ok(Value::Num(complex_magnitude(re, im))),
        Value::ComplexTensor(ct) => {
            crate::builtins::common::validation::reject_typed_complex_integer_tensor(&ct, "abs")?;
            abs_complex_tensor(ct)
        }
        Value::CharArray(ca) => abs_char_array(ca),
        Value::String(_) | Value::StringArray(_) => Err(builtin_error_with_detail(
            &ABS_ERROR_INVALID_INPUT,
            "expected numeric input",
        )),
        other => abs_real(other),
    }
}

async fn abs_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if runmat_accelerate_api::handle_integer_type(&handle).is_some() {
        let provider = runmat_accelerate_api::provider_for_handle(&handle).ok_or_else(|| {
            builtin_error_with_detail(
                &ABS_ERROR_INTERNAL,
                "GPU provider unavailable for integer input",
            )
        })?;
        let tensor = gpu_helpers::gather_tensor_async(&handle)
            .await
            .map_err(|err| builtin_error_with_detail(&ABS_ERROR_INTERNAL, err.to_string()))?;
        let output = abs_tensor(tensor)?;
        return match gpu_helpers::upload_tensor(provider, &output) {
            Ok(out) => Ok(Value::GpuTensor(out)),
            Err(_) => Ok(tensor::tensor_into_value(output)),
        };
    }
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.unary_abs(&handle).await {
            return Ok(Value::GpuTensor(out));
        }
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|err| builtin_error_with_detail(&ABS_ERROR_INTERNAL, err.to_string()))?;
    Ok(tensor::tensor_into_value(abs_tensor(tensor)?))
}

fn abs_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("abs", value)
        .map_err(|err| builtin_error_with_detail(&ABS_ERROR_INVALID_INPUT, err))?;
    Ok(tensor::tensor_into_value(abs_tensor(tensor)?))
}

fn abs_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|e| builtin_error_with_detail(&ABS_ERROR_INTERNAL, e))?;
    let output = match storage {
        NumericStorage::F64(values) => {
            NumericStorage::F64(values.into_iter().map(f64::abs).collect())
        }
        NumericStorage::F32(values) => {
            NumericStorage::F32(values.into_iter().map(f32::abs).collect())
        }
        integer => NumericStorage::from_integer_storage(abs_integer_storage(
            &integer
                .into_integer_storage()
                .expect("integer NumericStorage variant"),
        )),
    };
    Tensor::from_numeric_storage(output, shape)
        .map_err(|e| builtin_error_with_detail(&ABS_ERROR_INTERNAL, e))
}

fn abs_integer_scalar(value: IntValue) -> IntValue {
    match value {
        IntValue::I8(value) => IntValue::I8(value.saturating_abs()),
        IntValue::I16(value) => IntValue::I16(value.saturating_abs()),
        IntValue::I32(value) => IntValue::I32(value.saturating_abs()),
        IntValue::I64(value) => IntValue::I64(value.saturating_abs()),
        IntValue::U8(value) => IntValue::U8(value),
        IntValue::U16(value) => IntValue::U16(value),
        IntValue::U32(value) => IntValue::U32(value),
        IntValue::U64(value) => IntValue::U64(value),
    }
}

fn abs_integer_storage(storage: &IntegerStorage) -> IntegerStorage {
    match storage {
        IntegerStorage::I8(values) => {
            IntegerStorage::I8(values.iter().map(|value| value.saturating_abs()).collect())
        }
        IntegerStorage::I16(values) => {
            IntegerStorage::I16(values.iter().map(|value| value.saturating_abs()).collect())
        }
        IntegerStorage::I32(values) => {
            IntegerStorage::I32(values.iter().map(|value| value.saturating_abs()).collect())
        }
        IntegerStorage::I64(values) => {
            IntegerStorage::I64(values.iter().map(|value| value.saturating_abs()).collect())
        }
        IntegerStorage::U8(values) => IntegerStorage::U8(values.clone()),
        IntegerStorage::U16(values) => IntegerStorage::U16(values.clone()),
        IntegerStorage::U32(values) => IntegerStorage::U32(values.clone()),
        IntegerStorage::U64(values) => IntegerStorage::U64(values.clone()),
    }
}

fn abs_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let shape = ct.shape.clone();
    let storage = match ct.into_complex_storage() {
        ComplexStorage::F64(values) => NumericStorage::F64(
            values
                .into_iter()
                .map(|(real, imag)| complex_magnitude(real, imag))
                .collect(),
        ),
        ComplexStorage::F32(values) => NumericStorage::F32(
            values
                .into_iter()
                .map(|(real, imag)| real.hypot(imag))
                .collect(),
        ),
        ComplexStorage::Integer(_) => {
            return Err(builtin_error_with_detail(
                &ABS_ERROR_INVALID_INPUT,
                "typed complex integer input is not supported",
            ))
        }
    };
    let tensor = Tensor::from_numeric_storage(storage, shape)
        .map_err(|e| builtin_error_with_detail(&ABS_ERROR_INTERNAL, e))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn abs_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data = ca
        .data
        .iter()
        .map(|&ch| ch as u32 as f64)
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| builtin_error_with_detail(&ABS_ERROR_INTERNAL, e))?;
    Ok(tensor::tensor_into_value(tensor))
}

#[inline]
fn complex_magnitude(re: f64, im: f64) -> f64 {
    re.hypot(im)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;

    #[cfg(feature = "wgpu")]
    fn register_wgpu_provider_available() -> bool {
        runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_ok()
            && runmat_accelerate_api::provider().is_some()
    }
    use runmat_builtins::{IntValue, IntegerComplexStorage, ResolveContext, Tensor, Type};

    fn abs_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::abs_builtin(value))
    }

    #[test]
    fn abs_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = ABS_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = abs(X)"));
    }

    #[test]
    fn abs_type_preserves_tensor_shape() {
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
    fn abs_type_scalar_tensor_returns_num() {
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
    fn abs_scalar_negative() {
        let result = abs_builtin(Value::Num(-3.5)).expect("abs");
        match result {
            Value::Num(n) => assert!((n - 3.5).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn abs_integer_scalar_preserves_class_and_saturates_signed_minimum() {
        let result = abs_builtin(Value::Int(IntValue::I32(-8))).expect("abs");
        assert_eq!(result, Value::Int(IntValue::I32(8)));
        assert_eq!(
            abs_builtin(Value::Int(IntValue::I64(i64::MIN))).expect("abs"),
            Value::Int(IntValue::I64(i64::MAX))
        );
    }

    #[test]
    fn abs_preserves_native_single_and_rejects_typed_complex_integer() {
        let single = Tensor::from_f32(vec![-2.5, 0.0, 3.25], vec![1, 3]).unwrap();
        let output = abs_tensor(single).unwrap();
        assert_eq!(
            output.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![2.5, 0.0, 3.25])
        );

        let storage = IntegerComplexStorage::new(
            IntegerStorage::I16(vec![3, -4]),
            IntegerStorage::I16(vec![4, 3]),
        )
        .unwrap();
        let tensor = ComplexTensor::new_integer(storage, vec![1, 2]).unwrap();
        let error = abs_builtin(Value::ComplexTensor(tensor)).unwrap_err();
        assert!(error
            .message()
            .contains("complex numbers with integer types"));
    }

    #[test]
    fn abs_complex_single_preserves_native_class_shape_and_empty_storage() {
        let complex = ComplexTensor::from_f32(vec![(3.0, 4.0), (5.0, 12.0)], vec![2, 1]).unwrap();
        let Value::Tensor(output) = abs_builtin(Value::ComplexTensor(complex)).unwrap() else {
            panic!("expected single magnitude tensor");
        };
        assert_eq!(output.shape, vec![2, 1]);
        assert_eq!(
            output.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![5.0, 13.0])
        );

        let empty = ComplexTensor::from_f32(Vec::new(), vec![0, 3]).unwrap();
        let Value::Tensor(output) = abs_builtin(Value::ComplexTensor(empty)).unwrap() else {
            panic!("expected empty single magnitude tensor");
        };
        assert_eq!(output.shape, vec![0, 3]);
        assert_eq!(
            output.into_numeric_storage().unwrap(),
            NumericStorage::F32(Vec::new())
        );
    }

    #[test]
    fn abs_preserves_all_typed_integer_array_classes_exactly() {
        let cases = [
            (
                IntegerStorage::I8(vec![i8::MIN, -4, 0, i8::MAX]),
                IntegerStorage::I8(vec![i8::MAX, 4, 0, i8::MAX]),
            ),
            (
                IntegerStorage::I16(vec![i16::MIN, -4, 0, i16::MAX]),
                IntegerStorage::I16(vec![i16::MAX, 4, 0, i16::MAX]),
            ),
            (
                IntegerStorage::I32(vec![i32::MIN, -4, 0, i32::MAX]),
                IntegerStorage::I32(vec![i32::MAX, 4, 0, i32::MAX]),
            ),
            (
                IntegerStorage::I64(vec![i64::MIN, -4, 0, i64::MAX]),
                IntegerStorage::I64(vec![i64::MAX, 4, 0, i64::MAX]),
            ),
            (
                IntegerStorage::U8(vec![0, 4, u8::MAX]),
                IntegerStorage::U8(vec![0, 4, u8::MAX]),
            ),
            (
                IntegerStorage::U16(vec![0, 4, u16::MAX]),
                IntegerStorage::U16(vec![0, 4, u16::MAX]),
            ),
            (
                IntegerStorage::U32(vec![0, 4, u32::MAX]),
                IntegerStorage::U32(vec![0, 4, u32::MAX]),
            ),
            (
                IntegerStorage::U64(vec![0, 4, u64::MAX]),
                IntegerStorage::U64(vec![0, 4, u64::MAX]),
            ),
        ];
        for (input, expected) in cases {
            let input = Tensor::new_integer(input, vec![1, expected.len()]).expect("tensor");
            let Value::Tensor(result) = abs_builtin(Value::Tensor(input)).expect("abs") else {
                panic!("expected tensor");
            };
            assert_eq!(result.integer_storage(), Some(&expected));
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn abs_tensor_elements() {
        let tensor = Tensor::new(vec![-1.0, -2.0, 3.0, -4.0], vec![2, 2]).unwrap();
        let result = abs_builtin(Value::Tensor(tensor)).expect("abs");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.materialize_f64(), vec![1.0, 2.0, 3.0, 4.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn abs_complex_scalar() {
        let result = abs_builtin(Value::Complex(3.0, 4.0)).expect("abs");
        match result {
            Value::Num(n) => assert!((n - 5.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn abs_complex_tensor_to_real_tensor() {
        let complex = ComplexTensor::new(vec![(3.0, 4.0), (1.0, -1.0)], vec![2, 1]).unwrap();
        let result = abs_builtin(Value::ComplexTensor(complex)).expect("abs");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                assert!((t.materialize_f64()[0] - 5.0).abs() < 1e-12);
                assert!((t.materialize_f64()[1] - (2f64).sqrt()).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn abs_char_array_codes() {
        let char_array = CharArray::new("Az".chars().collect(), 1, 2).unwrap();
        let result = abs_builtin(Value::CharArray(char_array)).expect("abs");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.materialize_f64(), vec![65.0, 122.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn abs_string_rejected() {
        let err = abs_builtin(Value::from("hello")).expect_err("should error");
        let identifier = err.identifier().map(str::to_string);
        assert!(err.message().contains("expected numeric"));
        assert_eq!(identifier.as_deref(), ABS_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn abs_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![-2.0, -1.0, 0.0, 3.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = abs_builtin(Value::GpuTensor(handle)).expect("abs");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.materialize_f64(), vec![2.0, 1.0, 0.0, 3.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn abs_gpu_preserves_exact_integer_class_and_values() {
        test_support::with_test_provider(|provider| {
            let values = [i64::MIN, -9_007_199_254_740_993, 0, i64::MAX];
            let shape = [2usize, 2usize];
            let handle = provider
                .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                    data: runmat_accelerate_api::HostIntegerDataView::I64(&values),
                    shape: &shape,
                })
                .expect("upload integer gpu tensor");
            let result = abs_builtin(Value::GpuTensor(handle)).expect("abs");
            let Value::GpuTensor(ref output_handle) = result else {
                panic!("expected resident integer gpu tensor");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(output_handle),
                Some(runmat_accelerate_api::IntegerElementType::I64)
            );
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(
                gathered.integer_storage(),
                Some(&IntegerStorage::I64(vec![
                    i64::MAX,
                    9_007_199_254_740_993,
                    0,
                    i64::MAX,
                ]))
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn abs_complex_gpu_provider_stays_resident() {
        test_support::with_test_provider(|provider| {
            let complex = ComplexTensor::new(vec![(3.0, 4.0), (1.0, -1.0)], vec![2, 1]).unwrap();
            let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
            let result = abs_builtin(Value::GpuTensor(handle)).expect("abs");
            let Value::GpuTensor(out) = result else {
                panic!("expected gpu tensor");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&out),
                runmat_accelerate_api::GpuTensorStorage::Real
            );
            let gathered = test_support::gather(Value::GpuTensor(out)).expect("gather");
            assert_eq!(gathered.shape, vec![2, 1]);
            assert!((gathered.materialize_f64()[0] - 5.0).abs() < 1e-12);
            assert!((gathered.materialize_f64()[1] - (2f64).sqrt()).abs() < 1e-12);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn abs_wgpu_matches_cpu_elementwise() {
        let _guard = test_support::accel_test_lock();
        if !register_wgpu_provider_available() {
            return;
        }
        let tensor = Tensor::new(vec![-3.0, -1.0, 0.5, -0.25], vec![4, 1]).unwrap();
        let cpu = abs_real(Value::Tensor(tensor.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let h = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = block_on(abs_gpu(h)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(gt.shape, ct.shape);
                let tol = match runmat_accelerate_api::provider().unwrap().precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                    runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                };
                for (a, b) in gt.materialize_f64().iter().zip(ct.materialize_f64().iter()) {
                    assert!((*a - *b).abs() < tol, "|{} - {}| >= {}", a, b, tol);
                }
            }
            _ => panic!("unexpected result shape"),
        }
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn abs_wgpu_complex_matches_cpu() {
        let _guard = test_support::accel_test_lock();
        if !register_wgpu_provider_available() {
            return;
        }
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let complex = ComplexTensor::new(vec![(3.0, 4.0), (1.0, -1.0)], vec![2, 1]).unwrap();
        let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
        let gpu = block_on(abs_gpu(handle)).unwrap();
        let Value::GpuTensor(out) = gpu else {
            panic!("expected gpu tensor");
        };
        assert_eq!(
            runmat_accelerate_api::handle_storage(&out),
            runmat_accelerate_api::GpuTensorStorage::Real
        );
        let gathered = test_support::gather(Value::GpuTensor(out)).expect("gather");
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        assert!((gathered.materialize_f64()[0] - 5.0).abs() < tol);
        assert!((gathered.materialize_f64()[1] - (2f64).sqrt()).abs() < tol);
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn abs_wgpu_complex_preserves_infinity_with_nan_lane() {
        let _guard = test_support::accel_test_lock();
        if !register_wgpu_provider_available() {
            return;
        }
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let complex = ComplexTensor::new(
            vec![(f64::INFINITY, f64::NAN), (f64::NAN, f64::NEG_INFINITY)],
            vec![2, 1],
        )
        .unwrap();
        let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
        let gpu = block_on(abs_gpu(handle)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        assert_eq!(gathered.shape, vec![2, 1]);
        assert!(
            gathered.materialize_f64()[0].is_infinite()
                && gathered.materialize_f64()[0].is_sign_positive()
        );
        assert!(
            gathered.materialize_f64()[1].is_infinite()
                && gathered.materialize_f64()[1].is_sign_positive()
        );
    }
}
