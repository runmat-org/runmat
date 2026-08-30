//! MATLAB-compatible `imag` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
#[cfg(test)]
use runmat_builtins::IMAG_DESCRIPTOR;
use runmat_builtins::{BuiltinErrorDescriptor, IMAG_ERROR_INTERNAL, IMAG_ERROR_INVALID_INPUT};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, ComplexStorage, ComplexTensor, NumericStorage, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::imag")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "imag",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_imag" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    two_pass_threshold: None,
    workgroup_size: None,
    nan_mode: ReductionNaN::Include,
    accepts_nan_mode: false,
    notes: "Providers may implement unary_imag to materialise zero tensors for real inputs or extract imaginary lanes from complex-interleaved GPU tensors; the runtime gathers only when the hook is absent or host-only conversions are required.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::imag")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "imag",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let literal = match ctx.scalar_ty {
                ScalarType::F32 => "0.0".to_string(),
                ScalarType::F64 => "f64(0.0)".to_string(),
                other => return Err(FusionError::UnsupportedPrecision(other)),
            };
            Ok(literal)
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion kernels treat imag as a zero-producing transform for real tensors; providers can override via fused pipelines to keep tensors resident on the GPU.",
};

const BUILTIN_NAME: &str = "imag";

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
    name = "imag",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::imag"
)]
async fn imag_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => imag_gpu(handle).await,
        Value::Complex(_, im) => Ok(Value::Num(im)),
        Value::ComplexTensor(ct) => imag_complex_tensor(ct),
        Value::CharArray(ca) => imag_char_array(ca),
        Value::String(_) | Value::StringArray(_) => Err(builtin_error_with_detail(
            &IMAG_ERROR_INVALID_INPUT,
            "expected numeric input",
        )),
        x @ (Value::Tensor(_)
        | Value::LogicalArray(_)
        | Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)) => imag_real(x),
        other => Err(builtin_error_with_detail(
            &IMAG_ERROR_INVALID_INPUT,
            format!(
                "unsupported input type {:?}; expected numeric, logical, or char input",
                other
            ),
        )),
    }
}

async fn imag_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = gpu_helpers::exact_provider_for_handle(&handle).ok_or_else(|| {
        builtin_error_with_detail(&IMAG_ERROR_INTERNAL, "GPU provider unavailable for input")
    })?;
    let input_metadata = gpu_helpers::snapshot_handle_metadata(&handle);
    let input_provenance = runmat_accelerate_api::handle_provenance(&handle)
        .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic);
    let exact_host_path = runmat_accelerate_api::handle_integer_type(&handle).is_some()
        || runmat_accelerate_api::handle_is_logical(&handle);
    if gpu_helpers::expected_handle_numeric_element_type(&handle).is_err() {
        return Err(builtin_error_with_detail(
            &IMAG_ERROR_INTERNAL,
            "GPU input class metadata contradicts its physical storage",
        ));
    }
    let kernel_compatible =
        runmat_accelerate_api::handle_precision(&handle) == Some(provider.precision());
    if !exact_host_path && kernel_compatible {
        let result = provider.unary_imag(&handle).await;
        gpu_helpers::restore_handle_metadata(&handle, &input_metadata);
        match result {
            Ok(mut out) if valid_imag_gpu_output(&out, &handle, provider) => {
                runmat_accelerate_api::set_handle_provenance(&mut out, input_provenance);
                return Ok(gpu_helpers::resident_gpu_value(out));
            }
            Ok(out) => {
                gpu_helpers::free_unprotected_exact_owner(&out, &[&handle]);
                return Err(builtin_error_with_detail(
                    &IMAG_ERROR_INTERNAL,
                    "provider unary_imag returned malformed output",
                ));
            }
            Err(err) if err.to_string().contains("unary_imag not supported") => {}
            Err(err) => {
                return Err(builtin_error_with_detail(
                    &IMAG_ERROR_INTERNAL,
                    format!("provider unary_imag failed: {err}"),
                ));
            }
        }
    }
    let gathered_result =
        gpu_helpers::download_value_preserving_residency_async(provider, &handle).await;
    gpu_helpers::restore_handle_metadata(&handle, &input_metadata);
    let gathered = gathered_result
        .map_err(|err| builtin_error_with_detail(&IMAG_ERROR_INTERNAL, err.to_string()))?;
    let host = match gathered {
        Value::Complex(_, im) => Ok(Value::Num(im)),
        Value::ComplexTensor(ct) => imag_complex_tensor(ct),
        Value::Tensor(tensor) => Ok(tensor::tensor_into_value(imag_tensor(tensor)?)),
        other => imag_real(other),
    }?;
    gpu_helpers::restore_class_preserving_value(&handle, host, BUILTIN_NAME)
        .map_err(|err| builtin_error_with_detail(&IMAG_ERROR_INTERNAL, err.to_string()))
}

fn valid_imag_gpu_output(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
) -> bool {
    gpu_helpers::unary_gpu_output_matches(
        output,
        input,
        provider,
        gpu_helpers::UnaryGpuOutputContract {
            storage: runmat_accelerate_api::GpuTensorStorage::Real,
            precision: runmat_accelerate_api::handle_precision(input),
            integer: None,
            logical: false,
            alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinct,
        },
    )
}

fn imag_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("imag", value)
        .map_err(|e| builtin_error_with_detail(&IMAG_ERROR_INVALID_INPUT, e))?;
    Ok(tensor::tensor_into_value(imag_tensor(tensor)?))
}

fn imag_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|e| builtin_error_with_detail(&IMAG_ERROR_INTERNAL, e))?;
    let zeros = storage.zeros_like(storage.len());
    Tensor::from_numeric_storage(zeros, shape)
        .map_err(|e| builtin_error_with_detail(&IMAG_ERROR_INTERNAL, e))
}

fn imag_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let shape = ct.shape.clone();
    let storage = match ct.into_complex_storage() {
        ComplexStorage::F64(values) => {
            NumericStorage::F64(values.into_iter().map(|(_, imag)| imag).collect())
        }
        ComplexStorage::F32(values) => {
            NumericStorage::F32(values.into_iter().map(|(_, imag)| imag).collect())
        }
        ComplexStorage::Integer(storage) => NumericStorage::from_integer_storage(storage.imag),
    };
    let tensor = Tensor::from_numeric_storage(storage, shape)
        .map_err(|e| builtin_error_with_detail(&IMAG_ERROR_INTERNAL, e))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn imag_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let zeros = vec![0.0; ca.rows * ca.cols];
    let tensor = Tensor::new(zeros, vec![ca.rows, ca.cols])
        .map_err(|e| builtin_error_with_detail(&IMAG_ERROR_INTERNAL, e))?;
    Ok(tensor::tensor_into_value(tensor))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_value::{IntValue, LogicalArray, StringArray};

    fn imag_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::imag_builtin(value))
    }

    #[test]
    fn imag_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = IMAG_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = imag(X)"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn imag_scalar_real_zero() {
        let result = imag_builtin(Value::Num(-2.5)).expect("imag");
        match result {
            Value::Num(n) => assert_eq!(n, 0.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn imag_complex_scalar() {
        let result = imag_builtin(Value::Complex(3.0, 4.0)).expect("imag");
        match result {
            Value::Num(n) => assert_eq!(n, 4.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn imag_integer_complex_scalar_preserves_int64_value() {
        let complex = ComplexTensor::new_integer(
            runmat_value::IntegerComplexStorage::new(
                runmat_value::IntegerStorage::I64(vec![i64::MIN]),
                runmat_value::IntegerStorage::I64(vec![i64::MAX]),
            )
            .unwrap(),
            vec![1, 1],
        )
        .unwrap();
        let result = imag_builtin(Value::ComplexTensor(complex)).expect("imag");
        assert_eq!(result, Value::Int(IntValue::I64(i64::MAX)));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn imag_bool_scalar_zero() {
        let result = imag_builtin(Value::Bool(true)).expect("imag");
        match result {
            Value::Num(n) => assert_eq!(n, 0.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn imag_int_scalar_zero() {
        let result = imag_builtin(Value::Int(IntValue::I32(-42))).expect("imag");
        assert_eq!(result, Value::Int(IntValue::I32(0)));
    }

    #[test]
    fn imag_typed_real_integer_tensor_zeros_from_storage_without_mirror() {
        let tensor = Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![1, 9_223_372_036_854_775_809, u64::MAX]),
            vec![1, 3],
        )
        .expect("typed integer tensor");

        let result = imag_builtin(Value::Tensor(tensor)).expect("imag");
        let Value::Tensor(output) = result else {
            panic!("expected typed integer zero tensor");
        };
        assert_eq!(output.shape, vec![1, 3]);
        assert_eq!(
            output.integer_storage(),
            Some(&runmat_value::IntegerStorage::U64(vec![0, 0, 0]))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn imag_tensor_real_is_zero() {
        let tensor = Tensor::new(vec![1.0, -2.0, 3.5, 4.25], vec![4, 1]).unwrap();
        let result = imag_builtin(Value::Tensor(tensor)).expect("imag");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![4, 1]);
                assert!(t.materialize_f64().iter().all(|v| *v == 0.0));
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn imag_real_and_complex_single_preserve_native_class_and_shape() {
        let real = Tensor::from_f32(vec![1.0, -2.0], vec![1, 2]).unwrap();
        let Value::Tensor(output) = imag_builtin(Value::Tensor(real)).expect("imag") else {
            panic!("expected single zero tensor");
        };
        assert_eq!(output.shape, vec![1, 2]);
        assert_eq!(
            output.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![0.0, 0.0])
        );

        let complex = ComplexTensor::from_f32(vec![(1.25, 2.5), (-3.0, 4.0)], vec![2, 1]).unwrap();
        let Value::Tensor(output) = imag_builtin(Value::ComplexTensor(complex)).expect("imag")
        else {
            panic!("expected single imaginary tensor");
        };
        assert_eq!(output.shape, vec![2, 1]);
        assert_eq!(
            output.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![2.5, 4.0])
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn imag_empty_tensor_zero_length() {
        let tensor = Tensor::new(Vec::<f64>::new(), vec![0, 3]).unwrap();
        let result = imag_builtin(Value::Tensor(tensor)).expect("imag");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 3]);
                assert!(t.materialize_f64().is_empty());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn imag_empty_complex_single_preserves_native_class_and_shape() {
        let complex = ComplexTensor::from_f32(Vec::new(), vec![0, 3]).unwrap();
        let Value::Tensor(output) = imag_builtin(Value::ComplexTensor(complex)).expect("imag")
        else {
            panic!("expected empty single imaginary tensor");
        };
        assert_eq!(output.shape, vec![0, 3]);
        assert_eq!(
            output.into_numeric_storage().unwrap(),
            NumericStorage::F32(Vec::new())
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn imag_complex_tensor_to_tensor_of_imag_parts() {
        let complex =
            ComplexTensor::new(vec![(1.0, 2.0), (-3.0, 4.5)], vec![2, 1]).expect("complex tensor");
        let result = imag_builtin(Value::ComplexTensor(complex)).expect("imag");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                assert_eq!(t.materialize_f64(), vec![2.0, 4.5]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn imag_logical_array_zero() {
        let logical = LogicalArray::new(vec![0, 1, 1, 0], vec![2, 2]).expect("logical array");
        let result = imag_builtin(Value::LogicalArray(logical)).expect("imag");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.materialize_f64(), vec![0.0; 4]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn imag_char_array_zeroes() {
        let chars = CharArray::new("Az".chars().collect(), 1, 2).expect("char array");
        let result = imag_builtin(Value::CharArray(chars)).expect("imag");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.materialize_f64(), vec![0.0, 0.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn imag_string_error() {
        let err = imag_builtin(Value::from("hello")).expect_err("imag should error");
        let identifier = err.identifier().map(str::to_string);
        assert!(err.message().contains("expected numeric"));
        assert_eq!(identifier.as_deref(), IMAG_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn imag_string_array_error() {
        let arr =
            StringArray::new(vec!["a".to_string(), "b".to_string()], vec![2, 1]).expect("array");
        let err = imag_builtin(Value::StringArray(arr)).expect_err("imag should error");
        assert!(err.message().contains("expected numeric"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn imag_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = imag_builtin(Value::GpuTensor(handle)).expect("imag");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![4, 1]);
            assert!(gathered.materialize_f64().iter().all(|v| *v == 0.0));
        });
    }

    #[test]
    fn imag_rejects_contradictory_resident_class_before_provider_dispatch() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            runmat_accelerate_api::set_handle_class_name(&handle, "single");
            let err =
                block_on(imag_gpu(handle)).expect_err("contradictory class metadata must reject");
            assert!(err.message().contains("class metadata contradicts"));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn imag_complex_gpu_provider_stays_resident() {
        test_support::with_test_provider(|provider| {
            let complex = ComplexTensor::new(vec![(1.0, 2.0), (-3.0, 4.5)], vec![2, 1]).unwrap();
            let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
            let result = imag_builtin(Value::GpuTensor(handle)).expect("imag");
            let Value::GpuTensor(out) = result else {
                panic!("expected gpu tensor");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&out),
                runmat_accelerate_api::GpuTensorStorage::Real
            );
            let gathered = test_support::gather(Value::GpuTensor(out)).expect("gather");
            assert_eq!(gathered.shape, vec![2, 1]);
            assert_eq!(gathered.materialize_f64(), vec![2.0, 4.5]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn imag_wgpu_matches_cpu_zero() {
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        let tensor = Tensor::new(vec![0.0, 1.0, -2.5, 4.0], vec![4, 1]).unwrap();
        let cpu = imag_real(Value::Tensor(tensor.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let h = runmat_accelerate_api::AccelProvider::upload(provider, &view).unwrap();
        let gpu = block_on(imag_gpu(h)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        let cpu_tensor = match cpu {
            Value::Tensor(t) => t,
            Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
            other => panic!("unexpected cpu value {other:?}"),
        };
        assert_eq!(gathered.shape, cpu_tensor.shape);
        assert_eq!(
            gathered.materialize_f64().len(),
            cpu_tensor.materialize_f64().len()
        );
        for (g, c) in gathered
            .materialize_f64()
            .iter()
            .zip(cpu_tensor.materialize_f64().iter())
        {
            assert!((g - c).abs() < 1e-12, "imag mismatch {} vs {}", g, c);
        }
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn imag_wgpu_complex_matches_cpu() {
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        let complex = ComplexTensor::new(vec![(1.0, 2.0), (-3.0, 4.5)], vec![2, 1]).unwrap();
        let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
        let gpu = block_on(imag_gpu(handle)).unwrap();
        let Value::GpuTensor(out) = gpu else {
            panic!("expected gpu tensor");
        };
        assert_eq!(
            runmat_accelerate_api::handle_storage(&out),
            runmat_accelerate_api::GpuTensorStorage::Real
        );
        let gathered = test_support::gather(Value::GpuTensor(out)).expect("gather");
        assert_eq!(gathered.materialize_f64(), vec![2.0, 4.5]);
    }
}
