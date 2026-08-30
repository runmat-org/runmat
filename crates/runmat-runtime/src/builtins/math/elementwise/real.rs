//! MATLAB-compatible `real` builtin with GPU-aware semantics for RunMat.
use runmat_accelerate_api::GpuTensorHandle;
#[cfg(test)]
use runmat_builtins::REAL_DESCRIPTOR;
use runmat_builtins::{BuiltinErrorDescriptor, REAL_ERROR_INTERNAL, REAL_ERROR_INVALID_INPUT};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, ComplexStorage, ComplexTensor, NumericStorage, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::real")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "real",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_real" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may execute real via unary_real, including extracting real components from complex-interleaved GPU tensors. The runtime gathers only when the hook is absent or a host-only conversion is required.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::real")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "real",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx
                .inputs
                .first()
                .ok_or(FusionError::MissingInput(0))?;
            Ok(format!("({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion kernels treat real as an identity transform for real tensors; providers can override via fused pipelines when advantageous.",
};

const BUILTIN_NAME: &str = "real";

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
    name = "real",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::real"
)]
async fn real_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => real_gpu(handle).await,
        Value::Complex(re, _) => Ok(Value::Num(re)),
        Value::ComplexTensor(ct) => real_complex_tensor(ct),
        Value::CharArray(ca) => real_char_array(ca),
        Value::String(_) | Value::StringArray(_) => Err(builtin_error_with_detail(
            &REAL_ERROR_INVALID_INPUT,
            "expected numeric input",
        )),
        x @ (Value::Tensor(_)
        | Value::LogicalArray(_)
        | Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)) => real_real(x),
        other => Err(builtin_error_with_detail(
            &REAL_ERROR_INVALID_INPUT,
            format!(
                "unsupported input type {:?}; expected numeric, logical, or char input",
                other
            ),
        )),
    }
}

async fn real_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = gpu_helpers::exact_provider_for_handle(&handle).ok_or_else(|| {
        builtin_error_with_detail(&REAL_ERROR_INTERNAL, "GPU provider unavailable for input")
    })?;
    if gpu_helpers::expected_handle_numeric_element_type(&handle).is_err() {
        return Err(builtin_error_with_detail(
            &REAL_ERROR_INTERNAL,
            "GPU input class metadata contradicts its physical storage",
        ));
    }
    let kernel_compatible = runmat_accelerate_api::handle_integer_type(&handle).is_none()
        && !runmat_accelerate_api::handle_is_logical(&handle)
        && runmat_accelerate_api::handle_precision(&handle) == Some(provider.precision());
    if kernel_compatible {
        let input_metadata = gpu_helpers::snapshot_handle_metadata(&handle);
        let input_provenance = runmat_accelerate_api::handle_provenance(&handle)
            .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic);
        let result = provider.unary_real(&handle).await;
        gpu_helpers::restore_handle_metadata(&handle, &input_metadata);
        match result {
            Ok(mut out) if valid_real_gpu_output(&out, &handle, provider) => {
                runmat_accelerate_api::set_handle_provenance(&mut out, input_provenance);
                return Ok(gpu_helpers::resident_gpu_value(out));
            }
            Ok(out) => {
                gpu_helpers::free_unprotected_exact_owner(&out, &[&handle]);
                return Err(builtin_error_with_detail(
                    &REAL_ERROR_INTERNAL,
                    "provider unary_real returned malformed output",
                ));
            }
            Err(err) if err.to_string().contains("unary_real not supported") => {}
            Err(err) => {
                return Err(builtin_error_with_detail(
                    &REAL_ERROR_INTERNAL,
                    format!("provider unary_real failed: {err}"),
                ));
            }
        }
    }
    let input_metadata = gpu_helpers::snapshot_handle_metadata(&handle);
    let gathered_result =
        gpu_helpers::download_value_preserving_residency_async(provider, &handle).await;
    gpu_helpers::restore_handle_metadata(&handle, &input_metadata);
    let gathered = gathered_result
        .map_err(|err| builtin_error_with_detail(&REAL_ERROR_INTERNAL, err.to_string()))?;
    let host = match gathered {
        Value::Complex(re, _) => Ok(Value::Num(re)),
        Value::ComplexTensor(ct) => real_complex_tensor(ct),
        Value::Tensor(tensor) => Ok(tensor::tensor_into_value(real_tensor(tensor)?)),
        other => real_real(other),
    }?;
    gpu_helpers::restore_class_preserving_value(&handle, host, BUILTIN_NAME)
        .map_err(|err| builtin_error_with_detail(&REAL_ERROR_INTERNAL, err.to_string()))
}

fn valid_real_gpu_output(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
) -> bool {
    let alias = if runmat_accelerate_api::handle_storage(input)
        == runmat_accelerate_api::GpuTensorStorage::Real
    {
        gpu_helpers::GpuOutputAliasPolicy::AllowInput
    } else {
        gpu_helpers::GpuOutputAliasPolicy::RequireDistinct
    };
    gpu_helpers::unary_gpu_output_matches(
        output,
        input,
        provider,
        gpu_helpers::UnaryGpuOutputContract {
            storage: runmat_accelerate_api::GpuTensorStorage::Real,
            precision: runmat_accelerate_api::handle_precision(input),
            integer: None,
            logical: false,
            alias,
        },
    )
}

fn real_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("real", value)
        .map_err(|e| builtin_error_with_detail(&REAL_ERROR_INVALID_INPUT, e))?;
    Ok(tensor::tensor_into_value(real_tensor(tensor)?))
}

fn real_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    Ok(tensor)
}

fn real_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let shape = ct.shape.clone();
    let storage = match ct.into_complex_storage() {
        ComplexStorage::F64(values) => {
            NumericStorage::F64(values.into_iter().map(|(real, _)| real).collect())
        }
        ComplexStorage::F32(values) => {
            NumericStorage::F32(values.into_iter().map(|(real, _)| real).collect())
        }
        ComplexStorage::Integer(storage) => NumericStorage::from_integer_storage(storage.real),
    };
    let tensor = Tensor::from_numeric_storage(storage, shape)
        .map_err(|e| builtin_error_with_detail(&REAL_ERROR_INTERNAL, e))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn real_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data = ca
        .data
        .iter()
        .map(|&ch| ch as u32 as f64)
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| builtin_error_with_detail(&REAL_ERROR_INTERNAL, e))?;
    Ok(tensor::tensor_into_value(tensor))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_value::{IntValue, LogicalArray};

    fn real_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::real_builtin(value))
    }

    #[test]
    fn real_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = REAL_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = real(X)"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn real_scalar_num() {
        let result = real_builtin(Value::Num(-2.5)).expect("real");
        match result {
            Value::Num(n) => assert!((n + 2.5).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn real_complex_scalar() {
        let result = real_builtin(Value::Complex(3.0, 4.0)).expect("real");
        match result {
            Value::Num(n) => assert!((n - 3.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn real_int_scalar_preserves_integer_class() {
        let result = real_builtin(Value::Int(IntValue::I32(7))).expect("real");
        assert_eq!(result, Value::Int(IntValue::I32(7)));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn real_complex_tensor_to_real_tensor() {
        let complex =
            ComplexTensor::new(vec![(1.0, 2.0), (-3.0, 4.0)], vec![2, 1]).expect("complex tensor");
        let result = real_builtin(Value::ComplexTensor(complex)).expect("real");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                assert!((t.materialize_f64()[0] - 1.0).abs() < 1e-12);
                assert!((t.materialize_f64()[1] + 3.0).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn real_complex_single_preserves_native_class_shape_and_empty_storage() {
        let complex = ComplexTensor::from_f32(vec![(1.25, 2.5), (-3.0, 4.0)], vec![2, 1]).unwrap();
        let Value::Tensor(output) = real_builtin(Value::ComplexTensor(complex)).expect("real")
        else {
            panic!("expected single real tensor");
        };
        assert_eq!(output.shape, vec![2, 1]);
        assert_eq!(
            output.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![1.25, -3.0])
        );

        let empty = ComplexTensor::from_f32(Vec::new(), vec![0, 4]).unwrap();
        let Value::Tensor(output) = real_builtin(Value::ComplexTensor(empty)).expect("real") else {
            panic!("expected empty single real tensor");
        };
        assert_eq!(output.shape, vec![0, 4]);
        assert_eq!(
            output.into_numeric_storage().unwrap(),
            NumericStorage::F32(Vec::new())
        );
    }

    #[test]
    fn real_integer_complex_tensor_preserves_uint64_values() {
        let complex = ComplexTensor::new_integer(
            runmat_value::IntegerComplexStorage::new(
                runmat_value::IntegerStorage::U64(vec![9_223_372_036_854_775_809, u64::MAX]),
                runmat_value::IntegerStorage::U64(vec![2, 3]),
            )
            .unwrap(),
            vec![1, 2],
        )
        .unwrap();
        let result = real_builtin(Value::ComplexTensor(complex)).expect("real");
        let Value::Tensor(tensor) = result else {
            panic!("expected typed real tensor");
        };
        assert_eq!(
            tensor.integer_storage(),
            Some(&runmat_value::IntegerStorage::U64(vec![
                9_223_372_036_854_775_809,
                u64::MAX,
            ]))
        );
    }

    #[test]
    fn real_integer_complex_tensor_reads_component_storage_exactly() {
        let complex = ComplexTensor::new_integer(
            runmat_value::IntegerComplexStorage::new(
                runmat_value::IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
                runmat_value::IntegerStorage::I64(vec![7, -8]),
            )
            .unwrap(),
            vec![2, 1],
        )
        .unwrap();

        let result = real_builtin(Value::ComplexTensor(complex)).expect("real");
        let Value::Tensor(tensor) = result else {
            panic!("expected typed real tensor");
        };
        assert_eq!(tensor.shape, vec![2, 1]);
        assert_eq!(
            tensor.integer_storage(),
            Some(&runmat_value::IntegerStorage::I64(
                vec![i64::MIN, i64::MAX,]
            ))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn real_logical_array_to_numeric() {
        let logical = LogicalArray::new(vec![0, 1, 1, 0], vec![2, 2]).expect("logical array");
        let result = real_builtin(Value::LogicalArray(logical)).expect("real");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.materialize_f64(), vec![0.0, 1.0, 1.0, 0.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn real_char_array_codes() {
        let chars = CharArray::new("AZ".chars().collect(), 1, 2).expect("char array");
        let result = real_builtin(Value::CharArray(chars)).expect("real");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.materialize_f64(), vec![65.0, 90.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn real_string_error() {
        let err = real_builtin(Value::from("hello")).expect_err("real should error");
        let identifier = err.identifier().map(str::to_string);
        assert!(err.message().contains("expected numeric"));
        assert_eq!(identifier.as_deref(), REAL_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn real_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, -2.0, 3.5, -4.25], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = real_builtin(Value::GpuTensor(handle)).expect("real");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.materialize_f64(), tensor.materialize_f64());
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn real_complex_gpu_provider_stays_resident() {
        test_support::with_test_provider(|provider| {
            let complex = ComplexTensor::new(vec![(1.0, 2.0), (-3.0, 4.5)], vec![2, 1]).unwrap();
            let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
            let result = real_builtin(Value::GpuTensor(handle)).expect("real");
            let Value::GpuTensor(out) = result else {
                panic!("expected gpu tensor");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&out),
                runmat_accelerate_api::GpuTensorStorage::Real
            );
            let gathered = test_support::gather(Value::GpuTensor(out)).expect("gather");
            assert_eq!(gathered.shape, vec![2, 1]);
            assert_eq!(gathered.materialize_f64(), vec![1.0, -3.0]);
        });
    }

    #[test]
    fn real_resident_wide_integer_identity_preserves_class_and_owner() {
        test_support::with_test_provider(|provider| {
            let input = Tensor::new_integer(
                runmat_value::IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
                vec![1, 2],
            )
            .unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &input).expect("integer upload");
            let Value::GpuTensor(output) = real_builtin(Value::GpuTensor(handle)).expect("real")
            else {
                panic!("documented gpuArray path must remain resident");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&output),
                Some(runmat_accelerate_api::IntegerElementType::U64)
            );
            let gathered = test_support::gather(Value::GpuTensor(output)).expect("gather output");
            assert_eq!(
                gathered.integer_storage(),
                Some(&runmat_value::IntegerStorage::U64(vec![
                    9_007_199_254_740_993,
                    u64::MAX,
                ]))
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn real_wgpu_matches_cpu_identity() {
        if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_err()
        {
            return;
        }
        let tensor = Tensor::new(vec![0.0, 1.0, -2.5, 4.0], vec![4, 1]).unwrap();
        let cpu = real_real(Value::Tensor(tensor.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let h = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = block_on(real_gpu(h)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        let cpu_tensor = match cpu {
            Value::Tensor(t) => t,
            Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
            other => panic!("unexpected cpu value {other:?}"),
        };
        assert_eq!(gathered.shape, cpu_tensor.shape);
        let tol = match runmat_accelerate_api::provider().unwrap().precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (a, b) in gathered
            .materialize_f64()
            .iter()
            .zip(cpu_tensor.materialize_f64().iter())
        {
            assert!((a - b).abs() < tol, "|{} - {}| >= {}", a, b, tol);
        }
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn real_wgpu_complex_matches_cpu() {
        if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_err()
        {
            return;
        }
        let provider = runmat_accelerate_api::provider().unwrap();
        let complex = ComplexTensor::new(vec![(1.0, 2.0), (-3.0, 4.5)], vec![2, 1]).unwrap();
        let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
        let gpu = block_on(real_gpu(handle)).unwrap();
        let Value::GpuTensor(out) = gpu else {
            panic!("expected gpu tensor");
        };
        assert_eq!(
            runmat_accelerate_api::handle_storage(&out),
            runmat_accelerate_api::GpuTensorStorage::Real
        );
        let gathered = test_support::gather(Value::GpuTensor(out)).expect("gather");
        assert_eq!(gathered.materialize_f64(), vec![1.0, -3.0]);
    }
}
