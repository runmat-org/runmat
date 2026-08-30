//! MATLAB-compatible `conj` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
#[cfg(test)]
use runmat_builtins::CONJ_DESCRIPTOR;
use runmat_builtins::{
    BuiltinErrorDescriptor, CONJ_CHARACTER_INPUT_EXTENSION, CONJ_ERROR_INTERNAL,
    CONJ_ERROR_INVALID_INPUT,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    CharArray, ComplexStorage, ComplexTensor, IntegerComplexStorage, IntegerStorage, Tensor, Value,
};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::conj")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "conj",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_conj" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may execute conj via unary_conj for real tensors and complex-interleaved GPU tensors, preserving complex GPU residency when supported.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::conj")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "conj",
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
    notes:
        "Fusion kernels treat conj as an identity for real tensors; complex tensors fall back to the CPU path until native complex fusion is available.",
};

const BUILTIN_NAME: &str = "conj";

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
    name = "conj",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::conj"
)]
async fn conj_builtin(value: Value) -> BuiltinResult<Value> {
    if matches!(value, Value::CharArray(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CONJ_CHARACTER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    match value {
        Value::GpuTensor(handle) => conj_gpu(handle).await,
        Value::Complex(re, im) => conj_complex_scalar(re, im),
        Value::ComplexTensor(ct) => conj_complex_tensor(ct),
        Value::CharArray(ca) => conj_char_array(ca),
        Value::String(_) | Value::StringArray(_) => Err(builtin_error_with_detail(
            &CONJ_ERROR_INVALID_INPUT,
            "expected numeric input",
        )),
        x @ (Value::LogicalArray(_) | Value::Bool(_)) => Ok(x),
        x @ (Value::Tensor(_) | Value::Num(_) | Value::Int(_)) => conj_real(x),
        other => Err(builtin_error_with_detail(
            &CONJ_ERROR_INVALID_INPUT,
            format!(
                "unsupported input type {:?}; expected numeric, logical, or char data",
                other
            ),
        )),
    }
}

async fn conj_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let storage = runmat_accelerate_api::handle_storage(&handle);
    if gpu_helpers::expected_handle_numeric_element_type(&handle).is_err() {
        return Err(builtin_error_with_detail(
            &CONJ_ERROR_INTERNAL,
            "GPU input class metadata contradicts its physical storage",
        ));
    }
    if runmat_accelerate_api::handle_integer_type(&handle).is_some()
        && storage == runmat_accelerate_api::GpuTensorStorage::Real
    {
        return Ok(gpu_helpers::resident_gpu_value(handle));
    }
    if runmat_accelerate_api::handle_is_logical(&handle)
        && storage == runmat_accelerate_api::GpuTensorStorage::Real
    {
        return Ok(gpu_helpers::logical_gpu_value(handle));
    }
    let provider = gpu_helpers::exact_provider_for_handle(&handle).ok_or_else(|| {
        builtin_error_with_detail(&CONJ_ERROR_INTERNAL, "GPU provider unavailable for input")
    })?;
    if runmat_accelerate_api::handle_integer_type(&handle).is_none()
        && !runmat_accelerate_api::handle_is_logical(&handle)
        && runmat_accelerate_api::handle_precision(&handle) == Some(provider.precision())
    {
        let input_metadata = gpu_helpers::snapshot_handle_metadata(&handle);
        let input_provenance = runmat_accelerate_api::handle_provenance(&handle)
            .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic);
        let result = provider.unary_conj(&handle).await;
        gpu_helpers::restore_handle_metadata(&handle, &input_metadata);
        match result {
            Ok(mut out) if valid_conj_gpu_output(&out, &handle, provider) => {
                runmat_accelerate_api::set_handle_provenance(&mut out, input_provenance);
                return Ok(
                    if storage == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved {
                        gpu_helpers::complex_gpu_value(out)
                    } else {
                        gpu_helpers::resident_gpu_value(out)
                    },
                );
            }
            Ok(out) => {
                gpu_helpers::free_unprotected_exact_owner(&out, &[&handle]);
                return Err(builtin_error_with_detail(
                    &CONJ_ERROR_INTERNAL,
                    "provider unary_conj returned malformed output",
                ));
            }
            Err(err) if err.to_string().contains("unary_conj not supported") => {}
            Err(err) => {
                return Err(builtin_error_with_detail(
                    &CONJ_ERROR_INTERNAL,
                    format!("provider unary_conj failed: {err}"),
                ));
            }
        }
    }
    let input_metadata = gpu_helpers::snapshot_handle_metadata(&handle);
    let gathered_result =
        gpu_helpers::download_value_preserving_residency_async(provider, &handle).await;
    gpu_helpers::restore_handle_metadata(&handle, &input_metadata);
    let gathered = gathered_result
        .map_err(|err| builtin_error_with_detail(&CONJ_ERROR_INTERNAL, err.to_string()))?;
    let host = conj_host(gathered)?;
    gpu_helpers::restore_class_preserving_value(&handle, host, BUILTIN_NAME)
        .map_err(|err| builtin_error_with_detail(&CONJ_ERROR_INTERNAL, err.message()))
}

fn valid_conj_gpu_output(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
) -> bool {
    let storage = runmat_accelerate_api::handle_storage(input);
    let alias = if storage == runmat_accelerate_api::GpuTensorStorage::Real {
        gpu_helpers::GpuOutputAliasPolicy::AllowInput
    } else {
        gpu_helpers::GpuOutputAliasPolicy::RequireDistinct
    };
    gpu_helpers::unary_gpu_output_matches(
        output,
        input,
        provider,
        gpu_helpers::UnaryGpuOutputContract {
            storage,
            precision: runmat_accelerate_api::handle_precision(input),
            integer: None,
            logical: false,
            alias,
        },
    )
}

fn conj_host(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Complex(re, im) => conj_complex_scalar(re, im),
        Value::ComplexTensor(ct) => conj_complex_tensor(ct),
        x @ (Value::LogicalArray(_) | Value::Bool(_)) => Ok(x),
        x @ (Value::Tensor(_) | Value::Num(_) | Value::Int(_)) => conj_real(x),
        other => Err(builtin_error_with_detail(
            &CONJ_ERROR_INVALID_INPUT,
            format!("unsupported gathered input {other:?}"),
        )),
    }
}

fn conj_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("conj", value)
        .map_err(|e| builtin_error_with_detail(&CONJ_ERROR_INVALID_INPUT, e))?;
    Ok(tensor::tensor_into_value(conj_tensor(tensor)?))
}

fn conj_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    Ok(tensor)
}

fn conj_complex_scalar(re: f64, im: f64) -> BuiltinResult<Value> {
    Ok(Value::Complex(re, -im))
}

fn conj_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let shape = ct.shape.clone();
    let storage = match ct.into_complex_storage() {
        ComplexStorage::F64(values) => ComplexStorage::F64(
            values
                .into_iter()
                .map(|(real, imag)| (real, -imag))
                .collect(),
        ),
        ComplexStorage::F32(values) => ComplexStorage::F32(
            values
                .into_iter()
                .map(|(real, imag)| (real, -imag))
                .collect(),
        ),
        ComplexStorage::Integer(storage) => ComplexStorage::Integer(
            IntegerComplexStorage::new(
                storage.real,
                conjugate_integer_imaginary_storage(storage.imag),
            )
            .map_err(|e| builtin_error_with_detail(&CONJ_ERROR_INTERNAL, e))?,
        ),
    };
    let tensor = ComplexTensor::from_complex_storage(storage, shape)
        .map_err(|e| builtin_error_with_detail(&CONJ_ERROR_INTERNAL, e))?;
    Ok(Value::ComplexTensor(tensor))
}

pub(crate) fn conjugate_integer_imaginary_storage(storage: IntegerStorage) -> IntegerStorage {
    match storage {
        IntegerStorage::I8(values) => {
            IntegerStorage::I8(values.into_iter().map(i8::saturating_neg).collect())
        }
        IntegerStorage::I16(values) => {
            IntegerStorage::I16(values.into_iter().map(i16::saturating_neg).collect())
        }
        IntegerStorage::I32(values) => {
            IntegerStorage::I32(values.into_iter().map(i32::saturating_neg).collect())
        }
        IntegerStorage::I64(values) => {
            IntegerStorage::I64(values.into_iter().map(i64::saturating_neg).collect())
        }
        IntegerStorage::U8(values) => IntegerStorage::U8(vec![0; values.len()]),
        IntegerStorage::U16(values) => IntegerStorage::U16(vec![0; values.len()]),
        IntegerStorage::U32(values) => IntegerStorage::U32(vec![0; values.len()]),
        IntegerStorage::U64(values) => IntegerStorage::U64(vec![0; values.len()]),
    }
}

fn conj_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data = ca
        .data
        .iter()
        .map(|&ch| ch as u32 as f64)
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| builtin_error_with_detail(&CONJ_ERROR_INTERNAL, e))?;
    Ok(tensor::tensor_into_value(tensor))
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
    use runmat_value::{IntValue, LogicalArray};

    fn conj_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::conj_builtin(value))
    }

    #[test]
    fn conj_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = CONJ_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = conj(X)"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conj_scalar_real() {
        let result = conj_builtin(Value::Num(-2.5)).expect("conj");
        match result {
            Value::Num(n) => assert!((n + 2.5).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conj_complex_scalar() {
        let result = conj_builtin(Value::Complex(3.0, 4.0)).expect("conj");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 3.0).abs() < 1e-12);
                assert!((im + 4.0).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conj_complex_scalar_zero_imag_remains_complex() {
        let result = conj_builtin(Value::Complex(5.0, 0.0)).expect("conj");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 5.0).abs() < 1e-12);
                assert_eq!(im, -0.0);
            }
            other => panic!("expected complex scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conj_preserves_logical_class() {
        let logical =
            LogicalArray::new(vec![0, 1, 1, 0], vec![2, 2]).expect("logical array construction");
        let result = conj_builtin(Value::LogicalArray(logical)).expect("conj");
        match result {
            Value::LogicalArray(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![0, 1, 1, 0]);
            }
            other => panic!("expected logical result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conj_int_preserves_integer_class() {
        let result = conj_builtin(Value::Int(IntValue::I32(7))).expect("conj");
        match result {
            Value::Int(IntValue::I32(n)) => assert_eq!(n, 7),
            other => panic!("expected int32 scalar result, got {other:?}"),
        }
    }

    #[test]
    fn conj_real_integer_arrays_preserve_all_eight_classes_and_wide_values() {
        let cases = [
            IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
            IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
            IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
            IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
            IntegerStorage::U8(vec![0, u8::MAX]),
            IntegerStorage::U16(vec![0, u16::MAX]),
            IntegerStorage::U32(vec![0, u32::MAX]),
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
        ];
        for storage in cases {
            let input = Tensor::new_integer(storage.clone(), vec![1, 2]).unwrap();
            let Value::Tensor(output) = conj_builtin(Value::Tensor(input)).expect("conj") else {
                panic!("expected tensor");
            };
            assert_eq!(output.integer_storage(), Some(&storage));
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conj_complex_tensor_to_complex_tensor() {
        let tensor =
            ComplexTensor::new(vec![(1.0, 2.0), (-3.0, -4.0)], vec![2, 1]).expect("complex tensor");
        let result = conj_builtin(Value::ComplexTensor(tensor)).expect("conj");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![2, 1]);
                assert_eq!(ct.materialize_f64()[0], (1.0, -2.0));
                assert_eq!(ct.materialize_f64()[1], (-3.0, 4.0));
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conj_complex_tensor_zero_imag_remains_complex() {
        let tensor =
            ComplexTensor::new(vec![(1.0, 0.0), (2.0, -0.0)], vec![2, 1]).expect("complex tensor");
        let result = conj_builtin(Value::ComplexTensor(tensor)).expect("conj");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                assert_eq!(t.materialize_f64(), vec![(1.0, -0.0), (2.0, 0.0)]);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn conj_complex_single_preserves_native_class_shape_and_empty_storage() {
        let tensor = ComplexTensor::from_f32(vec![(1.25, 2.5), (-3.0, -4.0)], vec![1, 2]).unwrap();
        let Value::ComplexTensor(output) =
            conj_builtin(Value::ComplexTensor(tensor)).expect("conj")
        else {
            panic!("expected complex single tensor");
        };
        assert_eq!(output.shape, vec![1, 2]);
        assert_eq!(
            output.as_f32_slice().map(|values| values
                .iter()
                .copied()
                .map(<(f32, f32)>::from)
                .collect::<Vec<_>>()),
            Some(vec![(1.25, -2.5), (-3.0, 4.0)])
        );
        let empty = ComplexTensor::from_f32(Vec::new(), vec![0, 2]).unwrap();
        let Value::ComplexTensor(output) = conj_builtin(Value::ComplexTensor(empty)).expect("conj")
        else {
            panic!("expected empty complex single tensor");
        };
        assert_eq!(output.shape, vec![0, 2]);
        assert_eq!(output.as_f32_slice(), Some(&[][..]));
    }

    #[test]
    fn conj_typed_integer_imaginary_storage_preserves_class_and_saturates() {
        let cases = [
            (
                IntegerStorage::I8(vec![3, i8::MIN]),
                IntegerStorage::I8(vec![-3, i8::MAX]),
            ),
            (
                IntegerStorage::I16(vec![3, i16::MIN]),
                IntegerStorage::I16(vec![-3, i16::MAX]),
            ),
            (
                IntegerStorage::I32(vec![3, i32::MIN]),
                IntegerStorage::I32(vec![-3, i32::MAX]),
            ),
            (
                IntegerStorage::I64(vec![3, i64::MIN]),
                IntegerStorage::I64(vec![-3, i64::MAX]),
            ),
            (
                IntegerStorage::U8(vec![3, u8::MAX]),
                IntegerStorage::U8(vec![0, 0]),
            ),
            (
                IntegerStorage::U16(vec![3, u16::MAX]),
                IntegerStorage::U16(vec![0, 0]),
            ),
            (
                IntegerStorage::U32(vec![3, u32::MAX]),
                IntegerStorage::U32(vec![0, 0]),
            ),
            (
                IntegerStorage::U64(vec![3, u64::MAX]),
                IntegerStorage::U64(vec![0, 0]),
            ),
        ];
        for (input, expected) in cases {
            assert_eq!(conjugate_integer_imaginary_storage(input), expected);
        }
    }

    #[test]
    fn conj_complex_integer_tensor_reads_storage_without_mirror() {
        let complex = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::I16(vec![-10, 20]),
                IntegerStorage::I16(vec![3, i16::MIN]),
            )
            .unwrap(),
            vec![1, 2],
        )
        .unwrap();

        let result = conj_builtin(Value::ComplexTensor(complex)).expect("conj");
        let Value::ComplexTensor(output) = result else {
            panic!("expected typed complex integer tensor");
        };
        assert_eq!(output.shape, vec![1, 2]);
        assert_eq!(
            output.integer_storage().cloned(),
            Some(
                IntegerComplexStorage::new(
                    IntegerStorage::I16(vec![-10, 20]),
                    IntegerStorage::I16(vec![-3, i16::MAX]),
                )
                .unwrap()
            )
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conj_char_array_returns_double_codes() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let chars = CharArray::new("Hi".chars().collect(), 1, 2).expect("char array");
        let result = conj_builtin(Value::CharArray(chars)).expect("conj");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.materialize_f64(), vec![72.0, 105.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn conj_char_extension_is_compatibility_gated() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = conj_builtin(Value::CharArray(CharArray::new_row("x"))).unwrap_err();
        assert_eq!(
            err.identifier(),
            CONJ_CHARACTER_INPUT_EXTENSION.error_identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conj_errors_on_string_input() {
        let err = conj_builtin(Value::from("hello")).unwrap_err();
        let identifier = err.identifier().map(str::to_string);
        assert!(err.message().contains("expected numeric input"));
        assert_eq!(identifier.as_deref(), CONJ_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conj_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, -3.0, 4.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = conj_builtin(Value::GpuTensor(handle)).expect("conj");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.materialize_f64(), tensor.materialize_f64());
        });
    }

    #[test]
    fn conj_resident_integer_is_exact_identity() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new_integer(
                IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
                vec![2, 1],
            )
            .unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let buffer_id = handle.buffer_id;
            let result = conj_builtin(Value::GpuTensor(handle)).expect("conj");
            let Value::GpuTensor(output) = result else {
                panic!("expected resident integer");
            };
            assert_eq!(output.buffer_id, buffer_id);
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&output),
                Some(runmat_accelerate_api::IntegerElementType::U64)
            );
            let gathered = block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(output)))
                .expect("gather");
            let Value::Tensor(gathered) = gathered else {
                panic!("integer tensor");
            };
            assert_eq!(gathered.integer_storage(), tensor.integer_storage());
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conj_complex_gpu_provider_stays_resident() {
        test_support::with_test_provider(|provider| {
            let complex = ComplexTensor::new(vec![(1.0, 2.0), (-3.0, -4.0)], vec![2, 1]).unwrap();
            let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
            let result = conj_builtin(Value::GpuTensor(handle)).expect("conj");
            let Value::GpuTensor(out) = result else {
                panic!("expected gpu tensor");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&out),
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            );
            let gathered =
                block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(out))).expect("gather");
            let Value::ComplexTensor(ct) = gathered else {
                panic!("expected complex tensor");
            };
            assert_eq!(ct.shape, vec![2, 1]);
            assert_eq!(ct.materialize_f64(), vec![(1.0, -2.0), (-3.0, 4.0)]);
        });
    }

    #[test]
    fn conj_typed_complex_integer_gpu_stays_exact_and_resident() {
        test_support::with_test_provider(|provider| {
            let complex = ComplexTensor::new_integer(
                IntegerComplexStorage::new(
                    IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
                    IntegerStorage::I64(vec![i64::MIN, 17]),
                )
                .expect("storage"),
                vec![2, 1],
            )
            .expect("complex");
            let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
            let Value::GpuTensor(output) = conj_builtin(Value::GpuTensor(handle)).expect("conj")
            else {
                panic!("expected resident complex integer");
            };
            let Value::ComplexTensor(gathered) =
                block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(output)))
                    .expect("gather")
            else {
                panic!("expected complex integer");
            };
            let storage = gathered.integer_storage().expect("integer storage");
            assert_eq!(storage.real, IntegerStorage::I64(vec![i64::MIN, i64::MAX]));
            assert_eq!(storage.imag, IntegerStorage::I64(vec![i64::MAX, -17]));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn conj_wgpu_matches_cpu_for_real() {
        let _guard = test_support::accel_test_lock();
        if !register_wgpu_provider_available() {
            return;
        }
        let tensor = Tensor::new(vec![1.0, -2.0, 3.5, 0.0], vec![4, 1]).unwrap();
        let cpu = conj_real(Value::Tensor(tensor.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = block_on(conj_gpu(handle)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(ct.shape, gt.shape);
                assert_eq!(ct.materialize_f64(), gt.materialize_f64());
            }
            _ => panic!("unexpected shapes"),
        }
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn conj_wgpu_preserves_wide_uint64_identity_handle() {
        let _guard = test_support::accel_test_lock();
        if !register_wgpu_provider_available() {
            return;
        }
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let tensor = Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
            vec![2, 1],
        )
        .unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
        let buffer_id = handle.buffer_id;
        let Value::GpuTensor(output) = block_on(conj_gpu(handle)).expect("conj") else {
            panic!("expected resident integer");
        };
        assert_eq!(output.buffer_id, buffer_id);
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&output),
            Some(runmat_accelerate_api::IntegerElementType::U64)
        );
        let gathered =
            block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(output))).expect("gather");
        let Value::Tensor(gathered) = gathered else {
            panic!("expected integer tensor");
        };
        assert_eq!(gathered.integer_storage(), tensor.integer_storage());
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn conj_wgpu_complex_matches_cpu() {
        let _guard = test_support::accel_test_lock();
        if !register_wgpu_provider_available() {
            return;
        }
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let complex = ComplexTensor::new(vec![(1.0, 2.0), (-3.0, -4.0)], vec![2, 1]).unwrap();
        let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
        let gpu = block_on(conj_gpu(handle)).unwrap();
        let Value::GpuTensor(out) = gpu else {
            panic!("expected gpu tensor");
        };
        assert_eq!(
            runmat_accelerate_api::handle_storage(&out),
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        );
        let gathered =
            block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(out))).expect("gather");
        let Value::ComplexTensor(ct) = gathered else {
            panic!("expected complex tensor");
        };
        assert_eq!(ct.materialize_f64(), vec![(1.0, -2.0), (-3.0, 4.0)]);
    }
}
