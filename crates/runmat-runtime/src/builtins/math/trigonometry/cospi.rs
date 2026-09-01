//! MATLAB-compatible `cospi` builtin for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinErrorDescriptor, COSPI_CHARACTER_INPUT_EXTENSION, COSPI_ERROR_INTERNAL,
    COSPI_ERROR_INVALID_INPUT, COSPI_INTEGER_INPUT_EXTENSION, COSPI_LOGICAL_INPUT_EXTENSION,
};
#[cfg(test)]
use runmat_builtins::{COSPI_DESCRIPTOR, COSPI_INTEGER_CAPABILITIES};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, ComplexStorage, ComplexTensor, NumericDType, Tensor, Value};

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::trigonometry::pi_helpers::{cospi_complex, cospi_real};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "cospi";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::trigonometry::cospi")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: BUILTIN_NAME,
    op_kind: GpuOpKind::Custom("trig_pi"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "RunMat gathers gpuArray inputs, evaluates cospi on the host to preserve exact integer and half-integer results, and uploads a new result handle to the input's owning provider.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::trigonometry::cospi")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: BUILTIN_NAME,
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "Fusion is disabled because lowering to cos(x*pi) would lose cospi's exactness guarantees.",
};

#[runtime_builtin(
    name = "cospi",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::trigonometry::cospi"
)]
async fn cospi_builtin(value: Value) -> BuiltinResult<Value> {
    ensure_extensions(&value)?;
    crate::builtins::common::validation::reject_typed_complex_integer(&value, "cospi")?;
    match value {
        Value::GpuTensor(handle) => cospi_gpu(handle).await,
        Value::Complex(re, im) => {
            let (out_re, out_im) = cospi_complex(re, im);
            Ok(Value::Complex(out_re, out_im))
        }
        Value::ComplexTensor(tensor) => cospi_complex_tensor(tensor),
        Value::CharArray(array) => cospi_char_array(array),
        Value::String(_) | Value::StringArray(_) => Err(cospi_error(&COSPI_ERROR_INVALID_INPUT)),
        other => cospi_real_value(other),
    }
}

fn cospi_char_array(array: CharArray) -> BuiltinResult<Value> {
    let data = array
        .data
        .into_iter()
        .map(|value| if u32::from(value) % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let tensor = Tensor::new(data, array.shape)
        .map_err(|err| cospi_error_with_detail(&COSPI_ERROR_INTERNAL, err))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn ensure_extensions(value: &Value) -> BuiltinResult<()> {
    if is_integer(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &COSPI_INTEGER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(h) if runmat_accelerate_api::handle_is_logical(h))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &COSPI_LOGICAL_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if matches!(value, Value::CharArray(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &COSPI_CHARACTER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}
fn is_integer(value: &Value) -> bool {
    crate::builtins::common::validation::value_contains_native_integer_class(value)
}

async fn cospi_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider_for_handle(&handle);
    let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle)).await?;
    let host = match gathered {
        Value::Complex(re, im) => {
            let (out_re, out_im) = cospi_complex(re, im);
            Ok(Value::Complex(out_re, out_im))
        }
        Value::ComplexTensor(tensor) => cospi_complex_tensor(tensor),
        other => cospi_real_value(other),
    }?;
    if let Some(provider) = provider {
        upload_gpu_output(provider, host)
    } else {
        Ok(host)
    }
}

fn cospi_real_value(value: Value) -> BuiltinResult<Value> {
    if let Value::Int(ref integer) = value {
        return Ok(Value::Num(if integer_is_even(integer) {
            1.0
        } else {
            -1.0
        }));
    }
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value)
        .map_err(|err| cospi_error_with_detail(&COSPI_ERROR_INVALID_INPUT, err))?;
    cospi_tensor(tensor).map(tensor::tensor_into_value)
}

fn cospi_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    if let Some(storage) = tensor.integer_storage() {
        let data = storage
            .exact_values()
            .iter()
            .map(|v| if integer_is_even(v) { 1.0 } else { -1.0 })
            .collect();
        return Tensor::new(data, tensor.shape.clone())
            .map_err(|err| cospi_error_with_detail(&COSPI_ERROR_INTERNAL, err));
    }
    if tensor.numeric_dtype() == NumericDType::F32 {
        let data = tensor
            .as_f32_slice()
            .expect("single tensor storage")
            .iter()
            .map(|&v| cospi_real(f64::from(v)) as f32)
            .collect();
        return Tensor::from_f32(data, tensor.shape.clone())
            .map_err(|err| cospi_error_with_detail(&COSPI_ERROR_INTERNAL, err));
    }
    let data = tensor::tensor_values_f64_cow(&tensor)
        .iter()
        .map(|&value| cospi_real(value))
        .collect();
    Tensor::new(data, tensor.shape.clone())
        .map_err(|err| cospi_error_with_detail(&COSPI_ERROR_INTERNAL, err))
}

fn integer_is_even(value: &runmat_value::IntValue) -> bool {
    match value {
        runmat_value::IntValue::I8(v) => v % 2 == 0,
        runmat_value::IntValue::I16(v) => v % 2 == 0,
        runmat_value::IntValue::I32(v) => v % 2 == 0,
        runmat_value::IntValue::I64(v) => v % 2 == 0,
        runmat_value::IntValue::U8(v) => v % 2 == 0,
        runmat_value::IntValue::U16(v) => v % 2 == 0,
        runmat_value::IntValue::U32(v) => v % 2 == 0,
        runmat_value::IntValue::U64(v) => v % 2 == 0,
    }
}

fn cospi_complex_tensor(tensor: ComplexTensor) -> BuiltinResult<Value> {
    let shape = tensor.shape.clone();
    let converted = match tensor.into_complex_storage() {
        ComplexStorage::F32(values) => ComplexTensor::from_f32(
            values
                .into_iter()
                .map(|(re, im)| {
                    let (re, im) = cospi_complex(f64::from(re), f64::from(im));
                    (re as f32, im as f32)
                })
                .collect(),
            shape,
        ),
        ComplexStorage::F64(values) => ComplexTensor::new(
            values
                .into_iter()
                .map(|(re, im)| cospi_complex(re, im))
                .collect(),
            shape,
        ),
        ComplexStorage::Integer(_) => Err("typed complex integer input is unsupported".into()),
    }
    .map_err(|err| cospi_error_with_detail(&COSPI_ERROR_INTERNAL, err))?;
    Ok(complex_tensor_into_value(converted))
}

fn upload_gpu_output(
    provider: &dyn runmat_accelerate_api::AccelProvider,
    value: Value,
) -> BuiltinResult<Value> {
    match value {
        Value::Num(value) => upload_real_gpu_output(
            provider,
            Tensor::new(vec![value], vec![1, 1])
                .map_err(|e| cospi_error_with_detail(&COSPI_ERROR_INTERNAL, e))?,
        ),
        Value::Tensor(tensor) => upload_real_gpu_output(provider, tensor),
        Value::Complex(re, im) => upload_complex_gpu_output(
            provider,
            ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| cospi_error_with_detail(&COSPI_ERROR_INTERNAL, e))?,
        ),
        Value::ComplexTensor(tensor) => upload_complex_gpu_output(provider, tensor),
        other => Err(cospi_error_with_detail(
            &COSPI_ERROR_INTERNAL,
            format!("cannot restore GPU output {other:?}"),
        )),
    }
}

fn upload_real_gpu_output(
    provider: &dyn runmat_accelerate_api::AccelProvider,
    tensor: Tensor,
) -> BuiltinResult<Value> {
    let handle = gpu_helpers::upload_tensor(provider, &tensor)
        .map_err(|e| cospi_error_with_detail(&COSPI_ERROR_INTERNAL, e))?;
    Ok(gpu_helpers::resident_gpu_value(handle))
}

fn upload_complex_gpu_output(
    provider: &dyn runmat_accelerate_api::AccelProvider,
    tensor: ComplexTensor,
) -> BuiltinResult<Value> {
    let handle = gpu_helpers::upload_complex_tensor(provider, &tensor)?;
    Ok(gpu_helpers::complex_gpu_value(handle))
}

fn cospi_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn cospi_error_with_detail(
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
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::{IntValue, LogicalArray};

    use crate::builtins::common::test_support;

    fn call(value: Value) -> BuiltinResult<Value> {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        block_on(super::cospi_builtin(value))
    }

    fn expect_num(value: Value) -> f64 {
        match value {
            Value::Num(value) => value,
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn descriptor_covers_core_form() {
        assert_eq!(COSPI_DESCRIPTOR.signatures[0].label, "Y = cospi(X)");
        assert_eq!(COSPI_INTEGER_CAPABILITIES[0].inputs[0].classes.len(), 8);
    }

    #[test]
    fn integer_logical_and_character_extensions_are_exact_and_gated() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = block_on(super::cospi_builtin(Value::Int(IntValue::U64(u64::MAX))))
            .expect_err("strict mode rejects integer extension");
        assert_eq!(
            err.identifier(),
            COSPI_INTEGER_INPUT_EXTENSION.error_identifier
        );
        drop(_strict);
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        for (value, expected) in [
            (IntValue::I8(i8::MIN), 1.0),
            (IntValue::I16(i16::MIN), 1.0),
            (IntValue::I32(i32::MIN), 1.0),
            (IntValue::I64(i64::MIN), 1.0),
            (IntValue::U8(u8::MAX), -1.0),
            (IntValue::U16(u16::MAX), -1.0),
            (IntValue::U32(u32::MAX), -1.0),
            (IntValue::U64(u64::MAX), -1.0),
        ] {
            assert_eq!(
                expect_num(block_on(super::cospi_builtin(Value::Int(value))).unwrap()),
                expected
            );
        }

        let logical = LogicalArray::new(vec![0, 1], vec![1, 2]).unwrap();
        let Value::Tensor(out) = call(Value::LogicalArray(logical)).unwrap() else {
            panic!("expected tensor");
        };
        assert_eq!(out.materialize_f64(), vec![1.0, -1.0]);

        let Value::Tensor(out) = call(Value::CharArray(CharArray::new_row("AB"))).unwrap() else {
            panic!("expected character result tensor");
        };
        assert_eq!(out.shape, vec![1, 2]);
        assert_eq!(out.materialize_f64(), vec![-1.0, 1.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn real_and_complex_single_preserve_class() {
        let Value::Tensor(real) = block_on(super::cospi_builtin(Value::Tensor(
            Tensor::from_f32(vec![0.0, 0.5], vec![2, 1]).unwrap(),
        )))
        .unwrap() else {
            panic!("expected single tensor")
        };
        assert_eq!(real.numeric_dtype(), NumericDType::F32);
        let Value::ComplexTensor(complex) = block_on(super::cospi_builtin(Value::ComplexTensor(
            ComplexTensor::from_f32(vec![(0.5, 1.0)], vec![1, 1]).unwrap(),
        )))
        .unwrap() else {
            panic!("expected complex tensor")
        };
        assert_eq!(complex.numeric_dtype(), NumericDType::F32);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn scalar_exact_values() {
        assert_eq!(expect_num(call(Value::Num(0.0)).unwrap()), 1.0);
        assert_eq!(expect_num(call(Value::Num(0.5)).unwrap()), 0.0);
        assert_eq!(expect_num(call(Value::Num(1.0)).unwrap()), -1.0);
        assert_eq!(expect_num(call(Value::Num(1.5)).unwrap()), 0.0);
        assert_eq!(expect_num(call(Value::Num(-0.5)).unwrap()), 0.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn tensor_preserves_shape_and_exact_values() {
        let tensor = Tensor::new(vec![0.0, 0.5, 1.0, 1.5, 2.0], vec![1, 5]).unwrap();
        let Value::Tensor(out) = call(Value::Tensor(tensor)).unwrap() else {
            panic!("expected tensor");
        };
        assert_eq!(out.shape, vec![1, 5]);
        assert_eq!(out.materialize_f64(), vec![1.0, 0.0, -1.0, 0.0, 1.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn complex_inputs_use_analytic_extension() {
        let Value::Complex(re, im) = call(Value::Complex(0.5, 1.0)).unwrap() else {
            panic!("expected complex");
        };
        assert_eq!(re, 0.0);
        assert!((im + std::f64::consts::PI.sinh()).abs() < 1e-12);
    }

    #[test]
    fn complex_exact_zero_component_survives_overflowing_imaginary_scale() {
        let Value::Complex(re, im) = call(Value::Complex(0.5, f64::INFINITY)).unwrap() else {
            panic!("expected complex");
        };
        assert_eq!(re, 0.0);
        assert!(im.is_infinite() && im.is_sign_negative());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn gpu_fallback_restores_output_to_owner() {
        assert_eq!(GPU_SPEC.residency, ResidencyPolicy::NewHandle);
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 0.5, 1.0], vec![1, 3]).unwrap();
            let handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &tensor.materialize_f64(),
                    shape: &tensor.shape,
                })
                .expect("upload");
            let Value::GpuTensor(out_handle) = call(Value::GpuTensor(handle)).unwrap() else {
                panic!("expected owner-resident tensor");
            };
            assert_eq!(out_handle.device_id, provider.device_id());
            let out = test_support::gather(Value::GpuTensor(out_handle)).expect("gather output");
            assert_eq!(out.materialize_f64(), vec![1.0, 0.0, -1.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn cospi_reads_typed_integer_tensor_storage_exactly() {
        let tensor = Tensor::new_integer(
            runmat_value::IntegerStorage::I16(vec![-1, 0, 2]),
            vec![3, 1],
        )
        .expect("integer tensor");

        match call(Value::Tensor(tensor)).expect("cospi") {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                assert_eq!(out.materialize_f64(), vec![-1.0, 1.0, 1.0]);
                assert!(out.integer_storage().is_none());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn strings_are_rejected() {
        let err = call(Value::String("0.5".to_string())).unwrap_err();
        assert_eq!(err.identifier.as_deref(), Some("RunMat:cospi:InvalidInput"));
    }
}
