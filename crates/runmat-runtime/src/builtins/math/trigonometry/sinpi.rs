//! MATLAB-compatible `sinpi` builtin for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinErrorDescriptor, SINPI_CHARACTER_INPUT_EXTENSION, SINPI_ERROR_INTERNAL,
    SINPI_ERROR_INVALID_INPUT, SINPI_INTEGER_INPUT_EXTENSION, SINPI_LOGICAL_INPUT_EXTENSION,
};
#[cfg(test)]
use runmat_builtins::{SINPI_DESCRIPTOR, SINPI_INTEGER_CAPABILITIES};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, ComplexStorage, ComplexTensor, NumericDType, Tensor, Value};

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::trigonometry::pi_helpers::{sinpi_complex, sinpi_real};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "sinpi";
#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::trigonometry::sinpi")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: BUILTIN_NAME,
    op_kind: GpuOpKind::Custom("trig_pi"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "RunMat gathers gpuArray inputs and evaluates sinpi on the host to preserve exact integer and half-integer results.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::trigonometry::sinpi")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: BUILTIN_NAME,
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "Fusion is disabled because lowering to sin(x*pi) would lose sinpi's exactness guarantees.",
};

#[runtime_builtin(
    name = "sinpi",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::trigonometry::sinpi"
)]
async fn sinpi_builtin(value: Value) -> BuiltinResult<Value> {
    if crate::builtins::common::validation::value_contains_native_integer_class(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SINPI_INTEGER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if matches!(&value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(&value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SINPI_LOGICAL_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if matches!(&value, Value::CharArray(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SINPI_CHARACTER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    crate::builtins::common::validation::reject_typed_complex_integer(&value, "sinpi")?;
    match value {
        Value::GpuTensor(handle) => sinpi_gpu(handle).await,
        Value::Complex(re, im) => {
            let (out_re, out_im) = sinpi_complex(re, im);
            Ok(Value::Complex(out_re, out_im))
        }
        Value::ComplexTensor(tensor) => sinpi_complex_tensor(tensor),
        Value::CharArray(array) => sinpi_char_array(array),
        Value::String(_) | Value::StringArray(_) => Err(sinpi_error(&SINPI_ERROR_INVALID_INPUT)),
        other => sinpi_real_value(other),
    }
}

fn sinpi_char_array(array: CharArray) -> BuiltinResult<Value> {
    let tensor = Tensor::new(vec![0.0; array.data.len()], array.shape)
        .map_err(|err| sinpi_error_with_detail(&SINPI_ERROR_INTERNAL, err))?;
    Ok(tensor::tensor_into_value(tensor))
}

async fn sinpi_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle)).await?;
    match gathered {
        Value::Complex(re, im) => {
            let (out_re, out_im) = sinpi_complex(re, im);
            Ok(Value::Complex(out_re, out_im))
        }
        Value::ComplexTensor(tensor) => sinpi_complex_tensor(tensor),
        other => sinpi_real_value(other),
    }
}

fn sinpi_real_value(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value)
        .map_err(|err| sinpi_error_with_detail(&SINPI_ERROR_INVALID_INPUT, err))?;
    sinpi_tensor(tensor).map(tensor::tensor_into_value)
}

fn sinpi_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    if tensor.integer_storage().is_some() {
        return Tensor::new(vec![0.0; tensor.len()], tensor.shape.clone())
            .map_err(|err| sinpi_error_with_detail(&SINPI_ERROR_INTERNAL, err));
    }
    if tensor.numeric_dtype() == NumericDType::F32 {
        let data = tensor
            .as_f32_slice()
            .expect("single tensor storage")
            .iter()
            .map(|&value| sinpi_real(f64::from(value)) as f32)
            .collect();
        return Tensor::from_f32(data, tensor.shape.clone())
            .map_err(|err| sinpi_error_with_detail(&SINPI_ERROR_INTERNAL, err));
    }
    let data = tensor::tensor_values_f64_cow(&tensor)
        .iter()
        .map(|&value| sinpi_real(value))
        .collect();
    Tensor::new(data, tensor.shape.clone())
        .map_err(|err| sinpi_error_with_detail(&SINPI_ERROR_INTERNAL, err))
}

fn sinpi_complex_tensor(tensor: ComplexTensor) -> BuiltinResult<Value> {
    let shape = tensor.shape.clone();
    let converted = match tensor.into_complex_storage() {
        ComplexStorage::F32(values) => ComplexTensor::from_f32(
            values
                .into_iter()
                .map(|(re, im)| {
                    let (out_re, out_im) = sinpi_complex(f64::from(re), f64::from(im));
                    (out_re as f32, out_im as f32)
                })
                .collect(),
            shape,
        ),
        ComplexStorage::F64(values) => ComplexTensor::new(
            values
                .into_iter()
                .map(|(re, im)| sinpi_complex(re, im))
                .collect(),
            shape,
        ),
        ComplexStorage::Integer(_) => Err("typed complex integer input is unsupported".into()),
    }
    .map_err(|err| sinpi_error_with_detail(&SINPI_ERROR_INTERNAL, err))?;
    Ok(complex_tensor_into_value(converted))
}

fn sinpi_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn sinpi_error_with_detail(
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
        block_on(super::sinpi_builtin(value))
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
        assert_eq!(SINPI_DESCRIPTOR.signatures[0].label, "Y = sinpi(X)");
        assert_eq!(SINPI_INTEGER_CAPABILITIES[0].inputs[0].classes.len(), 8);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn scalar_exact_values() {
        assert_eq!(expect_num(call(Value::Num(0.0)).unwrap()), 0.0);
        assert_eq!(expect_num(call(Value::Num(0.5)).unwrap()), 1.0);
        assert_eq!(expect_num(call(Value::Num(1.0)).unwrap()), 0.0);
        assert_eq!(expect_num(call(Value::Num(1.5)).unwrap()), -1.0);
        assert_eq!(expect_num(call(Value::Num(-0.5)).unwrap()), -1.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn tensor_preserves_shape_and_exact_values() {
        let tensor = Tensor::new(vec![0.0, 0.5, 1.0, 1.5, 2.0], vec![1, 5]).unwrap();
        let Value::Tensor(out) = call(Value::Tensor(tensor)).unwrap() else {
            panic!("expected tensor");
        };
        assert_eq!(out.shape, vec![1, 5]);
        assert_eq!(out.materialize_f64(), vec![0.0, 1.0, 0.0, -1.0, 0.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn real_and_complex_single_preserve_class() {
        let tensor = Tensor::from_f32(vec![0.0, 0.5, 1.0], vec![1, 3]).unwrap();
        let Value::Tensor(out) = call(Value::Tensor(tensor)).unwrap() else {
            panic!("expected tensor");
        };
        assert_eq!(out.numeric_dtype(), NumericDType::F32);
        assert_eq!(out.as_f32_slice().unwrap(), &[0.0, 1.0, 0.0]);

        let complex = ComplexTensor::from_f32(vec![(0.5, 1.0)], vec![1, 1]).unwrap();
        let Value::ComplexTensor(out) = call(Value::ComplexTensor(complex)).unwrap() else {
            panic!("expected complex tensor");
        };
        assert_eq!(out.numeric_dtype(), NumericDType::F32);
        assert!((out.materialize_f64()[0].0 - std::f64::consts::PI.cosh()).abs() < 1e-5);
        assert_eq!(out.materialize_f64()[0].1, 0.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn integer_logical_and_character_extensions_are_exact_and_gated() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = call(Value::Int(IntValue::U64(u64::MAX)))
            .expect_err("strict mode rejects integer extension");
        assert_eq!(
            err.identifier(),
            SINPI_INTEGER_INPUT_EXTENSION.error_identifier
        );
        drop(_strict);

        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        for value in [
            IntValue::I8(i8::MIN),
            IntValue::I16(i16::MIN),
            IntValue::I32(i32::MIN),
            IntValue::I64(i64::MIN),
            IntValue::U8(u8::MAX),
            IntValue::U16(u16::MAX),
            IntValue::U32(u32::MAX),
            IntValue::U64(u64::MAX),
        ] {
            assert_eq!(expect_num(call(Value::Int(value)).unwrap()), 0.0);
        }
        let logical = LogicalArray::new(vec![0, 1], vec![1, 2]).unwrap();
        let Value::Tensor(out) = call(Value::LogicalArray(logical)).unwrap() else {
            panic!("expected tensor");
        };
        assert_eq!(out.materialize_f64(), vec![0.0, 0.0]);

        let Value::Tensor(out) = call(Value::CharArray(CharArray::new_row("AB"))).unwrap() else {
            panic!("expected character result tensor");
        };
        assert_eq!(out.shape, vec![1, 2]);
        assert_eq!(out.materialize_f64(), vec![0.0, 0.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn complex_inputs_use_analytic_extension() {
        let Value::Complex(re, im) = call(Value::Complex(0.5, 1.0)).unwrap() else {
            panic!("expected complex");
        };
        assert!((re - std::f64::consts::PI.cosh()).abs() < 1e-12);
        assert_eq!(im, 0.0);
    }

    #[test]
    fn complex_exact_zero_component_survives_overflowing_imaginary_scale() {
        let Value::Complex(re, im) = call(Value::Complex(0.5, f64::INFINITY)).unwrap() else {
            panic!("expected complex");
        };
        assert!(re.is_infinite() && re.is_sign_positive());
        assert_eq!(im, 0.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn gpu_input_is_gathered() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 0.5, 1.0], vec![1, 3]).unwrap();
            let handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &tensor.materialize_f64(),
                    shape: &tensor.shape,
                })
                .expect("upload");
            let Value::Tensor(out) = call(Value::GpuTensor(handle)).unwrap() else {
                panic!("expected tensor");
            };
            assert_eq!(out.materialize_f64(), vec![0.0, 1.0, 0.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn sinpi_reads_typed_integer_tensor_storage_exactly() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new_integer(
            runmat_value::IntegerStorage::I16(vec![-1, 0, 2]),
            vec![3, 1],
        )
        .expect("integer tensor");

        match call(Value::Tensor(tensor)).expect("sinpi") {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                assert!(out.materialize_f64().iter().all(|value| *value == 0.0));
                assert!(out.integer_storage().is_none());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn strings_are_rejected() {
        let err = call(Value::String("0.5".to_string())).unwrap_err();
        assert_eq!(err.identifier.as_deref(), Some("RunMat:sinpi:InvalidInput"));
    }
}
