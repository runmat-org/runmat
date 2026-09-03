//! `rad2deg` runtime binding and fusion declaration.

use runmat_builtins::{
    RAD2DEG_ERROR_INTERNAL, RAD2DEG_ERROR_INVALID_INPUT, RAD2DEG_INTEGER_INPUT_EXTENSION,
    RAD2DEG_LOGICAL_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::spec::{
    BuiltinFusionSpec, ConstantStrategy, FusionError, FusionExprContext, FusionKernelTemplate,
    ScalarType, ShapeRequirements,
};
use crate::BuiltinResult;

use super::execute::{self, AngleConversion};

const BUILTIN_NAME: &str = "rad2deg";
const RAD_TO_DEG: f64 = 180.0 / std::f64::consts::PI;
const CONVERSION: AngleConversion = AngleConversion {
    name: BUILTIN_NAME,
    scale_f64: RAD_TO_DEG,
    scale_f32: 180.0 / std::f32::consts::PI,
    invalid_input: &RAD2DEG_ERROR_INVALID_INPUT,
    internal_error: &RAD2DEG_ERROR_INTERNAL,
    integer_extension: &RAD2DEG_INTEGER_INPUT_EXTENSION,
    logical_extension: &RAD2DEG_LOGICAL_INPUT_EXTENSION,
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::trigonometry::angle_conversion::rad2deg"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "rad2deg",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            match ctx.scalar_ty {
                ScalarType::F64 => Ok(format!("({input} * f64({RAD_TO_DEG}))")),
                ScalarType::F32 => Ok(format!("({input} * {:.10})", 180.0 / std::f32::consts::PI)),
                other => Err(FusionError::UnsupportedPrecision(other)),
            }
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion emits a multiplication by 180/pi for radian-to-degree conversion.",
};

#[runtime_builtin(
    name = "rad2deg",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::trigonometry::angle_conversion::rad2deg"
)]
async fn rad2deg_builtin(value: Value) -> BuiltinResult<Value> {
    execute::apply(&CONVERSION, value).await
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::{gpu_helpers, test_support};
    use crate::RuntimeError;
    use futures::executor::block_on;
    use runmat_value::{ComplexTensor, IntValue, LogicalArray, NumericDType, Tensor};

    fn rad2deg_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::rad2deg_builtin(value))
    }

    fn error_message(err: &RuntimeError) -> String {
        err.message().to_string()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rad2deg_scalar() {
        let result = rad2deg_builtin(Value::Num(std::f64::consts::PI)).expect("rad2deg");
        match result {
            Value::Num(value) => assert!((value - 180.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rad2deg_tensor_preserves_shape() {
        let tensor = Tensor::new(
            vec![
                0.0,
                std::f64::consts::PI / 6.0,
                std::f64::consts::PI / 4.0,
                std::f64::consts::PI / 3.0,
                std::f64::consts::FRAC_PI_2,
            ],
            vec![1, 5],
        )
        .unwrap();
        let result = rad2deg_builtin(Value::Tensor(tensor)).expect("rad2deg");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 5]);
                let expected = [0.0, 30.0, 45.0, 60.0, 90.0];
                for (actual, expected) in tensor.materialize_f64().iter().zip(expected) {
                    assert!((actual - expected).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rad2deg_reads_typed_integer_tensor_storage_exactly() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor =
            Tensor::new_integer(runmat_value::IntegerStorage::I16(vec![0, 1, 2]), vec![3, 1])
                .expect("integer tensor");

        match rad2deg_builtin(Value::Tensor(tensor)).expect("rad2deg") {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                let expected = [0.0, RAD_TO_DEG, 2.0 * RAD_TO_DEG];
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
    fn rad2deg_int_promotes() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = rad2deg_builtin(Value::Int(IntValue::I32(1))).expect("rad2deg");
        match result {
            Value::Num(value) => assert!((value - RAD_TO_DEG).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rad2deg_logical_array_promotes() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let logical = LogicalArray::new(vec![0, 1], vec![1, 2]).unwrap();
        let result = rad2deg_builtin(Value::LogicalArray(logical)).expect("rad2deg");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 2]);
                assert_eq!(tensor.materialize_f64()[0], 0.0);
                assert!((tensor.materialize_f64()[1] - RAD_TO_DEG).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rad2deg_complex_scales_both_parts() {
        let result = rad2deg_builtin(Value::Complex(
            std::f64::consts::PI,
            std::f64::consts::FRAC_PI_2,
        ))
        .expect("rad2deg");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 180.0).abs() < 1e-12);
                assert!((im - 90.0).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[test]
    fn rad2deg_integer_extension_is_gated_and_wide_values_reject() {
        let strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = rad2deg_builtin(Value::Int(IntValue::I32(1)))
            .expect_err("strict mode rejects integer extension");
        assert_eq!(
            error.identifier(),
            RAD2DEG_INTEGER_INPUT_EXTENSION.error_identifier
        );
        drop(strict);

        let extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let error = rad2deg_builtin(Value::Int(IntValue::U64((1_u64 << 53) + 1)))
            .expect_err("inexact binary64 boundary rejects");
        assert!(error.message().contains("exactly representable as double"));
        drop(extensions);
    }

    #[test]
    fn rad2deg_preserves_single_tensor_class() {
        let tensor = Tensor::from_f32(vec![std::f32::consts::PI], vec![1, 1]).expect("single");
        match rad2deg_builtin(Value::Tensor(tensor)).expect("rad2deg") {
            Value::Tensor(output) => assert_eq!(output.numeric_dtype(), NumericDType::F32),
            other => panic!("expected single tensor, got {other:?}"),
        }
    }

    #[test]
    fn rad2deg_resident_complex_preserves_owner_storage_precision_and_shape() {
        test_support::with_test_provider(|provider| {
            let tensor = ComplexTensor::new(
                vec![(std::f64::consts::PI, std::f64::consts::FRAC_PI_2)],
                vec![1, 1],
            )
            .expect("complex double");
            let input = gpu_helpers::upload_complex_tensor(provider, &tensor).expect("upload");
            let output = rad2deg_builtin(Value::GpuTensor(input.clone())).expect("rad2deg");
            let Value::GpuTensor(handle) = &output else {
                panic!("expected resident result")
            };
            assert_eq!(handle.shape, vec![1, 1]);
            assert_eq!(handle.device_id, input.device_id);
            assert_eq!(
                runmat_accelerate_api::handle_storage(handle),
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            );
            assert!(runmat_accelerate_api::provider_for_handle(handle)
                .is_some_and(|owner| std::ptr::eq(owner, provider)));
            match block_on(gpu_helpers::gather_value_async(&output)).expect("gather") {
                Value::ComplexTensor(result) => {
                    let values = result.materialize_f64();
                    assert!((values[0].0 - 180.0).abs() < 1.0e-12);
                    assert!((values[0].1 - 90.0).abs() < 1.0e-12);
                }
                other => panic!("expected complex result, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rad2deg_string_errors() {
        let err = rad2deg_builtin(Value::String("pi".into())).expect_err("expected error");
        assert!(error_message(&err).contains("invalid input"));
        assert_eq!(err.identifier(), RAD2DEG_ERROR_INVALID_INPUT.identifier);
    }
}
