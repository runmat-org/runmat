//! MATLAB-compatible `deg2rad` builtin for RunMat.

use runmat_builtins::{
    DEG2RAD_ERROR_INTERNAL, DEG2RAD_ERROR_INVALID_INPUT, DEG2RAD_INTEGER_INPUT_EXTENSION,
    DEG2RAD_LOGICAL_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::spec::{
    BuiltinFusionSpec, ConstantStrategy, FusionError, FusionExprContext, FusionKernelTemplate,
    ScalarType, ShapeRequirements,
};
use crate::BuiltinResult;

use super::execute::{self, AngleConversion};

const BUILTIN_NAME: &str = "deg2rad";
const DEG_TO_RAD: f64 = std::f64::consts::PI / 180.0;
const CONVERSION: AngleConversion = AngleConversion {
    name: BUILTIN_NAME,
    scale_f64: DEG_TO_RAD,
    scale_f32: std::f32::consts::PI / 180.0,
    invalid_input: &DEG2RAD_ERROR_INVALID_INPUT,
    internal_error: &DEG2RAD_ERROR_INTERNAL,
    integer_extension: &DEG2RAD_INTEGER_INPUT_EXTENSION,
    logical_extension: &DEG2RAD_LOGICAL_INPUT_EXTENSION,
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::trigonometry::angle_conversion::deg2rad"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "deg2rad",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            match ctx.scalar_ty {
                ScalarType::F64 => Ok(format!("({input} * f64({DEG_TO_RAD}))")),
                ScalarType::F32 => Ok(format!("({input} * {:.10})", std::f32::consts::PI / 180.0)),
                other => Err(FusionError::UnsupportedPrecision(other)),
            }
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion emits a multiplication by pi/180 for degree-to-radian conversion.",
};

#[runtime_builtin(
    name = "deg2rad",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::trigonometry::angle_conversion::deg2rad"
)]
async fn deg2rad_builtin(value: Value) -> BuiltinResult<Value> {
    execute::apply(&CONVERSION, value).await
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::{gpu_helpers, test_support};
    use crate::RuntimeError;
    use futures::executor::block_on;
    use runmat_value::{ComplexTensor, IntValue, LogicalArray, NumericDType, Tensor};

    fn deg2rad_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::deg2rad_builtin(value))
    }

    fn error_message(err: &RuntimeError) -> String {
        err.message().to_string()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn deg2rad_scalar() {
        let result = deg2rad_builtin(Value::Num(90.0)).expect("deg2rad");
        match result {
            Value::Num(value) => assert!((value - std::f64::consts::FRAC_PI_2).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn deg2rad_tensor_preserves_shape() {
        let tensor = Tensor::new(vec![0.0, 30.0, 45.0, 60.0, 90.0], vec![1, 5]).unwrap();
        let result = deg2rad_builtin(Value::Tensor(tensor)).expect("deg2rad");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 5]);
                let expected = [
                    0.0,
                    std::f64::consts::PI / 6.0,
                    std::f64::consts::PI / 4.0,
                    std::f64::consts::PI / 3.0,
                    std::f64::consts::FRAC_PI_2,
                ];
                for (actual, expected) in tensor.materialize_f64().iter().zip(expected) {
                    assert!((actual - expected).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn deg2rad_reads_typed_integer_tensor_storage_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new_integer(
            runmat_value::IntegerStorage::I16(vec![0, 90, 180]),
            vec![3, 1],
        )
        .expect("integer tensor");

        match deg2rad_builtin(Value::Tensor(tensor)).expect("deg2rad") {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                let expected = [0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI];
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
    fn deg2rad_int_promotes() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = deg2rad_builtin(Value::Int(IntValue::I32(180))).expect("deg2rad");
        match result {
            Value::Num(value) => assert!((value - std::f64::consts::PI).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn deg2rad_integer_extension_is_gated_and_wide_values_reject() {
        let strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = deg2rad_builtin(Value::Int(IntValue::I32(1)))
            .expect_err("strict mode rejects integer extension");
        assert_eq!(
            error.identifier(),
            DEG2RAD_INTEGER_INPUT_EXTENSION.error_identifier
        );
        drop(strict);

        let compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let error = deg2rad_builtin(Value::Int(IntValue::U64((1u64 << 53) + 1)))
            .expect_err("inexact binary64 boundary rejects");
        assert!(error.message().contains("exactly representable as double"));
        drop(compat);
    }

    #[test]
    fn deg2rad_preserves_single_tensor_class() {
        let tensor = Tensor::from_f32(vec![90.0], vec![1, 1]).expect("single");
        match deg2rad_builtin(Value::Tensor(tensor)).expect("deg2rad") {
            Value::Tensor(output) => assert_eq!(output.numeric_dtype(), NumericDType::F32),
            other => panic!("expected single tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn deg2rad_logical_array_promotes() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let logical = LogicalArray::new(vec![0, 1], vec![1, 2]).unwrap();
        let result = deg2rad_builtin(Value::LogicalArray(logical)).expect("deg2rad");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 2]);
                assert_eq!(tensor.materialize_f64()[0], 0.0);
                assert!((tensor.materialize_f64()[1] - DEG_TO_RAD).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn deg2rad_complex_scales_both_parts() {
        let result = deg2rad_builtin(Value::Complex(180.0, 90.0)).expect("deg2rad");
        match result {
            Value::Complex(re, im) => {
                assert!((re - std::f64::consts::PI).abs() < 1e-12);
                assert!((im - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[test]
    fn deg2rad_resident_complex_preserves_owner_storage_precision_and_shape() {
        test_support::with_test_provider(|provider| {
            let tensor =
                ComplexTensor::new(vec![(180.00000000000003, 90.0), (0.0, -180.0)], vec![1, 2])
                    .expect("complex double");
            let input = gpu_helpers::upload_complex_tensor(provider, &tensor).expect("upload");
            let output = deg2rad_builtin(Value::GpuTensor(input.clone())).expect("deg2rad");
            let Value::GpuTensor(handle) = &output else {
                panic!("expected resident result")
            };
            assert_eq!(handle.shape, vec![1, 2]);
            assert_eq!(handle.device_id, input.device_id);
            assert_eq!(
                runmat_accelerate_api::handle_storage(handle),
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            );
            assert_eq!(
                runmat_accelerate_api::handle_precision(handle),
                Some(runmat_accelerate_api::ProviderPrecision::F64)
            );
            assert!(runmat_accelerate_api::provider_for_handle(handle)
                .is_some_and(|owner| std::ptr::eq(owner, provider)));
            match block_on(gpu_helpers::gather_value_async(&output)).expect("gather") {
                Value::ComplexTensor(result) => {
                    let values = result.materialize_f64();
                    assert!((values[0].0 - 180.00000000000003 * DEG_TO_RAD).abs() < 1.0e-14);
                    assert!((values[0].1 - std::f64::consts::FRAC_PI_2).abs() < 1.0e-14);
                }
                other => panic!("expected complex result, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn deg2rad_string_errors() {
        let err = deg2rad_builtin(Value::String("90".into())).expect_err("expected error");
        assert!(error_message(&err).contains("invalid input"));
        assert_eq!(err.identifier(), DEG2RAD_ERROR_INVALID_INPUT.identifier);
    }
}
