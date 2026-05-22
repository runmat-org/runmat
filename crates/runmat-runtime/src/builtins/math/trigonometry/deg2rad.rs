//! MATLAB-compatible `deg2rad` builtin for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BuiltinFusionSpec, ConstantStrategy, FusionError, FusionExprContext, FusionKernelTemplate,
    ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "deg2rad";
const DEG_TO_RAD: f64 = std::f64::consts::PI / 180.0;

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::trigonometry::deg2rad"
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

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "deg2rad",
    category = "math/trigonometry",
    summary = "Convert angles from degrees to radians.",
    keywords = "deg2rad,degrees,radians,angle,trigonometry,gpu",
    accel = "unary",
    type_resolver(numeric_unary_type),
    builtin_path = "crate::builtins::math::trigonometry::deg2rad"
)]
async fn deg2rad_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => deg2rad_gpu(handle).await,
        Value::Complex(re, im) => Ok(Value::Complex(re * DEG_TO_RAD, im * DEG_TO_RAD)),
        Value::ComplexTensor(tensor) => deg2rad_complex_tensor(tensor),
        Value::String(_) | Value::StringArray(_) => {
            Err(builtin_error("deg2rad: expected numeric input"))
        }
        other => deg2rad_real(other),
    }
}

async fn deg2rad_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    deg2rad_tensor(tensor).map(tensor::tensor_into_value)
}

fn deg2rad_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value).map_err(builtin_error)?;
    deg2rad_tensor(tensor).map(tensor::tensor_into_value)
}

fn deg2rad_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let data = tensor
        .data
        .iter()
        .map(|&value| value * DEG_TO_RAD)
        .collect::<Vec<_>>();
    Tensor::new(data, tensor.shape.clone()).map_err(|err| builtin_error(format!("deg2rad: {err}")))
}

fn deg2rad_complex_tensor(tensor: ComplexTensor) -> BuiltinResult<Value> {
    let data = tensor
        .data
        .iter()
        .map(|&(re, im)| (re * DEG_TO_RAD, im * DEG_TO_RAD))
        .collect::<Vec<_>>();
    let converted = ComplexTensor::new(data, tensor.shape.clone())
        .map_err(|err| builtin_error(format!("deg2rad: {err}")))?;
    Ok(complex_tensor_into_value(converted))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, LogicalArray, ResolveContext, Type};

    fn deg2rad_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::deg2rad_builtin(value))
    }

    fn error_message(err: RuntimeError) -> String {
        err.message().to_string()
    }

    #[test]
    fn deg2rad_type_preserves_tensor_shape() {
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
                for (actual, expected) in tensor.data.iter().zip(expected) {
                    assert!((actual - expected).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn deg2rad_int_promotes() {
        let result = deg2rad_builtin(Value::Int(IntValue::I32(180))).expect("deg2rad");
        match result {
            Value::Num(value) => assert!((value - std::f64::consts::PI).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn deg2rad_logical_array_promotes() {
        let logical = LogicalArray::new(vec![0, 1], vec![1, 2]).unwrap();
        let result = deg2rad_builtin(Value::LogicalArray(logical)).expect("deg2rad");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 2]);
                assert_eq!(tensor.data[0], 0.0);
                assert!((tensor.data[1] - DEG_TO_RAD).abs() < 1e-12);
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn deg2rad_string_errors() {
        let err = deg2rad_builtin(Value::String("90".into())).expect_err("expected error");
        assert!(error_message(err).contains("deg2rad: expected numeric input"));
    }
}
