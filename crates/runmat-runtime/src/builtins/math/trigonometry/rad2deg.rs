//! MATLAB-compatible `rad2deg` builtin for RunMat.

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

const BUILTIN_NAME: &str = "rad2deg";
const RAD_TO_DEG: f64 = 180.0 / std::f64::consts::PI;

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::trigonometry::rad2deg"
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

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "rad2deg",
    category = "math/trigonometry",
    summary = "Convert angles from radians to degrees.",
    keywords = "rad2deg,radians,degrees,angle,trigonometry,gpu",
    accel = "unary",
    type_resolver(numeric_unary_type),
    builtin_path = "crate::builtins::math::trigonometry::rad2deg"
)]
async fn rad2deg_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => rad2deg_gpu(handle).await,
        Value::Complex(re, im) => Ok(Value::Complex(re * RAD_TO_DEG, im * RAD_TO_DEG)),
        Value::ComplexTensor(tensor) => rad2deg_complex_tensor(tensor),
        Value::String(_) | Value::StringArray(_) => {
            Err(builtin_error("rad2deg: expected numeric input"))
        }
        other => rad2deg_real(other),
    }
}

async fn rad2deg_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    rad2deg_tensor(tensor).map(tensor::tensor_into_value)
}

fn rad2deg_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value).map_err(builtin_error)?;
    rad2deg_tensor(tensor).map(tensor::tensor_into_value)
}

fn rad2deg_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let data = tensor
        .data
        .iter()
        .map(|&value| value * RAD_TO_DEG)
        .collect::<Vec<_>>();
    Tensor::new(data, tensor.shape.clone()).map_err(|err| builtin_error(format!("rad2deg: {err}")))
}

fn rad2deg_complex_tensor(tensor: ComplexTensor) -> BuiltinResult<Value> {
    let data = tensor
        .data
        .iter()
        .map(|&(re, im)| (re * RAD_TO_DEG, im * RAD_TO_DEG))
        .collect::<Vec<_>>();
    let converted = ComplexTensor::new(data, tensor.shape.clone())
        .map_err(|err| builtin_error(format!("rad2deg: {err}")))?;
    Ok(complex_tensor_into_value(converted))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, LogicalArray, ResolveContext, Type};

    fn rad2deg_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::rad2deg_builtin(value))
    }

    fn error_message(err: RuntimeError) -> String {
        err.message().to_string()
    }

    #[test]
    fn rad2deg_type_preserves_tensor_shape() {
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
                for (actual, expected) in tensor.data.iter().zip(expected) {
                    assert!((actual - expected).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rad2deg_int_promotes() {
        let result = rad2deg_builtin(Value::Int(IntValue::I32(1))).expect("rad2deg");
        match result {
            Value::Num(value) => assert!((value - RAD_TO_DEG).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rad2deg_logical_array_promotes() {
        let logical = LogicalArray::new(vec![0, 1], vec![1, 2]).unwrap();
        let result = rad2deg_builtin(Value::LogicalArray(logical)).expect("rad2deg");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 2]);
                assert_eq!(tensor.data[0], 0.0);
                assert!((tensor.data[1] - RAD_TO_DEG).abs() < 1e-12);
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rad2deg_string_errors() {
        let err = rad2deg_builtin(Value::String("pi".into())).expect_err("expected error");
        assert!(error_message(err).contains("rad2deg: expected numeric input"));
    }
}
