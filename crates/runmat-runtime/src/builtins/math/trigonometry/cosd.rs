//! MATLAB-compatible `cosd` builtin for RunMat.
//!
//! `cosd(x)` returns the cosine of `x`, where `x` is expressed in degrees.
//! At canonical multiples of 60 and 90 degrees the result is snapped to the
//! exact rational value (`0`, `±0.5`, `±1`) so users observe MATLAB's
//! noise-free outputs instead of the floating-point drift produced by
//! `cos(x*pi/180)`.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::trigonometry::degree_helpers::reduce_degrees;
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "cosd";
const DEG_TO_RAD: f64 = std::f64::consts::PI / 180.0;

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

/// Element-wise scalar implementation with exact-value snapping at
/// canonical angles and NaN propagation for non-finite inputs.
#[inline]
fn cosd_scalar(x: f64) -> f64 {
    let Some(phi) = reduce_degrees(x) else {
        return f64::NAN;
    };
    // phi is in (-180, 180]
    if phi == 0.0 {
        1.0
    } else if phi == 180.0 {
        -1.0
    } else if phi == 90.0 || phi == -90.0 {
        0.0
    } else if phi == 60.0 || phi == -60.0 {
        0.5
    } else if phi == 120.0 || phi == -120.0 {
        -0.5
    } else {
        (x * DEG_TO_RAD).cos()
    }
}

/// Complex implementation mirrors `cos(z*pi/180)` using the standard
/// analytic extension; no exact-value snapping is applied because the
/// result is generically complex.
#[inline]
fn cosd_complex(re: f64, im: f64) -> (f64, f64) {
    let scaled_re = re * DEG_TO_RAD;
    let scaled_im = im * DEG_TO_RAD;
    (
        scaled_re.cos() * scaled_im.cosh(),
        -scaled_re.sin() * scaled_im.sinh(),
    )
}

#[runtime_builtin(
    name = "cosd",
    category = "math/trigonometry",
    summary = "Cosine of input expressed in degrees.",
    keywords = "cosd,cosine,degrees,trigonometry",
    accel = "unary",
    type_resolver(numeric_unary_type),
    builtin_path = "crate::builtins::math::trigonometry::cosd"
)]
async fn cosd_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => cosd_gpu(handle).await,
        Value::Complex(re, im) => {
            let (out_re, out_im) = cosd_complex(re, im);
            Ok(Value::Complex(out_re, out_im))
        }
        Value::ComplexTensor(ct) => cosd_complex_tensor(ct),
        Value::String(_) | Value::StringArray(_) => {
            Err(builtin_error("cosd: expected numeric input"))
        }
        other => cosd_real(other),
    }
}

async fn cosd_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    cosd_tensor(tensor).map(tensor::tensor_into_value)
}

fn cosd_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value).map_err(builtin_error)?;
    cosd_tensor(tensor).map(tensor::tensor_into_value)
}

fn cosd_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let data = tensor
        .data
        .iter()
        .map(|&value| cosd_scalar(value))
        .collect::<Vec<_>>();
    Tensor::new(data, tensor.shape.clone()).map_err(|err| builtin_error(format!("cosd: {err}")))
}

fn cosd_complex_tensor(tensor: ComplexTensor) -> BuiltinResult<Value> {
    let data = tensor
        .data
        .iter()
        .map(|&(re, im)| cosd_complex(re, im))
        .collect::<Vec<_>>();
    let converted = ComplexTensor::new(data, tensor.shape.clone())
        .map_err(|err| builtin_error(format!("cosd: {err}")))?;
    Ok(complex_tensor_into_value(converted))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, LogicalArray, ResolveContext, Type};

    fn cosd_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::cosd_builtin(value))
    }

    fn error_message(err: RuntimeError) -> String {
        err.message().to_string()
    }

    fn expect_num(value: Value) -> f64 {
        match value {
            Value::Num(v) => v,
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn cosd_type_preserves_tensor_shape() {
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
    fn cosd_exact_values_first_period() {
        assert_eq!(expect_num(cosd_builtin(Value::Num(0.0)).unwrap()), 1.0);
        assert_eq!(expect_num(cosd_builtin(Value::Num(60.0)).unwrap()), 0.5);
        assert_eq!(expect_num(cosd_builtin(Value::Num(90.0)).unwrap()), 0.0);
        assert_eq!(expect_num(cosd_builtin(Value::Num(120.0)).unwrap()), -0.5);
        assert_eq!(expect_num(cosd_builtin(Value::Num(180.0)).unwrap()), -1.0);
        assert_eq!(expect_num(cosd_builtin(Value::Num(240.0)).unwrap()), -0.5);
        assert_eq!(expect_num(cosd_builtin(Value::Num(270.0)).unwrap()), 0.0);
        assert_eq!(expect_num(cosd_builtin(Value::Num(300.0)).unwrap()), 0.5);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cosd_exact_values_negative_and_wrapped() {
        assert_eq!(expect_num(cosd_builtin(Value::Num(360.0)).unwrap()), 1.0);
        assert_eq!(expect_num(cosd_builtin(Value::Num(540.0)).unwrap()), -1.0);
        assert_eq!(expect_num(cosd_builtin(Value::Num(-90.0)).unwrap()), 0.0);
        assert_eq!(expect_num(cosd_builtin(Value::Num(-60.0)).unwrap()), 0.5);
        assert_eq!(expect_num(cosd_builtin(Value::Num(-180.0)).unwrap()), -1.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cosd_int_input_returns_exact() {
        assert_eq!(
            expect_num(cosd_builtin(Value::Int(IntValue::I32(90))).unwrap()),
            0.0,
        );
        assert_eq!(
            expect_num(cosd_builtin(Value::Int(IntValue::I32(0))).unwrap()),
            1.0,
        );
        assert_eq!(
            expect_num(cosd_builtin(Value::Int(IntValue::I64(-180))).unwrap()),
            -1.0,
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cosd_non_exact_value_matches_radian_formula() {
        let degrees = 45.0_f64;
        let actual = expect_num(cosd_builtin(Value::Num(degrees)).unwrap());
        let expected = (degrees * DEG_TO_RAD).cos();
        assert!((actual - expected).abs() < 1e-12);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cosd_tensor_preserves_shape() {
        let tensor = Tensor::new(vec![0.0, 90.0, 180.0, 270.0], vec![2, 2]).unwrap();
        let result = cosd_builtin(Value::Tensor(tensor)).expect("cosd");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![1.0, 0.0, -1.0, 0.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cosd_logical_array_promotes() {
        let logical = LogicalArray::new(vec![0, 1], vec![1, 2]).unwrap();
        let result = cosd_builtin(Value::LogicalArray(logical)).expect("cosd");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.data[0], 1.0);
                let expected = (1.0_f64 * DEG_TO_RAD).cos();
                assert!((t.data[1] - expected).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cosd_nan_propagates() {
        let result = expect_num(cosd_builtin(Value::Num(f64::NAN)).unwrap());
        assert!(result.is_nan());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cosd_inf_is_nan() {
        let pos = expect_num(cosd_builtin(Value::Num(f64::INFINITY)).unwrap());
        let neg = expect_num(cosd_builtin(Value::Num(f64::NEG_INFINITY)).unwrap());
        assert!(pos.is_nan());
        assert!(neg.is_nan());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cosd_complex_uses_radian_formula() {
        let result = cosd_builtin(Value::Complex(90.0, 0.0)).expect("cosd");
        match result {
            Value::Complex(re, im) => {
                let (expected_re, expected_im) = cosd_complex(90.0, 0.0);
                assert!((re - expected_re).abs() < 1e-15);
                assert!((im - expected_im).abs() < 1e-15);
                // imag is exactly zero on the real axis
                assert_eq!(im, 0.0);
                // cos(pi/2) is small but not snapped to zero for complex inputs
                assert!(re.abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cosd_complex_off_axis_matches_formula() {
        let result = cosd_builtin(Value::Complex(30.0, 45.0)).expect("cosd");
        match result {
            Value::Complex(re, im) => {
                let (expected_re, expected_im) = cosd_complex(30.0, 45.0);
                assert!((re - expected_re).abs() < 1e-12);
                assert!((im - expected_im).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cosd_string_errors() {
        let err = cosd_builtin(Value::String("90".into())).expect_err("expected error");
        assert!(error_message(err).contains("cosd: expected numeric input"));
    }
}
