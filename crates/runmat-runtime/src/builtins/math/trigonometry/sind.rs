//! MATLAB-compatible `sind` builtin for RunMat.
//!
//! `sind(x)` returns the sine of `x`, where `x` is expressed in degrees.
//! At canonical multiples of 30 and 90 degrees the result is snapped to the
//! exact rational value (`0`, `±0.5`, `±1`) so users observe MATLAB's
//! noise-free outputs instead of the floating-point drift produced by
//! `sin(x*pi/180)`.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::trigonometry::degree_helpers::reduce_degrees;
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "sind";
const DEG_TO_RAD: f64 = std::f64::consts::PI / 180.0;

const SIND_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Element-wise sine result with degree input semantics.",
}];

const SIND_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input scalar, array, logical array, complex value, or gpuArray.",
}];

const SIND_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = sind(X)",
    inputs: &SIND_INPUTS,
    outputs: &SIND_OUTPUT,
}];

const SIND_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SIND.INVALID_INPUT",
    identifier: Some("RunMat:sind:InvalidInput"),
    when: "Input cannot be interpreted as supported numeric/logical/complex data.",
    message: "sind: invalid input",
};

const SIND_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SIND.INTERNAL",
    identifier: Some("RunMat:sind:Internal"),
    when: "Internal gather/conversion/allocation flow failed.",
    message: "sind: internal error",
};

const SIND_ERRORS: [BuiltinErrorDescriptor; 2] = [SIND_ERROR_INVALID_INPUT, SIND_ERROR_INTERNAL];

pub const SIND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIND_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SIND_ERRORS,
};

fn sind_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn sind_error_with_detail(
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

/// Element-wise scalar implementation. Snaps to exact MATLAB values at
/// canonical phases and propagates NaN/Inf as NaN, matching MATLAB.
#[inline]
fn sind_scalar(x: f64) -> f64 {
    let Some(phi) = reduce_degrees(x) else {
        return f64::NAN;
    };
    // phi is in (-180, 180]
    if phi == 0.0 || phi == 180.0 {
        0.0
    } else if phi == 90.0 {
        1.0
    } else if phi == -90.0 {
        -1.0
    } else if phi == 30.0 || phi == 150.0 {
        0.5
    } else if phi == -30.0 || phi == -150.0 {
        -0.5
    } else {
        (x * DEG_TO_RAD).sin()
    }
}

/// Complex implementation mirrors `sin(z*pi/180)` using the standard
/// analytic extension; no exact-value snapping is applied because the
/// result is generically complex.
#[inline]
fn sind_complex(re: f64, im: f64) -> (f64, f64) {
    let scaled_re = re * DEG_TO_RAD;
    let scaled_im = im * DEG_TO_RAD;
    (
        scaled_re.sin() * scaled_im.cosh(),
        scaled_re.cos() * scaled_im.sinh(),
    )
}

#[runtime_builtin(
    name = "sind",
    category = "math/trigonometry",
    summary = "Compute element-wise sine values for degree-based angles.",
    keywords = "sind,sine,degrees,trigonometry",
    accel = "unary",
    type_resolver(numeric_unary_type),
    descriptor(crate::builtins::math::trigonometry::sind::SIND_DESCRIPTOR),
    builtin_path = "crate::builtins::math::trigonometry::sind"
)]
async fn sind_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => sind_gpu(handle).await,
        Value::Complex(re, im) => {
            let (out_re, out_im) = sind_complex(re, im);
            Ok(Value::Complex(out_re, out_im))
        }
        Value::ComplexTensor(ct) => sind_complex_tensor(ct),
        Value::String(_) | Value::StringArray(_) => Err(sind_error(&SIND_ERROR_INVALID_INPUT)),
        other => sind_real(other),
    }
}

async fn sind_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    sind_tensor(tensor).map(tensor::tensor_into_value)
}

fn sind_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value)
        .map_err(|e| sind_error_with_detail(&SIND_ERROR_INVALID_INPUT, e))?;
    sind_tensor(tensor).map(tensor::tensor_into_value)
}

fn sind_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let data = tensor
        .data
        .iter()
        .map(|&value| sind_scalar(value))
        .collect::<Vec<_>>();
    Tensor::new(data, tensor.shape.clone())
        .map_err(|err| sind_error_with_detail(&SIND_ERROR_INTERNAL, err))
}

fn sind_complex_tensor(tensor: ComplexTensor) -> BuiltinResult<Value> {
    let data = tensor
        .data
        .iter()
        .map(|&(re, im)| sind_complex(re, im))
        .collect::<Vec<_>>();
    let converted = ComplexTensor::new(data, tensor.shape.clone())
        .map_err(|err| sind_error_with_detail(&SIND_ERROR_INTERNAL, err))?;
    Ok(complex_tensor_into_value(converted))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, LogicalArray, ResolveContext, Type};

    fn sind_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::sind_builtin(value))
    }

    fn error_message(err: &RuntimeError) -> String {
        err.message().to_string()
    }

    #[test]
    fn sind_descriptor_signatures_cover_core_form() {
        let labels: Vec<&str> = SIND_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = sind(X)"));
    }

    fn expect_num(value: Value) -> f64 {
        match value {
            Value::Num(v) => v,
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn sind_type_preserves_tensor_shape() {
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
    fn sind_exact_values_first_period() {
        assert_eq!(expect_num(sind_builtin(Value::Num(0.0)).unwrap()), 0.0);
        assert_eq!(expect_num(sind_builtin(Value::Num(30.0)).unwrap()), 0.5);
        assert_eq!(expect_num(sind_builtin(Value::Num(90.0)).unwrap()), 1.0);
        assert_eq!(expect_num(sind_builtin(Value::Num(150.0)).unwrap()), 0.5);
        assert_eq!(expect_num(sind_builtin(Value::Num(180.0)).unwrap()), 0.0);
        assert_eq!(expect_num(sind_builtin(Value::Num(210.0)).unwrap()), -0.5);
        assert_eq!(expect_num(sind_builtin(Value::Num(270.0)).unwrap()), -1.0);
        assert_eq!(expect_num(sind_builtin(Value::Num(330.0)).unwrap()), -0.5);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sind_exact_values_negative_and_wrapped() {
        assert_eq!(expect_num(sind_builtin(Value::Num(360.0)).unwrap()), 0.0);
        assert_eq!(expect_num(sind_builtin(Value::Num(540.0)).unwrap()), 0.0);
        assert_eq!(expect_num(sind_builtin(Value::Num(-30.0)).unwrap()), -0.5);
        assert_eq!(expect_num(sind_builtin(Value::Num(-90.0)).unwrap()), -1.0);
        assert_eq!(expect_num(sind_builtin(Value::Num(-180.0)).unwrap()), 0.0);
        assert_eq!(expect_num(sind_builtin(Value::Num(450.0)).unwrap()), 1.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sind_int_input_returns_exact() {
        assert_eq!(
            expect_num(sind_builtin(Value::Int(IntValue::I32(180))).unwrap()),
            0.0,
        );
        assert_eq!(
            expect_num(sind_builtin(Value::Int(IntValue::I32(30))).unwrap()),
            0.5,
        );
        assert_eq!(
            expect_num(sind_builtin(Value::Int(IntValue::I64(-90))).unwrap()),
            -1.0,
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sind_non_exact_value_matches_radian_formula() {
        let degrees = 45.0_f64;
        let actual = expect_num(sind_builtin(Value::Num(degrees)).unwrap());
        let expected = (degrees * DEG_TO_RAD).sin();
        assert!((actual - expected).abs() < 1e-12);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sind_tensor_preserves_shape() {
        let tensor = Tensor::new(vec![0.0, 30.0, 90.0, 180.0], vec![2, 2]).unwrap();
        let result = sind_builtin(Value::Tensor(tensor)).expect("sind");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![0.0, 0.5, 1.0, 0.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sind_logical_array_promotes() {
        let logical = LogicalArray::new(vec![0, 1], vec![1, 2]).unwrap();
        let result = sind_builtin(Value::LogicalArray(logical)).expect("sind");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.data[0], 0.0);
                let expected = (1.0_f64 * DEG_TO_RAD).sin();
                assert!((t.data[1] - expected).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sind_nan_propagates() {
        let result = expect_num(sind_builtin(Value::Num(f64::NAN)).unwrap());
        assert!(result.is_nan());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sind_inf_is_nan() {
        let pos = expect_num(sind_builtin(Value::Num(f64::INFINITY)).unwrap());
        let neg = expect_num(sind_builtin(Value::Num(f64::NEG_INFINITY)).unwrap());
        assert!(pos.is_nan());
        assert!(neg.is_nan());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sind_complex_uses_radian_formula() {
        let result = sind_builtin(Value::Complex(180.0, 0.0)).expect("sind");
        match result {
            Value::Complex(re, im) => {
                let (expected_re, expected_im) = sind_complex(180.0, 0.0);
                assert!((re - expected_re).abs() < 1e-15);
                assert!((im - expected_im).abs() < 1e-15);
                // imag is exactly zero on the real axis
                assert_eq!(im, 0.0);
                // real part is sin(pi), which is small but not snapped to zero
                assert!(re.abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sind_complex_off_axis_matches_formula() {
        let result = sind_builtin(Value::Complex(60.0, 30.0)).expect("sind");
        match result {
            Value::Complex(re, im) => {
                let (expected_re, expected_im) = sind_complex(60.0, 30.0);
                assert!((re - expected_re).abs() < 1e-12);
                assert!((im - expected_im).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sind_string_errors() {
        let err = sind_builtin(Value::String("90".into())).expect_err("expected error");
        assert!(error_message(&err).contains("invalid input"));
        assert_eq!(err.identifier(), SIND_ERROR_INVALID_INPUT.identifier);
    }
}
