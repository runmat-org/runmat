//! MATLAB-compatible `tand` builtin for RunMat.
//!
//! `tand(x)` returns the tangent of `x`, where `x` is expressed in degrees.
//! At canonical multiples of 45 degrees the result is snapped to the exact
//! rational value (`0`, `±1`), and at odd multiples of 90 degrees the
//! result is `±Inf` (MATLAB returns `Inf` for `tand(90 + 180k)` and `-Inf`
//! for `tand(-90 + 180k)`). Non-finite inputs propagate as `NaN`.

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

const BUILTIN_NAME: &str = "tand";
const DEG_TO_RAD: f64 = std::f64::consts::PI / 180.0;

const TAND_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Element-wise tangent result with degree input semantics.",
}];

const TAND_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input scalar, array, logical array, complex value, or gpuArray.",
}];

const TAND_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = tand(X)",
    inputs: &TAND_INPUTS,
    outputs: &TAND_OUTPUT,
}];

const TAND_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TAND.INVALID_INPUT",
    identifier: Some("RunMat:tand:InvalidInput"),
    when: "Input cannot be interpreted as supported numeric/logical/complex data.",
    message: "tand: invalid input",
};

const TAND_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TAND.INTERNAL",
    identifier: Some("RunMat:tand:Internal"),
    when: "Internal gather/conversion/allocation flow failed.",
    message: "tand: internal error",
};

const TAND_ERRORS: [BuiltinErrorDescriptor; 2] = [TAND_ERROR_INVALID_INPUT, TAND_ERROR_INTERNAL];

pub const TAND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TAND_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TAND_ERRORS,
};

fn tand_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn tand_error_with_detail(
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
/// canonical phases and emits `±Inf` at the tangent poles, matching MATLAB.
#[inline]
fn tand_scalar(x: f64) -> f64 {
    let Some(phi) = reduce_degrees(x) else {
        return f64::NAN;
    };
    // phi is in (-180, 180]
    if phi == 0.0 || phi == 180.0 {
        0.0
    } else if phi == 90.0 {
        f64::INFINITY
    } else if phi == -90.0 {
        f64::NEG_INFINITY
    } else if phi == 45.0 || phi == -135.0 {
        1.0
    } else if phi == 135.0 || phi == -45.0 {
        -1.0
    } else {
        (x * DEG_TO_RAD).tan()
    }
}

/// Complex implementation mirrors `tan(z*pi/180)` using the standard
/// analytic extension; no exact-value snapping is applied because the
/// result is generically complex.
#[inline]
fn tand_complex(re: f64, im: f64) -> (f64, f64) {
    let scaled_re = re * DEG_TO_RAD;
    let scaled_im = im * DEG_TO_RAD;
    let two_re = 2.0 * scaled_re;
    let two_im = 2.0 * scaled_im;
    let denom = two_re.cos() + two_im.cosh();
    (two_re.sin() / denom, two_im.sinh() / denom)
}

#[runtime_builtin(
    name = "tand",
    category = "math/trigonometry",
    summary = "Compute element-wise tangent values for degree-based angles.",
    keywords = "tand,tangent,degrees,trigonometry",
    accel = "unary",
    type_resolver(numeric_unary_type),
    descriptor(crate::builtins::math::trigonometry::tand::TAND_DESCRIPTOR),
    builtin_path = "crate::builtins::math::trigonometry::tand"
)]
async fn tand_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => tand_gpu(handle).await,
        Value::Complex(re, im) => {
            let (out_re, out_im) = tand_complex(re, im);
            Ok(Value::Complex(out_re, out_im))
        }
        Value::ComplexTensor(ct) => tand_complex_tensor(ct),
        Value::String(_) | Value::StringArray(_) => Err(tand_error(&TAND_ERROR_INVALID_INPUT)),
        other => tand_real(other),
    }
}

async fn tand_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    tand_tensor(tensor).map(tensor::tensor_into_value)
}

fn tand_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value)
        .map_err(|e| tand_error_with_detail(&TAND_ERROR_INVALID_INPUT, e))?;
    tand_tensor(tensor).map(tensor::tensor_into_value)
}

fn tand_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let data = tensor
        .data
        .iter()
        .map(|&value| tand_scalar(value))
        .collect::<Vec<_>>();
    Tensor::new(data, tensor.shape.clone())
        .map_err(|err| tand_error_with_detail(&TAND_ERROR_INTERNAL, err))
}

fn tand_complex_tensor(tensor: ComplexTensor) -> BuiltinResult<Value> {
    let data = tensor
        .data
        .iter()
        .map(|&(re, im)| tand_complex(re, im))
        .collect::<Vec<_>>();
    let converted = ComplexTensor::new(data, tensor.shape.clone())
        .map_err(|err| tand_error_with_detail(&TAND_ERROR_INTERNAL, err))?;
    Ok(complex_tensor_into_value(converted))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, LogicalArray, ResolveContext, Type};

    fn tand_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::tand_builtin(value))
    }

    fn error_message(err: &RuntimeError) -> String {
        err.message().to_string()
    }

    #[test]
    fn tand_descriptor_signatures_cover_core_form() {
        let labels: Vec<&str> = TAND_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = tand(X)"));
    }

    fn expect_num(value: Value) -> f64 {
        match value {
            Value::Num(v) => v,
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn tand_type_preserves_tensor_shape() {
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
    fn tand_exact_values_first_period() {
        assert_eq!(expect_num(tand_builtin(Value::Num(0.0)).unwrap()), 0.0);
        assert_eq!(expect_num(tand_builtin(Value::Num(45.0)).unwrap()), 1.0);
        assert_eq!(expect_num(tand_builtin(Value::Num(135.0)).unwrap()), -1.0);
        assert_eq!(expect_num(tand_builtin(Value::Num(180.0)).unwrap()), 0.0);
        assert_eq!(expect_num(tand_builtin(Value::Num(225.0)).unwrap()), 1.0);
        assert_eq!(expect_num(tand_builtin(Value::Num(315.0)).unwrap()), -1.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tand_poles_emit_signed_infinity() {
        let pos = expect_num(tand_builtin(Value::Num(90.0)).unwrap());
        let neg = expect_num(tand_builtin(Value::Num(-90.0)).unwrap());
        let pos2 = expect_num(tand_builtin(Value::Num(450.0)).unwrap());
        let neg2 = expect_num(tand_builtin(Value::Num(270.0)).unwrap());
        assert_eq!(pos, f64::INFINITY);
        assert_eq!(neg, f64::NEG_INFINITY);
        assert_eq!(pos2, f64::INFINITY);
        assert_eq!(neg2, f64::NEG_INFINITY);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tand_exact_values_negative_wrapped() {
        assert_eq!(expect_num(tand_builtin(Value::Num(360.0)).unwrap()), 0.0);
        assert_eq!(expect_num(tand_builtin(Value::Num(-45.0)).unwrap()), -1.0);
        assert_eq!(expect_num(tand_builtin(Value::Num(-135.0)).unwrap()), 1.0);
        assert_eq!(expect_num(tand_builtin(Value::Num(-180.0)).unwrap()), 0.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tand_int_input_returns_exact() {
        assert_eq!(
            expect_num(tand_builtin(Value::Int(IntValue::I32(45))).unwrap()),
            1.0,
        );
        assert_eq!(
            expect_num(tand_builtin(Value::Int(IntValue::I32(0))).unwrap()),
            0.0,
        );
        assert_eq!(
            expect_num(tand_builtin(Value::Int(IntValue::I32(90))).unwrap()),
            f64::INFINITY,
        );
        assert_eq!(
            expect_num(tand_builtin(Value::Int(IntValue::I64(-90))).unwrap()),
            f64::NEG_INFINITY,
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tand_non_exact_value_matches_radian_formula() {
        let degrees = 30.0_f64;
        let actual = expect_num(tand_builtin(Value::Num(degrees)).unwrap());
        let expected = (degrees * DEG_TO_RAD).tan();
        assert!((actual - expected).abs() < 1e-12);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tand_tensor_preserves_shape() {
        let tensor = Tensor::new(vec![0.0, 45.0, 90.0, 135.0], vec![2, 2]).unwrap();
        let result = tand_builtin(Value::Tensor(tensor)).expect("tand");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data[0], 0.0);
                assert_eq!(t.data[1], 1.0);
                assert_eq!(t.data[2], f64::INFINITY);
                assert_eq!(t.data[3], -1.0);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tand_logical_array_promotes() {
        let logical = LogicalArray::new(vec![0, 1], vec![1, 2]).unwrap();
        let result = tand_builtin(Value::LogicalArray(logical)).expect("tand");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.data[0], 0.0);
                let expected = (1.0_f64 * DEG_TO_RAD).tan();
                assert!((t.data[1] - expected).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tand_nan_propagates() {
        let result = expect_num(tand_builtin(Value::Num(f64::NAN)).unwrap());
        assert!(result.is_nan());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tand_inf_is_nan() {
        let pos = expect_num(tand_builtin(Value::Num(f64::INFINITY)).unwrap());
        let neg = expect_num(tand_builtin(Value::Num(f64::NEG_INFINITY)).unwrap());
        assert!(pos.is_nan());
        assert!(neg.is_nan());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tand_complex_uses_radian_formula() {
        let result = tand_builtin(Value::Complex(45.0, 0.0)).expect("tand");
        match result {
            Value::Complex(re, im) => {
                let (expected_re, expected_im) = tand_complex(45.0, 0.0);
                assert!((re - expected_re).abs() < 1e-15);
                assert!((im - expected_im).abs() < 1e-15);
                // imag is zero on the real axis
                assert_eq!(im, 0.0);
                // tan(pi/4) ~= 1.0 but no exact snapping for complex
                assert!((re - 1.0).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tand_complex_off_axis_matches_formula() {
        let result = tand_builtin(Value::Complex(30.0, 20.0)).expect("tand");
        match result {
            Value::Complex(re, im) => {
                let (expected_re, expected_im) = tand_complex(30.0, 20.0);
                assert!((re - expected_re).abs() < 1e-12);
                assert!((im - expected_im).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tand_string_errors() {
        let err = tand_builtin(Value::String("90".into())).expect_err("expected error");
        assert!(error_message(&err).contains("invalid input"));
        assert_eq!(err.identifier(), TAND_ERROR_INVALID_INPUT.identifier);
    }
}
