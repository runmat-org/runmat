//! MATLAB-compatible `sind` builtin for RunMat.
//!
//! `sind(x)` returns the sine of `x`, where `x` is expressed in degrees.
//! At canonical multiples of 30 and 90 degrees the result is snapped to the
//! exact rational value (`0`, `±0.5`, `±1`) so users observe MATLAB's
//! noise-free outputs instead of the floating-point drift produced by
//! `sin(x*pi/180)`.

use runmat_accelerate_api::GpuTensorHandle;
#[cfg(test)]
use runmat_builtins::SIND_DESCRIPTOR;
use runmat_builtins::{
    BuiltinErrorDescriptor, SIND_ERROR_INTERNAL, SIND_ERROR_INVALID_INPUT,
    SIND_INTEGER_INPUT_EXTENSION, SIND_LOGICAL_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexStorage, ComplexTensor, NumericDType, Tensor, Value};

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::trigonometry::degree_helpers::reduce_degrees;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "sind";
const DEG_TO_RAD: f64 = std::f64::consts::PI / 180.0;
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
    binding_variant = "default",
    builtin_path = "crate::builtins::math::trigonometry::sind"
)]
async fn sind_builtin(value: Value) -> BuiltinResult<Value> {
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        &value,
        &SIND_INTEGER_INPUT_EXTENSION,
        BUILTIN_NAME,
        "X",
    )
    .await?;
    if matches!(&value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(&value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SIND_LOGICAL_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    crate::builtins::common::validation::reject_typed_complex_integer(&value, "sind")?;
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
    if tensor.numeric_dtype() == NumericDType::F32 {
        let data = tensor
            .as_f32_slice()
            .expect("single tensor storage")
            .iter()
            .map(|&value| sind_scalar(f64::from(value)) as f32)
            .collect();
        return Tensor::from_f32(data, tensor.shape.clone())
            .map_err(|err| sind_error_with_detail(&SIND_ERROR_INTERNAL, err));
    }
    let data = tensor::tensor_values_f64_cow(&tensor)
        .iter()
        .map(|&value| sind_scalar(value))
        .collect::<Vec<_>>();
    Tensor::new(data, tensor.shape.clone())
        .map_err(|err| sind_error_with_detail(&SIND_ERROR_INTERNAL, err))
}

fn sind_complex_tensor(tensor: ComplexTensor) -> BuiltinResult<Value> {
    let shape = tensor.shape.clone();
    let converted = match tensor.into_complex_storage() {
        ComplexStorage::F32(values) => ComplexTensor::from_f32(
            values
                .into_iter()
                .map(|(re, im)| {
                    let (re, im) = sind_complex(f64::from(re), f64::from(im));
                    (re as f32, im as f32)
                })
                .collect(),
            shape,
        ),
        ComplexStorage::F64(values) => ComplexTensor::new(
            values
                .into_iter()
                .map(|(re, im)| sind_complex(re, im))
                .collect(),
            shape,
        ),
        ComplexStorage::Integer(_) => Err("typed complex integer input is unsupported".into()),
    }
    .map_err(|err| sind_error_with_detail(&SIND_ERROR_INTERNAL, err))?;
    Ok(complex_tensor_into_value(converted))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::{IntValue, LogicalArray};

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
    fn sind_integer_extension_covers_all_classes_and_exactness_boundary() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = sind_builtin(Value::Int(IntValue::U8(30)))
            .expect_err("strict mode rejects integer extension");
        assert_eq!(
            err.identifier(),
            SIND_INTEGER_INPUT_EXTENSION.error_identifier
        );
        drop(_strict);

        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        for value in [
            IntValue::I8(0),
            IntValue::I16(0),
            IntValue::I32(0),
            IntValue::I64(0),
            IntValue::U8(0),
            IntValue::U16(0),
            IntValue::U32(0),
            IntValue::U64(0),
        ] {
            assert_eq!(expect_num(sind_builtin(Value::Int(value)).unwrap()), 0.0);
        }
        assert_eq!(
            expect_num(sind_builtin(Value::Int(IntValue::I64(-90))).unwrap()),
            -1.0
        );
        assert!(sind_builtin(Value::Int(IntValue::U64(1_u64 << 63))).is_ok());
        let err = sind_builtin(Value::Int(IntValue::U64((1_u64 << 53) + 1)))
            .expect_err("inexact binary64 boundary must reject");
        assert!(err.message().contains("exactly representable as double"));
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
                assert_eq!(t.materialize_f64(), vec![0.0, 0.5, 1.0, 0.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn sind_preserves_real_and_complex_single() {
        let Value::Tensor(real) = sind_builtin(Value::Tensor(
            Tensor::from_f32(vec![30.0, 90.0], vec![1, 2]).unwrap(),
        ))
        .unwrap() else {
            panic!("expected single tensor");
        };
        assert_eq!(real.numeric_dtype(), NumericDType::F32);
        assert_eq!(real.as_f32_slice().unwrap(), &[0.5, 1.0]);

        let Value::ComplexTensor(complex) = sind_builtin(Value::ComplexTensor(
            ComplexTensor::from_f32(vec![(60.0, 30.0)], vec![1, 1]).unwrap(),
        ))
        .unwrap() else {
            panic!("expected complex single tensor");
        };
        assert_eq!(complex.numeric_dtype(), NumericDType::F32);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sind_reads_typed_integer_tensor_storage_exactly() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new_integer(
            runmat_value::IntegerStorage::I16(vec![0, 30, 90]),
            vec![3, 1],
        )
        .expect("integer tensor");

        match sind_builtin(Value::Tensor(tensor)).expect("sind") {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                let expected = [0.0, 0.5, 1.0];
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
    fn sind_logical_array_promotes() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let logical = LogicalArray::new(vec![0, 1], vec![1, 2]).unwrap();
        let result = sind_builtin(Value::LogicalArray(logical)).expect("sind");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.materialize_f64()[0], 0.0);
                let expected = (1.0_f64 * DEG_TO_RAD).sin();
                assert!((t.materialize_f64()[1] - expected).abs() < 1e-12);
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
