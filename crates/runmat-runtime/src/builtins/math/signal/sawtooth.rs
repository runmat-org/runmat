//! MATLAB-compatible `sawtooth` builtin for RunMat.
//!
//! `y = sawtooth(t)` evaluates a sawtooth waveform with period `2*pi` at the
//! sample times in `t`. The optional second argument `xmax ∈ [0, 1]` controls
//! the position of the peak inside each period: `xmax = 1` (default) produces
//! a rising sawtooth, `xmax = 0` produces a falling sawtooth, and any value
//! in between (e.g. `xmax = 0.5` for a triangle wave) interpolates between
//! the two via a piecewise-linear ramp.

use std::f64::consts::PI;

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::tensor::{scalar_f64_from_value_async, tensor_into_value};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "sawtooth";
const TWO_PI: f64 = 2.0 * PI;

const SAWTOOTH_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sawtooth output sampled at t.",
}];

const SAWTOOTH_SIG_DEFAULT_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "t",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sample times.",
}];

const SAWTOOTH_SIG_XMAX_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "t",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample times.",
    },
    BuiltinParamDescriptor {
        name: "xmax",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("1"),
        description: "Peak position fraction in [0, 1].",
    },
];

const SAWTOOTH_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = sawtooth(t)",
        inputs: &SAWTOOTH_SIG_DEFAULT_INPUTS,
        outputs: &SAWTOOTH_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = sawtooth(t, xmax)",
        inputs: &SAWTOOTH_SIG_XMAX_INPUTS,
        outputs: &SAWTOOTH_OUTPUT,
    },
];

const SAWTOOTH_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAWTOOTH.INVALID_INPUT",
    identifier: Some("RunMat:sawtooth:InvalidInput"),
    when: "Primary input is not numeric-real tensor/scalar compatible.",
    message: "sawtooth: expected numeric input",
};

const SAWTOOTH_ERROR_COMPLEX_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAWTOOTH.COMPLEX_UNSUPPORTED",
    identifier: Some("RunMat:sawtooth:ComplexUnsupported"),
    when: "Primary input includes complex values.",
    message: "sawtooth: input must be real; complex values are not supported",
};

const SAWTOOTH_ERROR_XMAX_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAWTOOTH.XMAX_INVALID",
    identifier: Some("RunMat:sawtooth:XmaxInvalid"),
    when: "xmax argument is not a real numeric scalar.",
    message: "sawtooth: xmax must be a real numeric scalar in [0, 1]",
};

const SAWTOOTH_ERROR_XMAX_RANGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAWTOOTH.XMAX_RANGE",
    identifier: Some("RunMat:sawtooth:XmaxOutOfRange"),
    when: "xmax argument lies outside [0, 1] or is non-finite.",
    message: "sawtooth: xmax must be a finite scalar in [0, 1]",
};

const SAWTOOTH_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAWTOOTH.ARG_COUNT",
    identifier: Some("RunMat:sawtooth:ArgCount"),
    when: "More than one optional argument is provided.",
    message: "sawtooth: expected 1 or 2 arguments",
};

const SAWTOOTH_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAWTOOTH.INTERNAL",
    identifier: Some("RunMat:sawtooth:InternalError"),
    when: "Internal tensor construction or GPU gather fails.",
    message: "sawtooth: internal error",
};

const SAWTOOTH_ERRORS: [BuiltinErrorDescriptor; 6] = [
    SAWTOOTH_ERROR_INVALID_INPUT,
    SAWTOOTH_ERROR_COMPLEX_UNSUPPORTED,
    SAWTOOTH_ERROR_XMAX_INVALID,
    SAWTOOTH_ERROR_XMAX_RANGE,
    SAWTOOTH_ERROR_ARG_COUNT,
    SAWTOOTH_ERROR_INTERNAL,
];

pub const SAWTOOTH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SAWTOOTH_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SAWTOOTH_ERRORS,
};

fn sawtooth_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    sawtooth_error_with_message(error.message, error)
}

fn sawtooth_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    sawtooth_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn sawtooth_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn sawtooth_error_with_source(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
    source: RuntimeError,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

/// Element-wise scalar sawtooth.
///
/// `width = xmax * 2π` is the peak position within the period. For phase
/// values in `[0, width)` the output rises linearly from `-1` to `+1`. For
/// phase values in `[width, 2π)` the output falls linearly from `+1` back to
/// `-1`. The boundary cases `xmax == 0` (pure falling) and `xmax == 1` (pure
/// rising) reduce naturally because `phi ∈ [0, 2π)` never reaches the open
/// upper bound of the rising branch.
#[inline]
fn sawtooth_scalar(t: f64, xmax: f64) -> f64 {
    if !t.is_finite() {
        return f64::NAN;
    }
    let phi = t.rem_euclid(TWO_PI);
    let width = xmax * TWO_PI;
    if width <= 0.0 {
        1.0 - phi / PI
    } else if phi < width {
        -1.0 + 2.0 * phi / width
    } else {
        let falling_width = TWO_PI - width;
        if falling_width <= 0.0 {
            -1.0 + 2.0 * phi / width
        } else {
            1.0 - 2.0 * (phi - width) / falling_width
        }
    }
}

#[runtime_builtin(
    name = "sawtooth",
    category = "math/signal",
    summary = "Generate a periodic sawtooth waveform with optional peak position.",
    keywords = "sawtooth,waveform,signal processing,triangle,periodic",
    type_resolver(numeric_unary_type),
    descriptor(crate::builtins::math::signal::sawtooth::SAWTOOTH_DESCRIPTOR),
    builtin_path = "crate::builtins::math::signal::sawtooth"
)]
async fn sawtooth_builtin(t: Value, varargin: Vec<Value>) -> BuiltinResult<Value> {
    let xmax = parse_xmax(&varargin).await?;
    match t {
        Value::GpuTensor(handle) => sawtooth_gpu(handle, xmax).await,
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err(sawtooth_error(&SAWTOOTH_ERROR_COMPLEX_UNSUPPORTED))
        }
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
            Err(sawtooth_error(&SAWTOOTH_ERROR_INVALID_INPUT))
        }
        other => sawtooth_real(other, xmax),
    }
}

async fn parse_xmax(varargin: &[Value]) -> BuiltinResult<f64> {
    match varargin.len() {
        0 => Ok(1.0),
        1 => {
            let raw = scalar_f64_from_value_async(&varargin[0])
                .await
                .map_err(|err| sawtooth_error_with_detail(&SAWTOOTH_ERROR_XMAX_INVALID, err))?
                .ok_or_else(|| sawtooth_error(&SAWTOOTH_ERROR_XMAX_INVALID))?;
            if !raw.is_finite() || !(0.0..=1.0).contains(&raw) {
                return Err(sawtooth_error_with_detail(
                    &SAWTOOTH_ERROR_XMAX_RANGE,
                    format!("got {raw}"),
                ));
            }
            Ok(raw)
        }
        _ => Err(sawtooth_error_with_detail(
            &SAWTOOTH_ERROR_ARG_COUNT,
            format!("got {}", varargin.len() + 1),
        )),
    }
}

async fn sawtooth_gpu(handle: GpuTensorHandle, xmax: f64) -> BuiltinResult<Value> {
    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|source| {
            sawtooth_error_with_source(&SAWTOOTH_ERROR_INTERNAL, "gpu gather failed", source)
        })?;
    sawtooth_tensor(tensor, xmax).map(tensor_into_value)
}

fn sawtooth_real(value: Value, xmax: f64) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value)
        .map_err(|err| sawtooth_error_with_detail(&SAWTOOTH_ERROR_INVALID_INPUT, err))?;
    sawtooth_tensor(tensor, xmax).map(tensor_into_value)
}

fn sawtooth_tensor(tensor: Tensor, xmax: f64) -> BuiltinResult<Tensor> {
    let data = tensor
        .data
        .iter()
        .map(|&value| sawtooth_scalar(value, xmax))
        .collect::<Vec<_>>();
    Tensor::new(data, tensor.shape.clone())
        .map_err(|err| sawtooth_error_with_detail(&SAWTOOTH_ERROR_INTERNAL, &err))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{builtin_function_by_name, IntValue, LogicalArray, ResolveContext, Type};

    fn call(t: Value) -> BuiltinResult<Value> {
        block_on(sawtooth_builtin(t, Vec::new()))
    }

    fn call_with_xmax(t: Value, xmax: Value) -> BuiltinResult<Value> {
        block_on(sawtooth_builtin(t, vec![xmax]))
    }

    fn expect_num(value: Value) -> f64 {
        match value {
            Value::Num(v) => v,
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    fn expect_tensor(value: Value) -> Tensor {
        match value {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    fn assert_close(got: f64, want: f64) {
        assert!(
            (got - want).abs() < 1e-12,
            "got {got}, expected {want} (diff {})",
            (got - want).abs()
        );
    }

    #[test]
    fn sawtooth_type_preserves_tensor_shape() {
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

    #[test]
    fn sawtooth_descriptor_signatures_and_errors() {
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("sawtooth builtin");
        let descriptor = builtin.descriptor.expect("sawtooth descriptor");
        let labels: Vec<&str> = descriptor.signatures.iter().map(|sig| sig.label).collect();
        assert!(labels.contains(&"Y = sawtooth(t)"));
        assert!(labels.contains(&"Y = sawtooth(t, xmax)"));
        assert!(descriptor
            .errors
            .iter()
            .any(|err| err.code == "RM.SAWTOOTH.XMAX_RANGE"));
    }

    #[test]
    fn sawtooth_default_is_rising_within_first_period() {
        assert_close(expect_num(call(Value::Num(0.0)).unwrap()), -1.0);
        assert_close(expect_num(call(Value::Num(PI / 2.0)).unwrap()), -0.5);
        assert_close(expect_num(call(Value::Num(PI)).unwrap()), 0.0);
        assert_close(expect_num(call(Value::Num(3.0 * PI / 2.0)).unwrap()), 0.5);
    }

    #[test]
    fn sawtooth_period_boundary_wraps_to_minus_one() {
        assert_close(expect_num(call(Value::Num(TWO_PI)).unwrap()), -1.0);
        assert_close(expect_num(call(Value::Num(4.0 * PI)).unwrap()), -1.0);
        assert_close(expect_num(call(Value::Num(-TWO_PI)).unwrap()), -1.0);
    }

    #[test]
    fn sawtooth_negative_time_wraps_into_period() {
        assert_close(expect_num(call(Value::Num(-PI)).unwrap()), 0.0);
        assert_close(expect_num(call(Value::Num(-PI / 2.0)).unwrap()), 0.5);
    }

    #[test]
    fn sawtooth_vector_two_periods_ranges_from_minus_one_to_just_below_one() {
        let n: usize = 100;
        let total = 4.0 * PI;
        let step = total / (n as f64 - 1.0);
        let data: Vec<f64> = (0..n).map(|i| i as f64 * step).collect();
        let tensor = Tensor::new(data.clone(), vec![1, n]).unwrap();
        let result = expect_tensor(call(Value::Tensor(tensor)).unwrap());
        assert_eq!(result.shape, vec![1, n]);
        let min = result.data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = result
            .data
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (-1.0 - 1e-12..=-1.0 + 1e-12).contains(&min),
            "min should be -1, got {min}"
        );
        assert!(
            max <= 1.0 + 1e-12 && max > 0.95,
            "max should approach 1 from below, got {max}"
        );
        // First and last samples land exactly on period boundaries (t=0 and t=4*pi)
        // so the rising sawtooth resets to -1 at both ends.
        assert_close(result.data[0], -1.0);
        assert_close(*result.data.last().unwrap(), -1.0);

        // Each sample must satisfy the elementwise formula.
        for (idx, (&t, &y)) in data.iter().zip(result.data.iter()).enumerate() {
            assert_close(y, sawtooth_scalar(t, 1.0));
            if !y.is_finite() {
                panic!("non-finite sample at index {idx}");
            }
        }

        // There should be exactly two period boundaries in (0, 4*pi]: the inner
        // reset near index 50 (sample just after 2*pi) and the closing sample.
        let reset_count = result
            .data
            .iter()
            .enumerate()
            .filter(|(idx, &y)| *idx > 0 && y < result.data[idx - 1])
            .count();
        assert_eq!(
            reset_count, 2,
            "expected two period resets across two periods"
        );
    }

    #[test]
    fn sawtooth_xmax_half_is_triangle_wave_with_peak_at_pi() {
        let half = Value::Num(0.5);
        assert_close(
            expect_num(call_with_xmax(Value::Num(0.0), half.clone()).unwrap()),
            -1.0,
        );
        assert_close(
            expect_num(call_with_xmax(Value::Num(PI / 2.0), half.clone()).unwrap()),
            0.0,
        );
        assert_close(
            expect_num(call_with_xmax(Value::Num(PI), half.clone()).unwrap()),
            1.0,
        );
        assert_close(
            expect_num(call_with_xmax(Value::Num(3.0 * PI / 2.0), half.clone()).unwrap()),
            0.0,
        );
        assert_close(
            expect_num(call_with_xmax(Value::Num(TWO_PI), half).unwrap()),
            -1.0,
        );
    }

    #[test]
    fn sawtooth_xmax_zero_is_pure_falling() {
        let zero = Value::Num(0.0);
        assert_close(
            expect_num(call_with_xmax(Value::Num(0.0), zero.clone()).unwrap()),
            1.0,
        );
        assert_close(
            expect_num(call_with_xmax(Value::Num(PI), zero.clone()).unwrap()),
            0.0,
        );
        assert_close(
            expect_num(call_with_xmax(Value::Num(TWO_PI), zero).unwrap()),
            1.0,
        );
    }

    #[test]
    fn sawtooth_xmax_one_is_pure_rising() {
        let one = Value::Num(1.0);
        assert_close(
            expect_num(call_with_xmax(Value::Num(0.0), one.clone()).unwrap()),
            -1.0,
        );
        assert_close(
            expect_num(call_with_xmax(Value::Num(PI), one.clone()).unwrap()),
            0.0,
        );
        // The default call (no xmax) must agree with the explicit xmax = 1 form.
        for &t in &[-3.7, -1.0, 0.0, 0.25, PI, 5.5, 9.0] {
            assert_close(
                expect_num(call(Value::Num(t)).unwrap()),
                expect_num(call_with_xmax(Value::Num(t), one.clone()).unwrap()),
            );
        }
    }

    #[test]
    fn sawtooth_xmax_rejects_out_of_range() {
        assert!(call_with_xmax(Value::Num(0.0), Value::Num(-0.1)).is_err());
        assert!(call_with_xmax(Value::Num(0.0), Value::Num(1.1)).is_err());
        assert!(call_with_xmax(Value::Num(0.0), Value::Num(f64::NAN)).is_err());
        assert!(call_with_xmax(Value::Num(0.0), Value::Num(f64::INFINITY)).is_err());
    }

    #[test]
    fn sawtooth_xmax_rejects_non_scalar() {
        let tensor = Tensor::new(vec![0.5, 1.0], vec![1, 2]).unwrap();
        assert!(call_with_xmax(Value::Num(0.0), Value::Tensor(tensor)).is_err());
    }

    #[test]
    fn sawtooth_int_and_logical_promote_to_double() {
        let int_result = expect_num(call(Value::Int(IntValue::I32(0))).unwrap());
        assert_close(int_result, -1.0);

        let bool_result = expect_num(call(Value::Bool(false)).unwrap());
        assert_close(bool_result, -1.0);

        let logical = LogicalArray::new(vec![0, 1], vec![1, 2]).unwrap();
        let result = expect_tensor(call(Value::LogicalArray(logical)).unwrap());
        assert_eq!(result.shape, vec![1, 2]);
        assert_close(result.data[0], -1.0);
        assert_close(result.data[1], sawtooth_scalar(1.0, 1.0));
    }

    #[test]
    fn sawtooth_nonfinite_inputs_return_nan() {
        assert!(expect_num(call(Value::Num(f64::NAN)).unwrap()).is_nan());
        assert!(expect_num(call(Value::Num(f64::INFINITY)).unwrap()).is_nan());
        assert!(expect_num(call(Value::Num(f64::NEG_INFINITY)).unwrap()).is_nan());
    }

    #[test]
    fn sawtooth_complex_input_errors() {
        let err = call(Value::Complex(1.0, 2.0)).expect_err("complex sawtooth should error");
        assert!(err
            .message()
            .contains("sawtooth: input must be real; complex values are not supported"));
    }

    #[test]
    fn sawtooth_text_input_errors() {
        let err = call(Value::String("0".into())).expect_err("text sawtooth should error");
        assert!(err.message().contains("sawtooth: expected numeric input"));
    }

    #[test]
    fn sawtooth_preserves_matrix_shape() {
        let tensor = Tensor::new(vec![0.0, PI / 2.0, PI, 3.0 * PI / 2.0], vec![2, 2]).unwrap();
        let result = expect_tensor(call(Value::Tensor(tensor)).unwrap());
        assert_eq!(result.shape, vec![2, 2]);
        let expected = [-1.0, -0.5, 0.0, 0.5];
        for (got, want) in result.data.iter().zip(expected) {
            assert_close(*got, want);
        }
    }
}
