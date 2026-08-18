//! MATLAB-compatible `square` builtin for RunMat.
//!
//! `y = square(t)` evaluates a square wave with period `2*pi` at the sample
//! times in `t`, taking the value `+1` over the first half of each period and
//! `-1` over the second half. The optional second argument `duty ∈ [0, 100]`
//! is the duty cycle expressed as a percentage: the output is `+1` over the
//! first `duty/100 * 2*pi` of every period and `-1` over the remainder.

use std::f64::consts::PI;

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    NumericDType, NumericStorage, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::tensor::{scalar_f64_from_value_async, tensor_into_value};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "square";
const TWO_PI: f64 = 2.0 * PI;

const SQUARE_NONFLOATING_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "square-nonfloating-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "square with an integer or logical argument is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:SquareNonfloatingInputExtension"),
};

const SQUARE_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "square-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "interactive square gpuArray input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:SquareGpuInputExtension"),
};

pub const SQUARE_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    SQUARE_NONFLOATING_INPUT_EXTENSION,
    SQUARE_GPU_INPUT_EXTENSION,
];

const SQUARE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Square-wave output sampled at t.",
}];

const SQUARE_SIG_DEFAULT_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "t",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sample times.",
}];

const SQUARE_SIG_DUTY_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "t",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample times.",
    },
    BuiltinParamDescriptor {
        name: "duty",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("50"),
        description: "Duty cycle percentage in [0, 100].",
    },
];

const SQUARE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = square(t)",
        inputs: &SQUARE_SIG_DEFAULT_INPUTS,
        outputs: &SQUARE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = square(t, duty)",
        inputs: &SQUARE_SIG_DUTY_INPUTS,
        outputs: &SQUARE_OUTPUT,
    },
];

const SQUARE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SQUARE.INVALID_INPUT",
    identifier: Some("RunMat:square:InvalidInput"),
    when: "Primary input is not numeric-real tensor/scalar compatible.",
    message: "square: expected numeric input",
};

const SQUARE_ERROR_COMPLEX_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SQUARE.COMPLEX_UNSUPPORTED",
    identifier: Some("RunMat:square:ComplexUnsupported"),
    when: "Primary input includes complex values.",
    message: "square: input must be real; complex values are not supported",
};

const SQUARE_ERROR_DUTY_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SQUARE.DUTY_INVALID",
    identifier: Some("RunMat:square:DutyInvalid"),
    when: "Duty argument is not a real numeric scalar.",
    message: "square: duty must be a real numeric scalar in [0, 100]",
};

const SQUARE_ERROR_DUTY_RANGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SQUARE.DUTY_RANGE",
    identifier: Some("RunMat:square:DutyOutOfRange"),
    when: "Duty argument lies outside [0, 100] or is non-finite.",
    message: "square: duty must be a finite scalar in [0, 100]",
};

const SQUARE_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SQUARE.ARG_COUNT",
    identifier: Some("RunMat:square:ArgCount"),
    when: "More than one optional argument is provided.",
    message: "square: expected 1 or 2 arguments",
};

const SQUARE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SQUARE.INTERNAL",
    identifier: Some("RunMat:square:InternalError"),
    when: "Internal tensor construction or GPU gather fails.",
    message: "square: internal error",
};

const SQUARE_ERRORS: [BuiltinErrorDescriptor; 6] = [
    SQUARE_ERROR_INVALID_INPUT,
    SQUARE_ERROR_COMPLEX_UNSUPPORTED,
    SQUARE_ERROR_DUTY_INVALID,
    SQUARE_ERROR_DUTY_RANGE,
    SQUARE_ERROR_ARG_COUNT,
    SQUARE_ERROR_INTERNAL,
];

pub const SQUARE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SQUARE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SQUARE_ERRORS,
};

const SQUARE_INTEGER_T_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability { name: "t", classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "The compatibility target documents single and double sample times. RunMat mode admits typed integers only when every value is exactly representable at the binary64 waveform boundary." }];
const SQUARE_INTEGER_DUTY_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability { name: "duty", classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "The compatibility target documents single and double duty cycles. RunMat mode admits a typed integer scalar from 0 through 100 behind the nonfloating-input gate." }];
pub const SQUARE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "Y = square(integer_t)", inputs: &SQUARE_INTEGER_T_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Integer samples cross one checked binary64 phase boundary and produce a same-shaped double waveform. Automatic residency may gather transparently; explicit gpuArray input has its own extension gate." },
    BuiltinIntegerCapabilityDescriptor { form: "Y = square(t, integer_duty)", inputs: &SQUARE_INTEGER_DUTY_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "The scalar duty cycle controls the floating waveform threshold; output class and shape follow the supported t path." },
];

fn square_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    square_error_with_message(error.message, error)
}

fn square_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    square_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn square_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn square_error_with_source(
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

/// Element-wise scalar square wave.
///
/// Returns `+1` while the reduced phase is strictly less than
/// `duty/100 * 2π` and `-1` for the remainder of the period. `duty = 0`
/// therefore degenerates to a constant `-1` and `duty = 100` to a constant
/// `+1`, matching MATLAB's documented half-open interval convention.
#[inline]
fn square_scalar(t: f64, duty: f64) -> f64 {
    if !t.is_finite() {
        return f64::NAN;
    }
    let phi = t.rem_euclid(TWO_PI);
    let threshold = duty * TWO_PI / 100.0;
    if phi < threshold {
        1.0
    } else {
        -1.0
    }
}

#[runtime_builtin(
    name = "square",
    category = "math/signal",
    summary = "Generate periodic square waves with an optional duty-cycle argument.",
    keywords = "square,waveform,signal processing,duty cycle,periodic",
    type_resolver(numeric_unary_type),
    descriptor(crate::builtins::math::signal::square::SQUARE_DESCRIPTOR),
    extensions(crate::builtins::math::signal::square::SQUARE_EXTENSIONS),
    integer_capabilities(crate::builtins::math::signal::square::SQUARE_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::signal::square"
)]
async fn square_builtin(t: Value, varargin: Vec<Value>) -> BuiltinResult<Value> {
    if crate::builtins::common::validation::value_contains_explicit_gpu(&t)
        || varargin
            .iter()
            .any(crate::builtins::common::validation::value_contains_explicit_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SQUARE_GPU_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if varargin.len() <= 1
        && (is_supported_nonfloating_argument(&t)
            || varargin.iter().any(is_supported_nonfloating_argument))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SQUARE_NONFLOATING_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        &t,
        &SQUARE_NONFLOATING_INPUT_EXTENSION,
        BUILTIN_NAME,
        "sample-time",
    )
    .await?;
    if let Some(duty) = varargin.first() {
        crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
            duty,
            &SQUARE_NONFLOATING_INPUT_EXTENSION,
            BUILTIN_NAME,
            "duty-cycle",
        )
        .await?;
    }
    let duty = parse_duty(&varargin).await?;
    match t {
        Value::GpuTensor(handle) => square_gpu(handle, duty).await,
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err(square_error(&SQUARE_ERROR_COMPLEX_UNSUPPORTED))
        }
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
            Err(square_error(&SQUARE_ERROR_INVALID_INPUT))
        }
        other => square_real(other, duty),
    }
}

fn is_supported_nonfloating_argument(value: &Value) -> bool {
    match value {
        Value::Int(_) | Value::Bool(_) | Value::LogicalArray(_) => true,
        Value::Tensor(tensor) => !matches!(
            tensor.numeric_dtype(),
            NumericDType::F64 | NumericDType::F32
        ),
        Value::GpuTensor(handle) => {
            runmat_accelerate_api::handle_is_logical(handle)
                || runmat_accelerate_api::handle_integer_type(handle).is_some()
        }
        _ => false,
    }
}

async fn parse_duty(varargin: &[Value]) -> BuiltinResult<f64> {
    match varargin.len() {
        0 => Ok(50.0),
        1 => {
            let raw = scalar_f64_from_value_async(&varargin[0])
                .await
                .map_err(|err| square_error_with_detail(&SQUARE_ERROR_DUTY_INVALID, err))?
                .ok_or_else(|| square_error(&SQUARE_ERROR_DUTY_INVALID))?;
            if !raw.is_finite() || !(0.0..=100.0).contains(&raw) {
                return Err(square_error_with_detail(
                    &SQUARE_ERROR_DUTY_RANGE,
                    format!("got {raw}"),
                ));
            }
            Ok(raw)
        }
        _ => Err(square_error_with_detail(
            &SQUARE_ERROR_ARG_COUNT,
            format!("got {}", varargin.len() + 1),
        )),
    }
}

async fn square_gpu(handle: GpuTensorHandle, duty: f64) -> BuiltinResult<Value> {
    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|source| {
            square_error_with_source(&SQUARE_ERROR_INTERNAL, "gpu gather failed", source)
        })?;
    square_tensor(tensor, duty).map(tensor_into_value)
}

fn square_real(value: Value, duty: f64) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value)
        .map_err(|err| square_error_with_detail(&SQUARE_ERROR_INVALID_INPUT, err))?;
    square_tensor(tensor, duty).map(tensor_into_value)
}

fn square_tensor(tensor: Tensor, duty: f64) -> BuiltinResult<Tensor> {
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|err| square_error_with_detail(&SQUARE_ERROR_INTERNAL, &err))?;
    let storage = match storage {
        NumericStorage::F32(values) => NumericStorage::F32(
            values
                .into_iter()
                .map(|value| square_scalar(f64::from(value), duty) as f32)
                .collect(),
        ),
        storage => NumericStorage::F64(
            storage
                .materialize_f64()
                .into_iter()
                .map(|value| square_scalar(value, duty))
                .collect(),
        ),
    };
    Tensor::from_numeric_storage(storage, shape)
        .map_err(|err| square_error_with_detail(&SQUARE_ERROR_INTERNAL, &err))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{builtin_function_by_name, IntValue, LogicalArray, ResolveContext, Type};

    fn call(t: Value) -> BuiltinResult<Value> {
        block_on(square_builtin(t, Vec::new()))
    }

    fn call_with_duty(t: Value, duty: Value) -> BuiltinResult<Value> {
        block_on(square_builtin(t, vec![duty]))
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

    #[test]
    fn square_type_preserves_tensor_shape() {
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
    fn square_descriptor_signatures_and_errors() {
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("square builtin");
        let descriptor = builtin.descriptor.expect("square descriptor");
        let labels: Vec<&str> = descriptor.signatures.iter().map(|sig| sig.label).collect();
        assert!(labels.contains(&"Y = square(t)"));
        assert!(labels.contains(&"Y = square(t, duty)"));
        assert!(descriptor
            .errors
            .iter()
            .any(|err| err.code == "RM.SQUARE.DUTY_RANGE"));
    }

    #[test]
    fn square_default_is_fifty_percent_duty() {
        assert_eq!(expect_num(call(Value::Num(0.0)).unwrap()), 1.0);
        assert_eq!(expect_num(call(Value::Num(PI / 2.0)).unwrap()), 1.0);
        // The boundary at phi == pi falls into the second half (half-open semantics).
        assert_eq!(expect_num(call(Value::Num(PI)).unwrap()), -1.0);
        assert_eq!(expect_num(call(Value::Num(3.0 * PI / 2.0)).unwrap()), -1.0);
        // A full period wraps back to +1.
        assert_eq!(expect_num(call(Value::Num(TWO_PI)).unwrap()), 1.0);
    }

    #[test]
    fn square_vector_only_contains_plus_or_minus_one() {
        let n: usize = 64;
        let step = TWO_PI / (n as f64 - 1.0);
        let data: Vec<f64> = (0..n).map(|i| i as f64 * step).collect();
        let tensor = Tensor::new(data.clone(), vec![1, n]).unwrap();
        let result = expect_tensor(call(Value::Tensor(tensor)).unwrap());
        assert_eq!(result.shape, vec![1, n]);
        for (idx, &value) in result.materialize_f64().iter().enumerate() {
            assert!(
                value == 1.0 || value == -1.0,
                "index {idx}: expected only +-1 values, got {value}"
            );
        }
        // Each sample must agree with the elementwise formula.
        for (&t, &y) in data.iter().zip(result.materialize_f64().iter()) {
            assert_eq!(y, square_scalar(t, 50.0));
        }
        // The wave is +1 across the first half period, then -1 across the second
        // half. The final sample lands on the period boundary (t = 2*pi) and wraps
        // back to phi = 0 -> +1.
        assert_eq!(result.materialize_f64()[0], 1.0);
        let toggle_point = data.iter().position(|&t| t >= PI).unwrap();
        assert!(result.materialize_f64()[..toggle_point]
            .iter()
            .all(|&v| v == 1.0));
        assert!(result.materialize_f64()[toggle_point..n - 1]
            .iter()
            .all(|&v| v == -1.0));
        assert_eq!(*result.materialize_f64().last().unwrap(), 1.0);
    }

    #[test]
    fn square_negative_time_wraps_into_period() {
        // -pi/2 wraps to 3*pi/2 -> second half -> -1.
        assert_eq!(expect_num(call(Value::Num(-PI / 2.0)).unwrap()), -1.0);
        // -3*pi/2 wraps to pi/2 -> first half -> +1.
        assert_eq!(expect_num(call(Value::Num(-3.0 * PI / 2.0)).unwrap()), 1.0);
    }

    #[test]
    fn square_duty_zero_is_constant_minus_one() {
        for &t in &[0.0, PI / 2.0, PI, 3.0 * PI / 2.0, TWO_PI] {
            assert_eq!(
                expect_num(call_with_duty(Value::Num(t), Value::Num(0.0)).unwrap()),
                -1.0
            );
        }
    }

    #[test]
    fn square_duty_one_hundred_is_constant_plus_one() {
        for &t in &[0.0, PI / 2.0, PI, 3.0 * PI / 2.0, TWO_PI - 1e-9] {
            assert_eq!(
                expect_num(call_with_duty(Value::Num(t), Value::Num(100.0)).unwrap()),
                1.0
            );
        }
    }

    #[test]
    fn square_duty_twenty_five_only_first_quarter_is_positive() {
        let n: usize = 80;
        let step = TWO_PI / (n as f64);
        let data: Vec<f64> = (0..n).map(|i| i as f64 * step).collect();
        let tensor = Tensor::new(data.clone(), vec![1, n]).unwrap();
        let result =
            expect_tensor(call_with_duty(Value::Tensor(tensor), Value::Num(25.0)).unwrap());
        let threshold = PI / 2.0;
        for (idx, (&t, &y)) in data.iter().zip(result.materialize_f64().iter()).enumerate() {
            let expected = if t < threshold { 1.0 } else { -1.0 };
            assert_eq!(y, expected, "index {idx}, t={t}");
        }
    }

    #[test]
    fn square_duty_fifty_matches_default_call() {
        for &t in &[0.0, 0.7, PI - 1e-3, PI, PI + 1e-3, 5.5, -2.0, TWO_PI] {
            assert_eq!(
                expect_num(call(Value::Num(t)).unwrap()),
                expect_num(call_with_duty(Value::Num(t), Value::Num(50.0)).unwrap())
            );
        }
    }

    #[test]
    fn square_duty_rejects_out_of_range() {
        assert!(call_with_duty(Value::Num(0.0), Value::Num(-1.0)).is_err());
        assert!(call_with_duty(Value::Num(0.0), Value::Num(101.0)).is_err());
        assert!(call_with_duty(Value::Num(0.0), Value::Num(f64::NAN)).is_err());
        assert!(call_with_duty(Value::Num(0.0), Value::Num(f64::NEG_INFINITY)).is_err());
    }

    #[test]
    fn square_duty_rejects_non_scalar() {
        let tensor = Tensor::new(vec![25.0, 50.0], vec![1, 2]).unwrap();
        assert!(call_with_duty(Value::Num(0.0), Value::Tensor(tensor)).is_err());
    }

    #[test]
    fn square_int_and_logical_promote_to_double() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        assert_eq!(expect_num(call(Value::Int(IntValue::I32(0))).unwrap()), 1.0);
        assert_eq!(expect_num(call(Value::Bool(true)).unwrap()), 1.0);

        let logical = LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap();
        let result = expect_tensor(call(Value::LogicalArray(logical)).unwrap());
        assert_eq!(result.shape, vec![1, 2]);
        assert_eq!(result.materialize_f64(), vec![1.0, 1.0]);
    }

    #[test]
    fn square_extensions_follow_compatibility_mode() {
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error =
                call(Value::Int(IntValue::I16(0))).expect_err("MATLAB mode rejects integer input");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:SquareNonfloatingInputExtension")
            );
        }
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            assert!(call(Value::Int(IntValue::I16(0))).is_ok());
        }

        let handle = GpuTensorHandle {
            shape: vec![1, 2],
            device_id: 0,
            buffer_id: 9_300_004,
            descriptor: Default::default(),
        };
        runmat_accelerate_api::mark_handle_explicit(&handle);
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = call(Value::GpuTensor(handle))
            .expect_err("MATLAB mode rejects explicit interactive gpuArray input before gather");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:SquareGpuInputExtension")
        );
    }

    #[test]
    fn square_preserves_native_single_storage() {
        let input =
            Tensor::from_numeric_storage(NumericStorage::F32(vec![0.0, 4.0]), vec![1, 2]).unwrap();
        let result = expect_tensor(call(Value::Tensor(input)).unwrap());
        assert_eq!(result.numeric_dtype(), runmat_builtins::NumericDType::F32);
        assert_eq!(
            result.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![1.0, -1.0])
        );
    }

    #[test]
    fn square_nonfinite_inputs_return_nan() {
        assert!(expect_num(call(Value::Num(f64::NAN)).unwrap()).is_nan());
        assert!(expect_num(call(Value::Num(f64::INFINITY)).unwrap()).is_nan());
        assert!(expect_num(call(Value::Num(f64::NEG_INFINITY)).unwrap()).is_nan());
    }

    #[test]
    fn square_complex_input_errors() {
        let err = call(Value::Complex(1.0, 2.0)).expect_err("complex square should error");
        assert!(err
            .message()
            .contains("square: input must be real; complex values are not supported"));
    }

    #[test]
    fn square_text_input_errors() {
        let err = call(Value::String("0".into())).expect_err("text square should error");
        assert!(err.message().contains("square: expected numeric input"));
    }

    #[test]
    fn square_preserves_matrix_shape() {
        let tensor = Tensor::new(vec![0.0, PI / 2.0, PI, 3.0 * PI / 2.0], vec![2, 2]).unwrap();
        let result = expect_tensor(call(Value::Tensor(tensor)).unwrap());
        assert_eq!(result.shape, vec![2, 2]);
        assert_eq!(result.materialize_f64(), vec![1.0, 1.0, -1.0, -1.0]);
    }
}
