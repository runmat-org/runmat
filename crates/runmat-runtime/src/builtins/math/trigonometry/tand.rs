//! MATLAB-compatible `tand` builtin for RunMat.
//!
//! `tand(x)` returns the tangent of `x`, where `x` is expressed in degrees.
//! At canonical multiples of 45 degrees the result is snapped to the exact
//! rational value (`0`, `±1`), and at odd multiples of 90 degrees the
//! result is `±Inf` (MATLAB returns `Inf` for `tand(90 + 180k)` and `-Inf`
//! for `tand(-90 + 180k)`). Non-finite inputs propagate as `NaN`.

use runmat_accelerate_api::GpuTensorHandle;
#[cfg(test)]
use runmat_accelerate_api::HostTensorView;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexStorage, IntValue, NumericDType};
use runmat_value::{ComplexTensor, Tensor, Value};

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::builtins::math::trigonometry::degree_helpers::reduce_degrees;
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "tand";
const DEG_TO_RAD: f64 = std::f64::consts::PI / 180.0;

pub const TAND_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tand-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tand with typed-integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TandIntegerInputExtension"),
};
pub const TAND_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tand-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tand with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TandLogicalInputExtension"),
};
pub const TAND_CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tand-character-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tand with character input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TandCharacterInputExtension"),
};
pub const TAND_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    TAND_INTEGER_INPUT_EXTENSION,
    TAND_LOGICAL_INPUT_EXTENSION,
    TAND_CHARACTER_INPUT_EXTENSION,
];
const TAND_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "RunMat admits all eight real integer classes and reduces every value exactly modulo 360 before entering the floating degree-tangent kernel.",
}];
pub const TAND_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = tand(integer_X)",
        inputs: &TAND_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Exact integer modular reduction makes wide int64 and uint64 inputs unambiguous and preserves canonical zero, unit, and pole results; resident input gathers exactly through its owning provider.",
    }];

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
    extensions(TAND_EXTENSIONS),
    integer_capabilities(TAND_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::trigonometry::tand"
)]
async fn tand_builtin(value: Value) -> BuiltinResult<Value> {
    ensure_tand_extensions(&value)?;
    crate::builtins::common::validation::reject_typed_complex_integer(&value, "tand")?;
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

fn ensure_tand_extensions(value: &Value) -> BuiltinResult<()> {
    if crate::builtins::common::validation::value_has_native_integer_class(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TAND_INTEGER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TAND_LOGICAL_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if matches!(value, Value::CharArray(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TAND_CHARACTER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

async fn tand_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let source = handle.clone();
    let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    let host = match gathered {
        Value::Complex(re, im) => {
            let (out_re, out_im) = tand_complex(re, im);
            Ok(Value::Complex(out_re, out_im))
        }
        Value::ComplexTensor(tensor) => tand_complex_tensor(tensor),
        Value::Tensor(tensor) => tand_tensor(tensor).map(tensor::tensor_into_value),
        Value::Num(value) => Ok(Value::Num(tand_scalar(value))),
        other => Err(tand_error_with_detail(
            &TAND_ERROR_INVALID_INPUT,
            format!("unsupported gathered gpuArray value {other:?}"),
        )),
    }?;
    gpu_helpers::restore_class_preserving_value(&source, host, BUILTIN_NAME)
}

fn tand_real(value: Value) -> BuiltinResult<Value> {
    if let Value::Int(value) = value {
        return Ok(Value::Num(tand_integer_scalar(&value)));
    }
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value)
        .map_err(|e| tand_error_with_detail(&TAND_ERROR_INVALID_INPUT, e))?;
    tand_tensor(tensor).map(tensor::tensor_into_value)
}

fn tand_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    if let Some(storage) = tensor.integer_storage() {
        let data = storage
            .exact_values()
            .iter()
            .map(tand_integer_scalar)
            .collect();
        return Tensor::new(data, tensor.shape.clone())
            .map_err(|err| tand_error_with_detail(&TAND_ERROR_INTERNAL, err));
    }
    if tensor.numeric_dtype() == NumericDType::F32 {
        let data = tensor
            .as_f32_slice()
            .expect("single tensor storage")
            .iter()
            .map(|&value| tand_scalar(f64::from(value)) as f32)
            .collect();
        return Tensor::from_f32(data, tensor.shape.clone())
            .map_err(|err| tand_error_with_detail(&TAND_ERROR_INTERNAL, err));
    }
    let data = tensor::tensor_values_f64_cow(&tensor)
        .iter()
        .map(|&value| tand_scalar(value))
        .collect::<Vec<_>>();
    Tensor::new(data, tensor.shape.clone())
        .map_err(|err| tand_error_with_detail(&TAND_ERROR_INTERNAL, err))
}

fn tand_complex_tensor(tensor: ComplexTensor) -> BuiltinResult<Value> {
    let shape = tensor.shape.clone();
    let converted = match tensor.into_complex_storage() {
        ComplexStorage::F32(values) => ComplexTensor::from_f32(
            values
                .into_iter()
                .map(|(re, im)| {
                    let (out_re, out_im) = tand_complex(f64::from(re), f64::from(im));
                    (out_re as f32, out_im as f32)
                })
                .collect(),
            shape,
        ),
        ComplexStorage::F64(values) => ComplexTensor::new(
            values
                .into_iter()
                .map(|(re, im)| tand_complex(re, im))
                .collect(),
            shape,
        ),
        ComplexStorage::Integer(_) => Err("typed complex integer input is unsupported".into()),
    }
    .map_err(|err| tand_error_with_detail(&TAND_ERROR_INTERNAL, err))?;
    Ok(complex_tensor_into_value(converted))
}

fn tand_integer_scalar(value: &IntValue) -> f64 {
    let reduced = match value {
        IntValue::I8(value) => i64::from(*value) % 360,
        IntValue::I16(value) => i64::from(*value) % 360,
        IntValue::I32(value) => i64::from(*value) % 360,
        IntValue::I64(value) => *value % 360,
        IntValue::U8(value) => i64::from(*value) % 360,
        IntValue::U16(value) => i64::from(*value) % 360,
        IntValue::U32(value) => i64::from(*value) % 360,
        IntValue::U64(value) => (*value % 360) as i64,
    };
    tand_scalar(reduced as f64)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{ResolveContext, Type};
    use runmat_value::{IntValue, LogicalArray};

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
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
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
                assert_eq!(t.materialize_f64()[0], 0.0);
                assert_eq!(t.materialize_f64()[1], 1.0);
                assert_eq!(t.materialize_f64()[2], f64::INFINITY);
                assert_eq!(t.materialize_f64()[3], -1.0);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tand_reads_typed_integer_tensor_storage_exactly() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor =
            Tensor::new_integer(runmat_value::IntegerStorage::I16(vec![0, 45]), vec![1, 2])
                .expect("integer tensor");

        match tand_builtin(Value::Tensor(tensor)).expect("tand") {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                let expected = [0.0, 1.0];
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
    fn tand_logical_array_promotes() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let logical = LogicalArray::new(vec![0, 1], vec![1, 2]).unwrap();
        let result = tand_builtin(Value::LogicalArray(logical)).expect("tand");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.materialize_f64()[0], 0.0);
                let expected = (1.0_f64 * DEG_TO_RAD).tan();
                assert!((t.materialize_f64()[1] - expected).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn tand_gpu_fallback_preserves_single_and_source_owner() {
        test_support::with_f32_test_provider(|provider| {
            let input = [0.0, 45.0, 90.0];
            let source = provider
                .upload(&HostTensorView {
                    data: &input,
                    shape: &[3, 1],
                })
                .expect("upload");
            let source_device = source.device_id;
            let result =
                block_on(super::tand_builtin(Value::GpuTensor(source))).expect("tand fallback");
            let Value::GpuTensor(handle) = &result else {
                panic!("expected resident result")
            };
            assert_eq!(handle.device_id, source_device);
            let gathered = test_support::gather(result).expect("gather result");
            assert_eq!(gathered.numeric_dtype(), NumericDType::F32);
            assert_eq!(gathered.shape, vec![3, 1]);
        });
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
