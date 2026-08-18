//! MATLAB-compatible `cosd` builtin for RunMat.
//!
//! `cosd(x)` returns the cosine of `x`, where `x` is expressed in degrees.
//! At canonical multiples of 60 and 90 degrees the result is snapped to the
//! exact rational value (`0`, `±0.5`, `±1`) so users observe MATLAB's
//! noise-free outputs instead of the floating-point drift produced by
//! `cos(x*pi/180)`.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexStorage, ComplexTensor, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::trigonometry::degree_helpers::reduce_degrees;
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "cosd";
const DEG_TO_RAD: f64 = std::f64::consts::PI / 180.0;
pub const COSD_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cosd-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cosd with typed-integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CosdIntegerInputExtension"),
};
pub const COSD_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cosd-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cosd with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CosdLogicalInputExtension"),
};
pub const COSD_CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cosd-character-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cosd with character input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CosdCharacterInputExtension"),
};
pub const COSD_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    COSD_INTEGER_INPUT_EXTENSION,
    COSD_LOGICAL_INPUT_EXTENSION,
    COSD_CHARACTER_INPUT_EXTENSION,
];
const COSD_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "X", classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "All eight real integer classes require exact binary64 representability before degree reduction." }];
pub const COSD_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor { form: "Y = cosd(integer_X)", inputs: &COSD_INTEGER_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving, notes: "RunMat mode validates native integer storage before the floating degree boundary; the output is double and resident fallback is restored to the owner." }];

const COSD_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Element-wise cosine result with degree input semantics.",
}];

const COSD_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input scalar, array, logical array, complex value, or gpuArray.",
}];

const COSD_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = cosd(X)",
    inputs: &COSD_INPUTS,
    outputs: &COSD_OUTPUT,
}];

const COSD_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COSD.INVALID_INPUT",
    identifier: Some("RunMat:cosd:InvalidInput"),
    when: "Input cannot be interpreted as supported numeric/logical/complex data.",
    message: "cosd: invalid input",
};

const COSD_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COSD.INTERNAL",
    identifier: Some("RunMat:cosd:Internal"),
    when: "Internal gather/conversion/allocation flow failed.",
    message: "cosd: internal error",
};

const COSD_ERRORS: [BuiltinErrorDescriptor; 2] = [COSD_ERROR_INVALID_INPUT, COSD_ERROR_INTERNAL];

pub const COSD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COSD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &COSD_ERRORS,
};

fn cosd_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn cosd_error_with_detail(
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
    summary = "Compute cosine of degree-valued inputs.",
    keywords = "cosd,cosine,degrees,trigonometry",
    accel = "unary",
    type_resolver(numeric_unary_type),
    descriptor(crate::builtins::math::trigonometry::cosd::COSD_DESCRIPTOR),
    extensions(COSD_EXTENSIONS),
    integer_capabilities(COSD_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::trigonometry::cosd"
)]
async fn cosd_builtin(value: Value) -> BuiltinResult<Value> {
    ensure_extensions(&value)?;
    crate::builtins::common::validation::reject_typed_complex_integer(&value, "cosd")?;
    match value {
        Value::GpuTensor(handle) => cosd_gpu(handle).await,
        Value::Complex(re, im) => {
            let (out_re, out_im) = cosd_complex(re, im);
            Ok(Value::Complex(out_re, out_im))
        }
        Value::ComplexTensor(ct) => cosd_complex_tensor(ct),
        Value::String(_) | Value::StringArray(_) => Err(cosd_error(&COSD_ERROR_INVALID_INPUT)),
        other => cosd_real(other),
    }
}

fn ensure_extensions(value: &Value) -> BuiltinResult<()> {
    if is_integer(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &COSD_INTEGER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(h) if runmat_accelerate_api::handle_is_logical(h))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &COSD_LOGICAL_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if matches!(value, Value::CharArray(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &COSD_CHARACTER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    ensure_exact(value)
}
fn is_integer(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(t) if t.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(h) if runmat_accelerate_api::handle_integer_type(h).is_some())
}
fn ensure_exact(value: &Value) -> BuiltinResult<()> {
    let ok = super::cos::integer_is_exact_f64;
    let valid = match value {
        Value::Int(v) => ok(v),
        Value::Tensor(t) => t
            .integer_storage()
            .is_none_or(|s| s.exact_values().iter().all(ok)),
        _ => true,
    };
    if valid {
        Ok(())
    } else {
        Err(cosd_error_with_detail(
            &COSD_ERROR_INVALID_INPUT,
            "integer input must be exactly representable as double",
        ))
    }
}

async fn cosd_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider_for_handle(&handle);
    let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle)).await?;
    ensure_exact(&gathered)?;
    let host = match gathered {
        Value::Complex(re, im) => {
            let (re, im) = cosd_complex(re, im);
            Value::Complex(re, im)
        }
        Value::ComplexTensor(tensor) => cosd_complex_tensor(tensor)?,
        other => cosd_real(other)?,
    };
    if let Some(provider) = provider {
        upload_gpu_output(provider, host)
    } else {
        Ok(host)
    }
}

fn cosd_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value)
        .map_err(|e| cosd_error_with_detail(&COSD_ERROR_INVALID_INPUT, e))?;
    cosd_tensor(tensor).map(tensor::tensor_into_value)
}

fn cosd_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    if tensor.numeric_dtype() == runmat_builtins::NumericDType::F32 {
        let data = tensor
            .as_f32_slice()
            .expect("single tensor storage")
            .iter()
            .map(|&v| cosd_scalar(f64::from(v)) as f32)
            .collect();
        return Tensor::from_f32(data, tensor.shape.clone())
            .map_err(|e| cosd_error_with_detail(&COSD_ERROR_INTERNAL, e));
    }
    let data = tensor::tensor_values_f64_cow(&tensor)
        .iter()
        .map(|&value| cosd_scalar(value))
        .collect::<Vec<_>>();
    Tensor::new(data, tensor.shape.clone())
        .map_err(|err| cosd_error_with_detail(&COSD_ERROR_INTERNAL, err))
}

fn cosd_complex_tensor(tensor: ComplexTensor) -> BuiltinResult<Value> {
    let shape = tensor.shape.clone();
    let converted = match tensor.into_complex_storage() {
        ComplexStorage::F32(values) => ComplexTensor::from_f32(
            values
                .into_iter()
                .map(|(re, im)| {
                    let (re, im) = cosd_complex(f64::from(re), f64::from(im));
                    (re as f32, im as f32)
                })
                .collect(),
            shape,
        ),
        ComplexStorage::F64(values) => ComplexTensor::new(
            values
                .into_iter()
                .map(|(re, im)| cosd_complex(re, im))
                .collect(),
            shape,
        ),
        ComplexStorage::Integer(_) => Err("typed complex integer input is unsupported".into()),
    }
    .map_err(|err| cosd_error_with_detail(&COSD_ERROR_INTERNAL, err))?;
    Ok(complex_tensor_into_value(converted))
}

fn upload_gpu_output(
    provider: &dyn runmat_accelerate_api::AccelProvider,
    value: Value,
) -> BuiltinResult<Value> {
    match value {
        Value::Num(value) => upload_real_gpu_output(
            provider,
            Tensor::new(vec![value], vec![1, 1])
                .map_err(|e| cosd_error_with_detail(&COSD_ERROR_INTERNAL, e))?,
        ),
        Value::Tensor(tensor) => upload_real_gpu_output(provider, tensor),
        Value::Complex(re, im) => upload_complex_gpu_output(
            provider,
            ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| cosd_error_with_detail(&COSD_ERROR_INTERNAL, e))?,
        ),
        Value::ComplexTensor(tensor) => upload_complex_gpu_output(provider, tensor),
        other => Err(cosd_error_with_detail(
            &COSD_ERROR_INTERNAL,
            format!("cannot restore GPU output {other:?}"),
        )),
    }
}

fn upload_real_gpu_output(
    provider: &dyn runmat_accelerate_api::AccelProvider,
    tensor: Tensor,
) -> BuiltinResult<Value> {
    let handle = gpu_helpers::upload_tensor(provider, &tensor)
        .map_err(|e| cosd_error_with_detail(&COSD_ERROR_INTERNAL, e))?;
    Ok(gpu_helpers::resident_gpu_value(handle))
}

fn upload_complex_gpu_output(
    provider: &dyn runmat_accelerate_api::AccelProvider,
    tensor: ComplexTensor,
) -> BuiltinResult<Value> {
    let handle = gpu_helpers::upload_complex_tensor(provider, &tensor)?;
    Ok(gpu_helpers::complex_gpu_value(handle))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, LogicalArray, NumericDType, ResolveContext, Type};

    fn cosd_builtin(value: Value) -> BuiltinResult<Value> {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        block_on(super::cosd_builtin(value))
    }

    fn error_message(err: &RuntimeError) -> String {
        err.message().to_string()
    }

    #[test]
    fn cosd_descriptor_signatures_cover_core_form() {
        let labels: Vec<&str> = COSD_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = cosd(X)"));
        assert_eq!(COSD_INTEGER_CAPABILITIES[0].inputs[0].classes.len(), 8);
    }

    #[test]
    fn cosd_integer_gate_boundary_and_single_precision() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        assert!(block_on(super::cosd_builtin(Value::Int(IntValue::I8(0)))).is_err());
        drop(_strict);
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
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
            assert!(block_on(super::cosd_builtin(Value::Int(value))).is_ok());
        }
        assert!(block_on(super::cosd_builtin(Value::Int(IntValue::U64(
            (1_u64 << 53) + 1
        ))))
        .is_err());
        assert!(block_on(super::cosd_builtin(Value::Int(IntValue::U64(1_u64 << 54)))).is_ok());
        let Value::Tensor(real) = block_on(super::cosd_builtin(Value::Tensor(
            Tensor::from_f32(vec![0.0, 60.0], vec![2, 1]).unwrap(),
        )))
        .unwrap() else {
            panic!("expected single tensor")
        };
        assert_eq!(real.numeric_dtype(), NumericDType::F32);
        let Value::ComplexTensor(complex) = block_on(super::cosd_builtin(Value::ComplexTensor(
            ComplexTensor::from_f32(vec![(30.0, 1.0)], vec![1, 1]).unwrap(),
        )))
        .unwrap() else {
            panic!("expected complex tensor")
        };
        assert_eq!(complex.numeric_dtype(), NumericDType::F32);
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
                assert_eq!(t.materialize_f64(), vec![1.0, 0.0, -1.0, 0.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cosd_reads_typed_integer_tensor_storage_exactly() {
        let tensor = Tensor::new_integer(
            runmat_builtins::IntegerStorage::I16(vec![0, 60, 180]),
            vec![3, 1],
        )
        .expect("integer tensor");

        match cosd_builtin(Value::Tensor(tensor)).expect("cosd") {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                let expected = [1.0, 0.5, -1.0];
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
    fn cosd_logical_array_promotes() {
        let logical = LogicalArray::new(vec![0, 1], vec![1, 2]).unwrap();
        let result = cosd_builtin(Value::LogicalArray(logical)).expect("cosd");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.materialize_f64()[0], 1.0);
                let expected = (1.0_f64 * DEG_TO_RAD).cos();
                assert!((t.materialize_f64()[1] - expected).abs() < 1e-12);
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
        assert!(error_message(&err).contains("invalid input"));
        assert_eq!(err.identifier(), COSD_ERROR_INVALID_INPUT.identifier);
    }
}
