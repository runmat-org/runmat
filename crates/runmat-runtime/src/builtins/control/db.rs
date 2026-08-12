//! MATLAB-compatible `db` decibel conversion builtin for RunMat.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, NumericDType, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::broadcast::BroadcastPlan;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::control::type_resolvers::db_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "db";
const DB_NONFLOATING_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "db-nonfloating-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "db with integer, logical, or complex-integer computation input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DbNonfloatingInputExtension"),
};
pub const DB_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [DB_NONFLOATING_INPUT_EXTENSION];

const DB_OUTPUT_YDB: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "yDb",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Decibel-converted output.",
}];
const DB_INPUTS_Y: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input signal magnitude or power quantity.",
}];
const DB_INPUTS_Y_MODE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input signal magnitude or power quantity.",
    },
    BuiltinParamDescriptor {
        name: "modeOrR",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("\"voltage\""),
        description: "Mode string ('voltage' or 'power') or resistance reference.",
    },
];
const DB_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "yDb = db(y)",
        inputs: &DB_INPUTS_Y,
        outputs: &DB_OUTPUT_YDB,
    },
    BuiltinSignatureDescriptor {
        label: "yDb = db(y, \"voltage\")",
        inputs: &DB_INPUTS_Y_MODE,
        outputs: &DB_OUTPUT_YDB,
    },
    BuiltinSignatureDescriptor {
        label: "yDb = db(y, \"power\")",
        inputs: &DB_INPUTS_Y_MODE,
        outputs: &DB_OUTPUT_YDB,
    },
    BuiltinSignatureDescriptor {
        label: "yDb = db(y, R)",
        inputs: &DB_INPUTS_Y_MODE,
        outputs: &DB_OUTPUT_YDB,
    },
];
const DB_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DB.INVALID_ARGUMENT",
    identifier: Some("RunMat:db:InvalidArgument"),
    when: "Inputs do not match supported db invocation forms.",
    message: "db: invalid argument",
};
const DB_ERROR_INVALID_MODE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DB.INVALID_MODE",
    identifier: Some("RunMat:db:InvalidMode"),
    when: "Mode string is not recognized or is not a scalar text value.",
    message: "db: invalid mode",
};
const DB_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DB.INVALID_INPUT",
    identifier: Some("RunMat:db:InvalidInput"),
    when: "Input signal cannot be interpreted as numeric magnitude data.",
    message: "db: invalid input",
};
const DB_ERROR_INVALID_RESISTANCE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DB.INVALID_RESISTANCE",
    identifier: Some("RunMat:db:InvalidResistance"),
    when: "Resistance reference is non-numeric, complex, non-finite, or non-positive.",
    message: "db: invalid resistance",
};
const DB_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DB.SIZE_MISMATCH",
    identifier: Some("RunMat:db:SizeMismatch"),
    when: "Signal and resistance inputs are not broadcast compatible.",
    message: "db: array sizes are not compatible for broadcasting",
};
const DB_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DB.INTERNAL",
    identifier: Some("RunMat:db:Internal"),
    when: "Internal tensor conversion or allocation failed.",
    message: "db: internal error",
};
const DB_ERRORS: [BuiltinErrorDescriptor; 6] = [
    DB_ERROR_INVALID_ARGUMENT,
    DB_ERROR_INVALID_MODE,
    DB_ERROR_INVALID_INPUT,
    DB_ERROR_INVALID_RESISTANCE,
    DB_ERROR_SIZE_MISMATCH,
    DB_ERROR_INTERNAL,
];
pub const DB_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DB_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DB_ERRORS,
};

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "y",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Real and paired-complex integer computation input is gated by db-nonfloating-input and enters the decibel floating domain.",
    },
    BuiltinIntegerInputCapability {
        name: "R",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer positive resistance is gated by db-nonfloating-input and broadcasts with the signal magnitude.",
    },
];

pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "yDb = db(y, mode_or_R)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::BroadcastCompatible,
        notes: "Nonfloating computation input is RunMat-only; ordinary integer/logical input promotes to double, while a native-single resistance can select single output, and resident inputs gather for host conversion.",
    }];

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::control::db")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "db",
    op_kind: GpuOpKind::Custom("decibel-conversion"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host-side decibel conversion; gpuArray inputs are gathered before applying mode parsing, complex magnitudes, and optional resistance broadcasting.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::control::db")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "db",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "db is a compound element-wise conversion with string mode parsing and optional resistance input; it terminates fusion and executes on the host.",
};

fn db_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    db_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn db_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[derive(Clone, Debug)]
enum DbMode {
    Voltage,
    Power,
    Resistance(Value),
}

#[runtime_builtin(
    name = "db",
    category = "control",
    summary = "Convert numeric values to decibels.",
    keywords = "db,decibel,voltage,power,resistance,complex",
    accel = "metadata",
    type_resolver(db_type),
    descriptor(crate::builtins::control::db::DB_DESCRIPTOR),
    extensions(crate::builtins::control::db::DB_EXTENSIONS),
    integer_capabilities(crate::builtins::control::db::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::control::db"
)]
async fn db_builtin(y: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(db_error_with_detail(
            &DB_ERROR_INVALID_ARGUMENT,
            "expected db(y), db(y, 'voltage'), db(y, 'power'), or db(y, R)",
        ));
    }
    let nonfloating_extension =
        is_nonfloating_extension_value(&y) || rest.iter().any(is_nonfloating_extension_value);
    if is_resident_nonfloating_extension_value(&y)
        || rest.iter().any(is_resident_nonfloating_extension_value)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DB_NONFLOATING_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }

    let y = crate::gather_if_needed_async(&y).await?;
    let mode = match rest.into_iter().next() {
        Some(arg) => parse_mode(crate::gather_if_needed_async(&arg).await?)?,
        None => DbMode::Voltage,
    };

    match mode {
        DbMode::Voltage => {
            let input = magnitude_input(y)?;
            ensure_nonfloating_extension_enabled(nonfloating_extension)?;
            map_real_input(input, |m| 20.0 * m.log10())
        }
        DbMode::Power => {
            let input = power_input(y)?;
            ensure_nonfloating_extension_enabled(nonfloating_extension)?;
            map_real_input(input, |power| 10.0 * power.log10())
        }
        DbMode::Resistance(reference) => {
            let magnitudes = magnitude_input(y)?;
            let reference = resistance_input(reference)?;
            ensure_nonfloating_extension_enabled(nonfloating_extension)?;
            db_with_resistance(&magnitudes, &reference)
        }
    }
}

fn ensure_nonfloating_extension_enabled(enabled: bool) -> BuiltinResult<()> {
    if enabled {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DB_NONFLOATING_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

fn is_nonfloating_extension_value(value: &Value) -> bool {
    matches!(
        value,
        Value::Int(_) | Value::Bool(_) | Value::LogicalArray(_)
    ) || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::ComplexTensor(tensor) if tensor.integer_storage().is_some())
        || is_resident_nonfloating_extension_value(value)
}

fn is_resident_nonfloating_extension_value(value: &Value) -> bool {
    matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some() || runmat_accelerate_api::handle_is_logical(handle))
}

fn parse_mode(value: Value) -> BuiltinResult<DbMode> {
    match value {
        Value::String(text) => parse_mode_string(&text),
        Value::StringArray(array) if array.data.len() == 1 => parse_mode_string(&array.data[0]),
        Value::StringArray(_) => Err(db_error_with_detail(
            &DB_ERROR_INVALID_MODE,
            "mode must be a scalar string",
        )),
        Value::CharArray(array) if array.rows == 1 => {
            let text = array.data.iter().collect::<String>();
            parse_mode_string(&text)
        }
        Value::CharArray(_) => Err(db_error_with_detail(
            &DB_ERROR_INVALID_MODE,
            "mode must be a character row vector",
        )),
        other => Ok(DbMode::Resistance(other)),
    }
}

fn parse_mode_string(text: &str) -> BuiltinResult<DbMode> {
    match text.to_ascii_lowercase().as_str() {
        "voltage" => Ok(DbMode::Voltage),
        "power" => Ok(DbMode::Power),
        _ => Err(db_error_with_detail(
            &DB_ERROR_INVALID_MODE,
            format!("unknown mode '{text}', expected 'voltage' or 'power'"),
        )),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputPrecision {
    Double,
    Single,
}

#[derive(Clone, Debug)]
struct RealInput {
    values: Vec<f64>,
    shape: Vec<usize>,
    precision: OutputPrecision,
}

fn tensor_precision(tensor: &Tensor) -> OutputPrecision {
    if tensor.numeric_dtype() == NumericDType::F32 {
        OutputPrecision::Single
    } else {
        OutputPrecision::Double
    }
}

fn magnitude_input(value: Value) -> BuiltinResult<RealInput> {
    match value {
        Value::Complex(re, im) => Ok(RealInput {
            values: vec![re.hypot(im)],
            shape: vec![1, 1],
            precision: OutputPrecision::Double,
        }),
        Value::ComplexTensor(tensor) => Ok(complex_magnitudes(tensor)),
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => Err(
            db_error_with_detail(&DB_ERROR_INVALID_INPUT, "expected numeric input"),
        ),
        other => {
            let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, other)
                .map_err(|e| db_error_with_detail(&DB_ERROR_INVALID_INPUT, e))?;
            let precision = tensor_precision(&tensor);
            let shape = tensor.shape.clone();
            let values = tensor::tensor_into_values_f64(tensor)
                .into_iter()
                .map(f64::abs)
                .collect::<Vec<_>>();
            Ok(RealInput {
                values,
                shape,
                precision,
            })
        }
    }
}

fn complex_magnitudes(tensor: ComplexTensor) -> RealInput {
    let shape = tensor.shape.clone();
    let precision = if tensor.numeric_dtype() == NumericDType::F32 {
        OutputPrecision::Single
    } else {
        OutputPrecision::Double
    };
    let values = tensor::complex_tensor_into_values_complex64(tensor)
        .into_iter()
        .map(|value| value.norm())
        .collect::<Vec<_>>();
    RealInput {
        values,
        shape,
        precision,
    }
}

fn power_input(value: Value) -> BuiltinResult<RealInput> {
    match value {
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(db_error_with_detail(
            &DB_ERROR_INVALID_INPUT,
            "power measurements must be real and nonnegative",
        )),
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => Err(
            db_error_with_detail(&DB_ERROR_INVALID_INPUT, "expected numeric input"),
        ),
        other => {
            let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, other)
                .map_err(|e| db_error_with_detail(&DB_ERROR_INVALID_INPUT, e))?;
            let precision = tensor_precision(&tensor);
            let shape = tensor.shape.clone();
            let values = tensor::tensor_into_values_f64(tensor);
            if values.iter().any(|value| *value < 0.0) {
                return Err(db_error_with_detail(
                    &DB_ERROR_INVALID_INPUT,
                    "power measurements must be nonnegative",
                ));
            }
            Ok(RealInput {
                values,
                shape,
                precision,
            })
        }
    }
}

fn resistance_input(value: Value) -> BuiltinResult<RealInput> {
    match value {
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(db_error_with_detail(
            &DB_ERROR_INVALID_RESISTANCE,
            "resistance must be real",
        )),
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => Err(
            db_error_with_detail(&DB_ERROR_INVALID_RESISTANCE, "resistance must be numeric"),
        ),
        other => {
            let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, other)
                .map_err(|e| db_error_with_detail(&DB_ERROR_INVALID_RESISTANCE, e))?;
            let precision = tensor_precision(&tensor);
            let shape = tensor.shape.clone();
            let values = tensor::tensor_into_values_f64(tensor);
            for &resistance in &values {
                if !resistance.is_finite() || resistance <= 0.0 {
                    return Err(db_error_with_detail(
                        &DB_ERROR_INVALID_RESISTANCE,
                        "resistance values must be finite and positive",
                    ));
                }
            }
            Ok(RealInput {
                values,
                shape,
                precision,
            })
        }
    }
}

fn build_output(
    values: Vec<f64>,
    shape: Vec<usize>,
    precision: OutputPrecision,
) -> BuiltinResult<Value> {
    let tensor = match precision {
        OutputPrecision::Double => Tensor::new(values, shape),
        OutputPrecision::Single => Tensor::from_f32(
            values.into_iter().map(|value| value as f32).collect(),
            shape,
        ),
    }
    .map_err(|e| {
        db_error_with_detail(
            &DB_ERROR_INTERNAL,
            format!("failed to build output tensor: {e}"),
        )
    })?;
    match precision {
        OutputPrecision::Double => Ok(tensor::tensor_into_value(tensor)),
        OutputPrecision::Single => Ok(Value::Tensor(tensor)),
    }
}

fn map_real_input<F>(input: RealInput, op: F) -> BuiltinResult<Value>
where
    F: Fn(f64) -> f64,
{
    let data = input.values.into_iter().map(op).collect::<Vec<_>>();
    build_output(data, input.shape, input.precision)
}

fn db_with_resistance(magnitudes: &RealInput, reference: &RealInput) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&magnitudes.shape, &reference.shape)
        .map_err(|err| db_error_with_detail(&DB_ERROR_SIZE_MISMATCH, err))?;
    let precision = if matches!(magnitudes.precision, OutputPrecision::Single)
        || matches!(reference.precision, OutputPrecision::Single)
    {
        OutputPrecision::Single
    } else {
        OutputPrecision::Double
    };
    if plan.is_empty() {
        return build_output(Vec::new(), plan.output_shape().to_vec(), precision);
    }

    let mut data = vec![0.0; plan.len()];
    for (out_idx, y_idx, r_idx) in plan.iter() {
        let magnitude = magnitudes.values[y_idx];
        let resistance = reference.values[r_idx];
        data[out_idx] = 10.0 * ((magnitude * magnitude) / resistance).log10();
    }
    build_output(data, plan.output_shape().to_vec(), precision)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{
        CharArray, IntValue, IntegerComplexStorage, IntegerStorage, LogicalArray, ResolveContext,
        StringArray, Type,
    };

    fn db_builtin(y: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::db_builtin(y, rest))
    }

    fn assert_num_close(value: Value, expected: f64) {
        match value {
            Value::Num(actual) => assert!(
                (actual - expected).abs() < 1e-12,
                "expected {expected}, got {actual}"
            ),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    fn assert_tensor_close(value: Value, expected_shape: &[usize], expected: &[f64]) {
        match value {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, expected_shape);
                assert_eq!(tensor.materialize_f64().len(), expected.len());
                for (&actual, &expected) in tensor.materialize_f64().iter().zip(expected) {
                    if expected.is_infinite() {
                        assert_eq!(actual, expected);
                    } else {
                        assert!(
                            (actual - expected).abs() < 1e-12,
                            "expected {expected}, got {actual}"
                        );
                    }
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    fn integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Tensor {
        Tensor::new_integer(storage, shape).expect("integer tensor")
    }

    fn complex_integer_tensor(
        real: IntegerStorage,
        imag: IntegerStorage,
        shape: Vec<usize>,
    ) -> ComplexTensor {
        let storage = IntegerComplexStorage::new(real, imag).expect("complex integer storage");
        ComplexTensor::new_integer(storage, shape).expect("complex integer tensor")
    }

    #[test]
    fn db_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = DB_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"yDb = db(y)"));
        assert!(labels.contains(&"yDb = db(y, \"voltage\")"));
        assert!(labels.contains(&"yDb = db(y, \"power\")"));
        assert!(labels.contains(&"yDb = db(y, R)"));
    }

    #[test]
    fn db_type_unary_preserves_tensor_shape() {
        let out = db_type(
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
    fn db_type_scalar_returns_num() {
        let out = db_type(&[Type::Num], &ResolveContext::new(Vec::new()));
        assert_eq!(out, Type::Num);
    }

    #[test]
    fn db_type_string_mode_uses_input_shape() {
        let out = db_type(
            &[
                Type::Tensor {
                    shape: Some(vec![Some(4), Some(1)]),
                },
                Type::String,
            ],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(4), Some(1)])
            }
        );
    }

    #[test]
    fn db_type_text_modes_use_unary_shape_rules() {
        let string_array_type = Type::from_value(&Value::StringArray(
            StringArray::new(vec!["power".into()], vec![1, 1]).unwrap(),
        ));
        let char_array_type = Type::from_value(&Value::CharArray(CharArray::new_row("power")));

        for mode in [Type::String, string_array_type, char_array_type] {
            let out = db_type(
                &[
                    Type::Tensor {
                        shape: Some(vec![Some(1), Some(1)]),
                    },
                    mode,
                ],
                &ResolveContext::new(Vec::new()),
            );
            assert_eq!(out, Type::Num);
        }
    }

    #[test]
    fn db_type_resistance_broadcasts_shapes() {
        let out = db_type(
            &[
                Type::Tensor {
                    shape: Some(vec![Some(2), Some(1)]),
                },
                Type::Tensor {
                    shape: Some(vec![Some(1), Some(3)]),
                },
            ],
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
    fn db_default_voltage_scalar() {
        assert_num_close(db_builtin(Value::Num(10.0), Vec::new()).expect("db"), 20.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn db_voltage_mode_matches_default() {
        let result = db_builtin(
            Value::Num(10.0),
            vec![Value::CharArray(CharArray::new_row("voltage"))],
        )
        .expect("db");
        assert_num_close(result, 20.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn db_power_mode_scalar() {
        let result = db_builtin(
            Value::Num(100.0),
            vec![Value::CharArray(CharArray::new_row("power"))],
        )
        .expect("db");
        assert_num_close(result, 20.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn db_power_mode_rejects_negative_and_complex_values() {
        for value in [Value::Num(-1.0), Value::Complex(1.0, 1.0)] {
            let err = db_builtin(value, vec![Value::CharArray(CharArray::new_row("power"))])
                .expect_err("invalid power input");
            assert!(err.message().contains("nonnegative"));
            assert_eq!(err.identifier(), DB_ERROR_INVALID_INPUT.identifier);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn db_negative_input_uses_magnitude() {
        assert_num_close(db_builtin(Value::Num(-10.0), Vec::new()).expect("db"), 20.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn db_zero_input_returns_negative_infinity() {
        match db_builtin(Value::Num(0.0), Vec::new()).expect("db") {
            Value::Num(value) => assert_eq!(value, f64::NEG_INFINITY),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn db_complex_scalar_uses_magnitude() {
        assert_num_close(
            db_builtin(Value::Complex(3.0, 4.0), Vec::new()).expect("db"),
            20.0 * 5.0f64.log10(),
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn db_tensor_elements() {
        let tensor = Tensor::new(vec![1.0, 10.0, 100.0], vec![1, 3]).unwrap();
        let result = db_builtin(Value::Tensor(tensor), Vec::new()).expect("db");
        assert_tensor_close(result, &[1, 3], &[0.0, 20.0, 40.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn db_preserves_native_single_output() {
        let tensor = Tensor::from_f32(vec![1.0, 10.0, 100.0], vec![1, 3]).unwrap();
        let result = db_builtin(Value::Tensor(tensor), Vec::new()).expect("db");
        let Value::Tensor(tensor) = result else {
            panic!("expected native-single tensor");
        };
        assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
        assert_eq!(tensor.materialize_f64(), vec![0.0, 20.0, 40.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn db_complex_tensor_returns_real_tensor() {
        let tensor = ComplexTensor::new(vec![(3.0, 4.0), (0.0, -10.0)], vec![2, 1]).unwrap();
        let result = db_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("db");
        assert_tensor_close(result, &[2, 1], &[20.0 * 5.0f64.log10(), 20.0]);
    }

    #[test]
    fn db_complex_single_returns_real_single() {
        let tensor = ComplexTensor::from_f32(vec![(3.0, 4.0), (0.0, -10.0)], vec![2, 1])
            .expect("complex single input");
        let result = db_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("db");
        let Value::Tensor(tensor) = result else {
            panic!("expected real single tensor");
        };
        assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
        assert_eq!(tensor.shape, vec![2, 1]);
        assert_eq!(
            tensor.materialize_f64(),
            vec![f64::from((20.0_f64 * 5.0_f64.log10()) as f32), 20.0]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn db_resistance_scalar() {
        let result = db_builtin(Value::Num(10.0), vec![Value::Num(50.0)]).expect("db");
        assert_num_close(result, 10.0 * (2.0f64).log10());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn db_resistance_broadcasts() {
        let y = Tensor::new(vec![10.0, 20.0], vec![2, 1]).unwrap();
        let r = Tensor::new(vec![50.0, 100.0, 200.0], vec![1, 3]).unwrap();
        let result = db_builtin(Value::Tensor(y), vec![Value::Tensor(r)]).expect("db");
        assert_tensor_close(
            result,
            &[2, 3],
            &[
                10.0 * (100.0f64 / 50.0).log10(),
                10.0 * (400.0f64 / 50.0).log10(),
                10.0 * (100.0f64 / 100.0).log10(),
                10.0 * (400.0f64 / 100.0).log10(),
                10.0 * (100.0f64 / 200.0).log10(),
                10.0 * (400.0f64 / 200.0).log10(),
            ],
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn db_single_resistance_selects_single_output() {
        let resistance = Tensor::from_f32(vec![50.0], vec![1, 1]).unwrap();
        let result =
            db_builtin(Value::Num(10.0), vec![Value::Tensor(resistance)]).expect("single db");
        let Value::Tensor(tensor) = result else {
            panic!("expected native-single tensor");
        };
        assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
        assert_eq!(
            tensor.materialize_f64(),
            vec![f64::from((10.0_f64 * 2.0_f64.log10()) as f32)]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn db_magnitude_and_resistance_read_typed_integer_storage_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let y = integer_tensor(IntegerStorage::I16(vec![-10, 20]), vec![2, 1]);
        let r = integer_tensor(IntegerStorage::U16(vec![50, 100]), vec![1, 2]);
        let result = db_builtin(Value::Tensor(y), vec![Value::Tensor(r)]).expect("db");
        assert_tensor_close(
            result,
            &[2, 2],
            &[
                10.0 * (100.0f64 / 50.0).log10(),
                10.0 * (400.0f64 / 50.0).log10(),
                10.0 * (100.0f64 / 100.0).log10(),
                10.0 * (400.0f64 / 100.0).log10(),
            ],
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn db_complex_magnitudes_read_typed_integer_storage_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = complex_integer_tensor(
            IntegerStorage::I16(vec![3, 0]),
            IntegerStorage::I16(vec![4, -10]),
            vec![2, 1],
        );
        let result = db_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("db");
        assert_tensor_close(result, &[2, 1], &[20.0 * 5.0f64.log10(), 20.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn db_logical_and_integer_inputs_promote_to_double() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let logical = LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap();
        let result = db_builtin(Value::LogicalArray(logical), Vec::new()).expect("db");
        assert_tensor_close(result, &[1, 2], &[0.0, f64::NEG_INFINITY]);

        let result = db_builtin(Value::Int(IntValue::I32(10)), Vec::new()).expect("db");
        assert_num_close(result, 20.0);
    }

    #[test]
    fn db_nonfloating_inputs_follow_compatibility_mode() {
        let integer = || {
            Value::Tensor(integer_tensor(
                IntegerStorage::I16(vec![10, 20]),
                vec![1, 2],
            ))
        };
        let logical = || {
            Value::LogicalArray(LogicalArray::new(vec![1, 0], vec![1, 2]).expect("logical input"))
        };
        let complex_integer = || {
            Value::ComplexTensor(complex_integer_tensor(
                IntegerStorage::I16(vec![3]),
                IntegerStorage::I16(vec![4]),
                vec![1, 1],
            ))
        };
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            for input in [integer(), logical(), complex_integer()] {
                let error = db_builtin(input, vec![])
                    .expect_err("MATLAB mode rejects nonfloating db input");
                assert_eq!(
                    error.identifier(),
                    Some("RunMat:compatibility:DbNonfloatingInputExtension")
                );
            }
            let error = db_builtin(Value::Num(10.0), vec![Value::Int(IntValue::U16(50))])
                .expect_err("MATLAB mode rejects typed-integer resistance");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:DbNonfloatingInputExtension")
            );

            let invalid = db_builtin(Value::Num(10.0), vec![Value::Int(IntValue::I16(0))])
                .expect_err("invalid resistance retains ordinary validation");
            assert_eq!(invalid.identifier(), DB_ERROR_INVALID_RESISTANCE.identifier);

            let resident = runmat_accelerate_api::GpuTensorHandle {
                shape: vec![1, 1],
                device_id: 0,
                buffer_id: 9_306_001,
            };
            runmat_accelerate_api::set_handle_integer_type(
                &resident,
                runmat_accelerate_api::IntegerElementType::I16,
            );
            let error = db_builtin(Value::GpuTensor(resident.clone()), vec![])
                .expect_err("MATLAB mode rejects resident integer before gather");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:DbNonfloatingInputExtension")
            );
            runmat_accelerate_api::clear_handle_integer_type(&resident);
        }
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            assert!(db_builtin(integer(), vec![]).is_ok());
            assert!(db_builtin(logical(), vec![]).is_ok());
            assert!(db_builtin(complex_integer(), vec![]).is_ok());
            assert!(db_builtin(Value::Num(10.0), vec![Value::Int(IntValue::U16(50))]).is_ok());
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn db_rejects_invalid_mode() {
        let err = db_builtin(
            Value::Num(1.0),
            vec![Value::CharArray(CharArray::new_row("energy"))],
        )
        .expect_err("invalid mode");
        assert!(err.message().contains("unknown mode"));
        assert_eq!(err.identifier(), DB_ERROR_INVALID_MODE.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn db_rejects_nonpositive_resistance() {
        let err =
            db_builtin(Value::Num(1.0), vec![Value::Num(0.0)]).expect_err("invalid resistance");
        assert!(err.message().contains("finite and positive"));
        assert_eq!(err.identifier(), DB_ERROR_INVALID_RESISTANCE.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn db_rejects_nonnumeric_input() {
        let err = db_builtin(Value::from("hello"), Vec::new()).expect_err("invalid input");
        assert!(err.message().contains("expected numeric"));
        assert_eq!(err.identifier(), DB_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn db_gpu_input_gathers_to_host() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 10.0, 100.0], vec![1, 3]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = db_builtin(Value::GpuTensor(handle), Vec::new()).expect("db");
            assert_tensor_close(result, &[1, 3], &[0.0, 20.0, 40.0]);
        });
    }
}
