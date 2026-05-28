//! MATLAB-compatible `db` decibel conversion builtin for RunMat.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, Tensor, Value,
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
    summary = "Convert numeric values to decibels using MATLAB-compatible voltage, power, or resistance forms.",
    keywords = "db,decibel,voltage,power,resistance,complex",
    accel = "metadata",
    type_resolver(db_type),
    descriptor(crate::builtins::control::db::DB_DESCRIPTOR),
    builtin_path = "crate::builtins::control::db"
)]
async fn db_builtin(y: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(db_error_with_detail(
            &DB_ERROR_INVALID_ARGUMENT,
            "expected db(y), db(y, 'voltage'), db(y, 'power'), or db(y, R)",
        ));
    }

    let y = crate::gather_if_needed_async(&y).await?;
    let mode = match rest.into_iter().next() {
        Some(arg) => parse_mode(crate::gather_if_needed_async(&arg).await?)?,
        None => DbMode::Voltage,
    };

    let magnitudes = magnitude_tensor(y)?;
    match mode {
        DbMode::Voltage => map_magnitudes(magnitudes, |m| 20.0 * m.log10()),
        DbMode::Power => map_magnitudes(magnitudes, |m| 10.0 * m.log10()),
        DbMode::Resistance(reference) => {
            let reference = resistance_tensor(reference)?;
            db_with_resistance(&magnitudes, &reference)
        }
    }
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

fn magnitude_tensor(value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::Complex(re, im) => Tensor::new(vec![re.hypot(im)], vec![1, 1]).map_err(|e| {
            db_error_with_detail(
                &DB_ERROR_INTERNAL,
                format!("failed to build scalar magnitude tensor: {e}"),
            )
        }),
        Value::ComplexTensor(tensor) => complex_magnitudes(tensor),
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => Err(
            db_error_with_detail(&DB_ERROR_INVALID_INPUT, "expected numeric input"),
        ),
        other => {
            let mut tensor = tensor::value_into_tensor_for(BUILTIN_NAME, other)
                .map_err(|e| db_error_with_detail(&DB_ERROR_INVALID_INPUT, e))?;
            for value in &mut tensor.data {
                *value = value.abs();
            }
            Ok(tensor)
        }
    }
}

fn complex_magnitudes(tensor: ComplexTensor) -> BuiltinResult<Tensor> {
    let data = tensor
        .data
        .iter()
        .map(|&(re, im)| re.hypot(im))
        .collect::<Vec<_>>();
    Tensor::new(data, tensor.shape).map_err(|e| {
        db_error_with_detail(
            &DB_ERROR_INTERNAL,
            format!("failed to build magnitude tensor: {e}"),
        )
    })
}

fn resistance_tensor(value: Value) -> BuiltinResult<Tensor> {
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
            for &resistance in &tensor.data {
                if !resistance.is_finite() || resistance <= 0.0 {
                    return Err(db_error_with_detail(
                        &DB_ERROR_INVALID_RESISTANCE,
                        "resistance values must be finite and positive",
                    ));
                }
            }
            Ok(tensor)
        }
    }
}

fn map_magnitudes<F>(input: Tensor, op: F) -> BuiltinResult<Value>
where
    F: Fn(f64) -> f64,
{
    let data = input
        .data
        .iter()
        .map(|&value| op(value))
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, input.shape).map_err(|e| {
        db_error_with_detail(
            &DB_ERROR_INTERNAL,
            format!("failed to build output tensor: {e}"),
        )
    })?;
    Ok(tensor::tensor_into_value(tensor))
}

fn db_with_resistance(magnitudes: &Tensor, reference: &Tensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&magnitudes.shape, &reference.shape)
        .map_err(|err| db_error_with_detail(&DB_ERROR_SIZE_MISMATCH, err))?;
    if plan.is_empty() {
        let tensor = Tensor::new(Vec::new(), plan.output_shape().to_vec()).map_err(|e| {
            db_error_with_detail(
                &DB_ERROR_INTERNAL,
                format!("failed to build empty output tensor: {e}"),
            )
        })?;
        return Ok(tensor::tensor_into_value(tensor));
    }

    let mut data = vec![0.0; plan.len()];
    for (out_idx, y_idx, r_idx) in plan.iter() {
        let magnitude = magnitudes.data[y_idx];
        let resistance = reference.data[r_idx];
        data[out_idx] = 10.0 * ((magnitude * magnitude) / resistance).log10();
    }
    let tensor = Tensor::new(data, plan.output_shape().to_vec()).map_err(|e| {
        db_error_with_detail(
            &DB_ERROR_INTERNAL,
            format!("failed to build output tensor: {e}"),
        )
    })?;
    Ok(tensor::tensor_into_value(tensor))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{CharArray, IntValue, LogicalArray, ResolveContext, StringArray, Type};

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
                assert_eq!(tensor.data.len(), expected.len());
                for (&actual, &expected) in tensor.data.iter().zip(expected) {
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
    fn db_complex_tensor_returns_real_tensor() {
        let tensor = ComplexTensor::new(vec![(3.0, 4.0), (0.0, -10.0)], vec![2, 1]).unwrap();
        let result = db_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("db");
        assert_tensor_close(result, &[2, 1], &[20.0 * 5.0f64.log10(), 20.0]);
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
    fn db_logical_and_integer_inputs_promote_to_double() {
        let logical = LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap();
        let result = db_builtin(Value::LogicalArray(logical), Vec::new()).expect("db");
        assert_tensor_close(result, &[1, 2], &[0.0, f64::NEG_INFINITY]);

        let result = db_builtin(Value::Int(IntValue::I32(10)), Vec::new()).expect("db");
        assert_num_close(result, 20.0);
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
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = db_builtin(Value::GpuTensor(handle), Vec::new()).expect("db");
            assert_tensor_close(result, &[1, 3], &[0.0, 20.0, 40.0]);
        });
    }
}
