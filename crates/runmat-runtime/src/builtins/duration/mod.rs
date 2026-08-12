use std::cell::Cell;
use std::collections::HashMap;

use runmat_builtins::{
    Access, BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor,
    BuiltinExtensionDescriptor, BuiltinExtensionMode, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, CharArray, ClassDef, MethodDef, ObjectInstance, PropertyDef,
    StringArray, Tensor, Value,
};

use crate::builtins::common::tensor;
use crate::{
    build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError, OBJECT_INDEX_MEMBER,
    OBJECT_INDEX_PAREN, OBJECT_SUBSASGN_METHOD, OBJECT_SUBSREF_METHOD,
};

const BUILTIN_NAME: &str = "duration";
const DURATION_CLASS: &str = "duration";
const DAYS_FIELD: &str = "__days";
const FORMAT_FIELD: &str = "Format";
pub(crate) const DEFAULT_DURATION_FORMAT: &str = "hh:mm:ss";
const SECONDS_PER_DAY: f64 = 86_400.0;

const DURATION_SHORT_COMPONENT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "duration-short-component-form",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "duration with one hour component or two hour/minute components is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DurationShortComponentFormExtension"),
};
const DURATION_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "duration-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "duration with resident numeric input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DurationGpuInputExtension"),
};
const DURATION_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    DURATION_SHORT_COMPONENT_EXTENSION,
    DURATION_GPU_INPUT_EXTENSION,
];
const DURATION_INTEGER_COMPONENT_INPUTS: [BuiltinIntegerInputCapability; 4] = [
    BuiltinIntegerInputCapability {
        name: "H",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Numeric hour arrays can be scalar-expanded; values enter the duration floating representation.",
    },
    BuiltinIntegerInputCapability {
        name: "MI",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Numeric minute arrays can be scalar-expanded; nonscalars must match the other component sizes.",
    },
    BuiltinIntegerInputCapability {
        name: "S",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Numeric second arrays can be scalar-expanded; nonscalars must match the other component sizes.",
    },
    BuiltinIntegerInputCapability {
        name: "MS",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The optional fourth component contributes milliseconds and follows the same scalar-expansion rule.",
    },
];
const DURATION_INTEGER_MATRIX_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "X must be a numeric matrix with exactly three columns ordered as hours, minutes, and seconds.",
    }];
pub const DURATION_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "D = duration(integer_H, integer_MI, integer_S, integer_MS?)",
        inputs: &DURATION_INTEGER_COMPONENT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "All numeric classes share MATLAB scalar expansion and produce a host duration object backed by binary64 day counts; resident inputs are a separately gated RunMat extension and gather before construction.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "D = duration(integer_X)",
        inputs: &DURATION_INTEGER_MATRIX_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Each row of the N-by-3 matrix creates one duration; output is an N-by-1 host duration array.",
    },
];

thread_local! {
    static DURATION_CLASS_REGISTERED: Cell<bool> = const { Cell::new(false) };
}

const DURATION_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DURATION.INVALID_ARGUMENT",
    identifier: Some("RunMat:duration:InvalidArgument"),
    when: "Arguments or option grammar do not match supported duration forms.",
    message: "duration: invalid argument",
};
const DURATION_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DURATION.INVALID_INPUT",
    identifier: Some("RunMat:duration:InvalidInput"),
    when: "Input values cannot be converted/broadcast/formatted to a valid duration result.",
    message: "duration: invalid input",
};
const DURATION_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DURATION.INTERNAL",
    identifier: Some("RunMat:duration:Internal"),
    when: "Internal duration state or indexing/evaluation failed unexpectedly.",
    message: "duration: internal operation failed",
};
const DURATION_ERRORS: [BuiltinErrorDescriptor; 3] = [
    DURATION_ERROR_INVALID_ARGUMENT,
    DURATION_ERROR_INVALID_INPUT,
    DURATION_ERROR_INTERNAL,
];

const OUT_DURATION: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "t",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Duration object result.",
}];
const OUT_ANY: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Method result.",
}];
const DURATION_ARGS_ONLY: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Duration constructor arguments.",
}];
const DURATION_FOUR_COMPONENT_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "hours",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Hour component.",
    },
    BuiltinParamDescriptor {
        name: "minutes",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Minute component.",
    },
    BuiltinParamDescriptor {
        name: "seconds",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second component.",
    },
    BuiltinParamDescriptor {
        name: "milliseconds",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Millisecond component.",
    },
];
const DURATION_BINARY_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "lhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Left duration operand.",
    },
    BuiltinParamDescriptor {
        name: "rhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right duration/datetime operand.",
    },
];
const DURATION_SUBSREF_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Duration receiver object.",
    },
    BuiltinParamDescriptor {
        name: "kind",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indexing kind token.",
    },
    BuiltinParamDescriptor {
        name: "payload",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Index/member payload.",
    },
];
const DURATION_SUBSASGN_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Duration receiver object.",
    },
    BuiltinParamDescriptor {
        name: "kind",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indexing kind token.",
    },
    BuiltinParamDescriptor {
        name: "payload",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Index/member payload.",
    },
    BuiltinParamDescriptor {
        name: "rhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Assigned value.",
    },
];

const DURATION_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "t = duration(X)",
        inputs: &[BuiltinParamDescriptor {
            name: "X",
            ty: BuiltinParamType::NumericArray,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "N-by-3 matrix of hour, minute, and second components.",
        }],
        outputs: &OUT_DURATION,
    },
    BuiltinSignatureDescriptor {
        label: "t = duration(hours, minutes)",
        inputs: &[
            BuiltinParamDescriptor {
                name: "hours",
                ty: BuiltinParamType::NumericArray,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Hour component.",
            },
            BuiltinParamDescriptor {
                name: "minutes",
                ty: BuiltinParamType::NumericArray,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Minute component.",
            },
        ],
        outputs: &OUT_DURATION,
    },
    BuiltinSignatureDescriptor {
        label: "t = duration(hours, minutes, seconds)",
        inputs: &[
            BuiltinParamDescriptor {
                name: "hours",
                ty: BuiltinParamType::NumericArray,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Hour component.",
            },
            BuiltinParamDescriptor {
                name: "minutes",
                ty: BuiltinParamType::NumericArray,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Minute component.",
            },
            BuiltinParamDescriptor {
                name: "seconds",
                ty: BuiltinParamType::NumericArray,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Second component.",
            },
        ],
        outputs: &OUT_DURATION,
    },
    BuiltinSignatureDescriptor {
        label: "t = duration(hours, minutes, seconds, milliseconds)",
        inputs: &DURATION_FOUR_COMPONENT_INPUTS,
        outputs: &OUT_DURATION,
    },
    BuiltinSignatureDescriptor {
        label: "t = duration(___, \"Format\", format)",
        inputs: &DURATION_ARGS_ONLY,
        outputs: &OUT_DURATION,
    },
    BuiltinSignatureDescriptor {
        label: "t = duration(___, Name, Value, ...)",
        inputs: &DURATION_ARGS_ONLY,
        outputs: &OUT_DURATION,
    },
];
const DURATION_SUBSREF_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = duration.subsref(obj, kind, payload)",
    inputs: &DURATION_SUBSREF_INPUTS,
    outputs: &OUT_ANY,
}];
const DURATION_SUBSASGN_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "out = duration.subsasgn(obj, kind, payload, rhs)",
        inputs: &DURATION_SUBSASGN_INPUTS,
        outputs: &OUT_ANY,
    }];
const DURATION_BINARY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = duration.op(lhs, rhs)",
    inputs: &DURATION_BINARY_INPUTS,
    outputs: &OUT_ANY,
}];

pub const DURATION_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DURATION_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DURATION_ERRORS,
};
pub const DURATION_SUBSREF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DURATION_SUBSREF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &DURATION_ERRORS,
};
pub const DURATION_SUBSASGN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DURATION_SUBSASGN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &DURATION_ERRORS,
};
pub const DURATION_BINARY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DURATION_BINARY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &DURATION_ERRORS,
};

fn duration_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn ensure_duration_class_registered() {
    DURATION_CLASS_REGISTERED.with(|registered| {
        if registered.get() {
            return;
        }
        let mut properties = HashMap::new();
        properties.insert(
            FORMAT_FIELD.to_string(),
            PropertyDef {
                name: FORMAT_FIELD.to_string(),
                is_static: false,
                is_constant: false,
                is_dependent: false,
                get_access: Access::Public,
                set_access: Access::Public,
                default_value: Some(Value::String(DEFAULT_DURATION_FORMAT.to_string())),
            },
        );

        let mut methods = HashMap::new();
        for name in [
            OBJECT_SUBSREF_METHOD,
            OBJECT_SUBSASGN_METHOD,
            "plus",
            "minus",
            "eq",
            "ne",
            "lt",
            "le",
            "gt",
            "ge",
        ] {
            methods.insert(
                name.to_string(),
                MethodDef {
                    name: name.to_string(),
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: Access::Public,
                    function_name: format!("{DURATION_CLASS}.{name}"),
                    implicit_class_argument: None,
                },
            );
        }

        runmat_builtins::register_class(ClassDef {
            name: DURATION_CLASS.to_string(),
            parent: None,
            properties,
            methods,
        });
        registered.set(true);
    });
}

pub fn is_duration_object(value: &Value) -> bool {
    matches!(value, Value::Object(obj) if obj.is_class(DURATION_CLASS))
}

async fn gather_args(args: &[Value]) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(args.len());
    for arg in args {
        out.push(
            gather_if_needed_async(arg)
                .await
                .map_err(|err| duration_error(format!("duration: {}", err.message())))?,
        );
    }
    Ok(out)
}

fn scalar_text(value: &Value, context: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        Value::CharArray(array) if array.rows == 1 => Ok(array.data.iter().collect()),
        _ => Err(duration_error(format!(
            "duration: {context} must be a string scalar or character vector"
        ))),
    }
}

fn parse_trailing_format(args: &[Value]) -> BuiltinResult<(usize, Option<String>)> {
    let mut positional_end = args.len();
    let mut format = None;

    while positional_end >= 2 {
        let name = match scalar_text(&args[positional_end - 2], "option name") {
            Ok(text) => text,
            Err(_) => break,
        };
        if !name.trim().eq_ignore_ascii_case("format") {
            break;
        }
        format = Some(scalar_text(&args[positional_end - 1], "Format option")?);
        positional_end -= 2;
    }

    Ok((positional_end, format))
}

fn tensor_from_numeric(value: Value, context: &str) -> BuiltinResult<Tensor> {
    tensor::value_into_tensor_for(context, value)
        .map_err(|message| duration_error(format!("duration: {message}")))
}

fn component_tensor(value: Value, context: &str) -> BuiltinResult<Tensor> {
    let tensor = tensor_from_numeric(value, context)?;
    let shape = tensor::default_shape_for(&tensor.shape, tensor.len());
    let values = tensor::tensor_into_values_f64(tensor);
    Tensor::new(values, shape).map_err(|err| duration_error(format!("duration: {err}")))
}

fn format_for_object(obj: &ObjectInstance) -> String {
    match obj.properties.get(FORMAT_FIELD) {
        Some(Value::String(text)) => text.clone(),
        Some(Value::StringArray(array)) if array.data.len() == 1 => array.data[0].clone(),
        Some(Value::CharArray(array)) if array.rows == 1 => array.data.iter().collect(),
        _ => DEFAULT_DURATION_FORMAT.to_string(),
    }
}

pub(crate) fn duration_tensor_from_duration_value(value: &Value) -> BuiltinResult<Tensor> {
    match value {
        Value::Object(obj) if obj.is_class(DURATION_CLASS) => {
            match obj.properties.get(DAYS_FIELD) {
                Some(Value::Tensor(tensor)) => Ok(tensor.clone()),
                Some(Value::Num(value)) => Tensor::new(vec![*value], vec![1, 1])
                    .map_err(|err| duration_error(format!("duration: {err}"))),
                Some(other) => Err(duration_error(format!(
                    "duration: invalid internal day storage {other:?}"
                ))),
                None => Err(duration_error("duration: missing internal day storage")),
            }
        }
        _ => Err(duration_error("duration: expected a duration value")),
    }
}

pub(crate) fn duration_format_from_value(value: &Value) -> String {
    match value {
        Value::Object(obj) if obj.is_class(DURATION_CLASS) => format_for_object(obj),
        _ => DEFAULT_DURATION_FORMAT.to_string(),
    }
}

pub(crate) fn duration_object_from_days_tensor(
    days: Tensor,
    format: impl Into<String>,
) -> BuiltinResult<Value> {
    ensure_duration_class_registered();
    let mut object = ObjectInstance::new(DURATION_CLASS.to_string());
    object
        .properties
        .insert(DAYS_FIELD.to_string(), Value::Tensor(days));
    object
        .properties
        .insert(FORMAT_FIELD.to_string(), Value::String(format.into()));
    Ok(Value::Object(object))
}

fn duration_object_from_days(
    days: Vec<f64>,
    shape: Vec<usize>,
    format: impl Into<String>,
) -> BuiltinResult<Value> {
    let tensor =
        Tensor::new(days, shape).map_err(|err| duration_error(format!("duration: {err}")))?;
    duration_object_from_days_tensor(tensor, format)
}

async fn duration_unit_value(
    value: Value,
    unit_name: &str,
    days_per_unit: f64,
) -> BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| duration_error(format!("{unit_name}: {}", err.message())))?;
    if is_duration_object(&value) {
        let days = duration_tensor_from_duration_value(&value)?;
        let day_values = tensor::tensor_values_f64_cow(&days);
        let data = day_values
            .iter()
            .map(|day| day / days_per_unit)
            .collect::<Vec<_>>();
        return if data.len() == 1 {
            Ok(Value::Num(data[0]))
        } else {
            Ok(Value::Tensor(
                Tensor::new(
                    data,
                    tensor::default_shape_for(&days.shape, day_values.len()),
                )
                .map_err(|err| duration_error(format!("{unit_name}: {err}")))?,
            ))
        };
    }
    let numeric = component_tensor(value, unit_name)?;
    let shape = tensor::default_shape_for(&numeric.shape, numeric.len());
    let values = tensor::tensor_into_values_f64(numeric);
    let days = values
        .iter()
        .map(|value| {
            if !value.is_finite() {
                Err(duration_error(format!(
                    "{unit_name}: values must be finite"
                )))
            } else {
                let days = value * days_per_unit;
                if days.is_finite() {
                    Ok(days)
                } else {
                    Err(duration_error(format!(
                        "{unit_name}: resulting duration is outside supported range"
                    )))
                }
            }
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    duration_object_from_days(days, shape, DEFAULT_DURATION_FORMAT)
}

fn broadcast_component_data(
    arrays: &[Tensor],
    labels: &[&str],
) -> BuiltinResult<(Vec<Vec<f64>>, Vec<usize>)> {
    let mut target_shape = vec![1, 1];
    let mut target_len = 1usize;

    for array in arrays {
        let len = array.len();
        if len > 1 {
            let shape = tensor::default_shape_for(&array.shape, len);
            if target_len == 1 {
                target_len = len;
                target_shape = shape;
            } else if len != target_len || shape != target_shape {
                return Err(duration_error(
                    "duration: non-scalar component inputs must have matching sizes",
                ));
            }
        }
    }

    let mut broadcasted = Vec::with_capacity(arrays.len());
    for (idx, array) in arrays.iter().enumerate() {
        let values = array
            .as_f64_slice()
            .expect("duration components are normalized to double storage");
        if values.len() == 1 {
            broadcasted.push(vec![values[0]; target_len]);
        } else if values.len() == target_len {
            broadcasted.push(values.to_vec());
        } else {
            return Err(duration_error(format!(
                "duration: {} input size does not match the other components",
                labels[idx]
            )));
        }
    }

    Ok((broadcasted, target_shape))
}

fn build_from_components(args: Vec<Value>, format: Option<String>) -> BuiltinResult<Value> {
    let labels = ["hours", "minutes", "seconds", "milliseconds"];
    let mut arrays = Vec::with_capacity(args.len());
    for (idx, arg) in args.into_iter().enumerate() {
        arrays.push(component_tensor(arg, labels[idx])?);
    }
    while arrays.len() < 4 {
        arrays.push(Tensor::new(vec![0.0], vec![1, 1]).unwrap());
    }

    let (broadcasted, shape) = broadcast_component_data(&arrays, &labels)?;
    let len = broadcasted[0].len();
    let mut days = Vec::with_capacity(len);
    for idx in 0..len {
        let total_seconds = broadcasted[0][idx] * 3600.0
            + broadcasted[1][idx] * 60.0
            + broadcasted[2][idx]
            + broadcasted[3][idx] / 1000.0;
        days.push(total_seconds / SECONDS_PER_DAY);
    }

    duration_object_from_days(
        days,
        shape,
        format.unwrap_or_else(|| DEFAULT_DURATION_FORMAT.to_string()),
    )
}

fn is_public_duration_matrix(value: &Value) -> bool {
    match value {
        Value::Tensor(tensor) => tensor.shape.len() <= 2 && tensor.cols() == 3,
        Value::GpuTensor(handle) => {
            handle.shape.len() <= 2 && handle.shape.get(1).copied().unwrap_or(1) == 3
        }
        _ => false,
    }
}

fn is_numeric_duration_input(value: &Value) -> bool {
    matches!(
        value,
        Value::Num(_) | Value::Int(_) | Value::Tensor(_) | Value::GpuTensor(_)
    )
}

fn build_from_matrix(value: Value, format: Option<String>) -> BuiltinResult<Value> {
    let matrix = component_tensor(value, "X")?;
    if matrix.shape.len() > 2 || matrix.cols() != 3 {
        return Err(duration_error(
            "duration: X must be a numeric matrix with exactly three columns",
        ));
    }
    let rows = matrix.rows();
    let values = matrix
        .as_f64_slice()
        .expect("duration matrix is normalized to double storage");
    let mut days = Vec::with_capacity(rows);
    for row in 0..rows {
        let total_seconds =
            values[row] * 3600.0 + values[row + rows] * 60.0 + values[row + 2 * rows];
        days.push(total_seconds / SECONDS_PER_DAY);
    }
    duration_object_from_days(
        days,
        vec![rows, 1],
        format.unwrap_or_else(|| DEFAULT_DURATION_FORMAT.to_string()),
    )
}

fn format_seconds_field(seconds: f64) -> String {
    let whole = seconds.floor();
    let fractional = seconds - whole;
    if fractional.abs() <= 1e-9 {
        format!("{:02}", whole as i64)
    } else {
        let mut text = format!("{:06.3}", seconds);
        while text.contains('.') && text.ends_with('0') {
            text.pop();
        }
        if text.ends_with('.') {
            text.pop();
        }
        text
    }
}

fn format_duration_value(days: f64, format: &str) -> BuiltinResult<String> {
    if days.is_nan() {
        return Ok("NaN".to_string());
    }
    if days == f64::INFINITY {
        return Ok("Inf".to_string());
    }
    if days == f64::NEG_INFINITY {
        return Ok("-Inf".to_string());
    }

    let total_seconds = days * SECONDS_PER_DAY;
    let sign = if total_seconds < 0.0 { "-" } else { "" };
    let total_seconds = total_seconds.abs();
    let total_hours = (total_seconds / 3600.0).floor();
    let total_minutes = (total_seconds / 60.0).floor();
    let hours = total_hours as i64;
    let minutes_component = ((total_seconds / 60.0).floor() as i64) % 60;
    let seconds_component =
        total_seconds - (hours as f64 * 3600.0) - (minutes_component as f64 * 60.0);

    let rendered = match format {
        "hh:mm:ss" => format!(
            "{sign}{hours:02}:{minutes_component:02}:{}",
            format_seconds_field(seconds_component)
        ),
        "hh:mm" => format!("{sign}{hours:02}:{minutes_component:02}"),
        "mm:ss" => format!(
            "{sign}{:02}:{}",
            total_minutes as i64,
            format_seconds_field(total_seconds - total_minutes * 60.0)
        ),
        "s" | "ss" => {
            let mut text = format!("{:.3}", total_seconds);
            while text.contains('.') && text.ends_with('0') {
                text.pop();
            }
            if text.ends_with('.') {
                text.pop();
            }
            format!("{sign}{text}")
        }
        other => {
            return Err(duration_error(format!(
                "duration: unsupported Format value '{other}'"
            )))
        }
    };

    Ok(rendered)
}

pub fn duration_string_array(value: &Value) -> BuiltinResult<Option<StringArray>> {
    let Value::Object(obj) = value else {
        return Ok(None);
    };
    if !obj.is_class(DURATION_CLASS) {
        return Ok(None);
    }
    let days = duration_tensor_from_duration_value(value)?;
    let format = format_for_object(obj);
    let day_values = tensor::tensor_values_f64_cow(&days);
    let mut strings = Vec::with_capacity(day_values.len());
    for value in day_values.iter() {
        strings.push(format_duration_value(*value, &format)?);
    }
    let shape = tensor::default_shape_for(&days.shape, day_values.len());
    let array = StringArray::new(strings, shape)
        .map_err(|err| duration_error(format!("duration: {err}")))?;
    Ok(Some(array))
}

pub fn duration_display_text(value: &Value) -> BuiltinResult<Option<String>> {
    let Some(array) = duration_string_array(value)? else {
        return Ok(None);
    };
    if array.data.len() == 1 {
        return Ok(Some(array.data[0].clone()));
    }

    let rows = array.rows;
    let cols = array.cols;
    let mut widths = vec![0usize; cols];
    for col in 0..cols {
        for row in 0..rows {
            let idx = row + col * rows;
            widths[col] = widths[col].max(array.data[idx].len());
        }
    }

    let mut lines = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut line = String::new();
        for col in 0..cols {
            if col > 0 {
                line.push_str("  ");
            }
            let idx = row + col * rows;
            let text = &array.data[idx];
            line.push_str(text);
            let padding = widths[col].saturating_sub(text.len());
            if padding > 0 {
                line.push_str(&" ".repeat(padding));
            }
        }
        lines.push(line);
    }

    Ok(Some(lines.join("\n")))
}

pub fn duration_summary(value: &Value) -> BuiltinResult<Option<String>> {
    let Value::Object(obj) = value else {
        return Ok(None);
    };
    if !obj.is_class(DURATION_CLASS) {
        return Ok(None);
    }
    let days = duration_tensor_from_duration_value(value)?;
    let len = days.len();
    if len == 1 {
        return duration_display_text(value);
    }
    let shape = tensor::default_shape_for(&days.shape, len);
    Ok(Some(format!(
        "[{} duration]",
        shape
            .iter()
            .map(|dim| dim.to_string())
            .collect::<Vec<_>>()
            .join("x")
    )))
}

pub fn duration_char_array(value: &Value) -> BuiltinResult<Option<CharArray>> {
    let Some(array) = duration_string_array(value)? else {
        return Ok(None);
    };
    let width = array.data.iter().map(String::len).max().unwrap_or(0);
    let rows = array.data.len();
    let mut data = vec![' '; rows * width];
    for (row, text) in array.data.iter().enumerate() {
        for (col, ch) in text.chars().enumerate() {
            data[row * width + col] = ch;
        }
    }
    let out = CharArray::new(data, rows, width)
        .map_err(|err| duration_error(format!("duration: {err}")))?;
    Ok(Some(out))
}

fn compare_duration(
    lhs: Value,
    rhs: Value,
    op: &str,
    cmp: impl Fn(f64, f64) -> bool,
) -> BuiltinResult<Value> {
    let lhs_days = duration_tensor_from_duration_value(&lhs)?;
    let rhs_days = duration_tensor_from_duration_value(&rhs)?;
    let (left, right, shape) =
        tensor::binary_numeric_tensors(&lhs_days, &rhs_days, op, BUILTIN_NAME)?;
    let out = left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| if cmp(*a, *b) { 1.0 } else { 0.0 })
        .collect::<Vec<_>>();
    if out.len() == 1 {
        Ok(Value::Num(out[0]))
    } else {
        Ok(Value::Tensor(Tensor::new(out, shape).map_err(|err| {
            duration_error(format!("duration: {err}"))
        })?))
    }
}

async fn duration_indexing(obj: Value, payload: Value) -> BuiltinResult<Value> {
    let Value::Object(object) = obj else {
        return Err(duration_error(
            "duration.subsref: receiver must be a duration object",
        ));
    };
    let format = format_for_object(&object);
    let days = duration_tensor_from_duration_value(&Value::Object(object.clone()))?;

    let Value::Cell(cell) = payload else {
        return Err(duration_error(
            "duration.subsref: indexing payload must be a cell array",
        ));
    };
    if cell.data.is_empty() {
        return duration_object_from_days_tensor(days, format);
    }
    if cell.data.len() != 1 {
        return Err(duration_error(
            "duration.subsref: only linear duration indexing is currently supported",
        ));
    }
    let selector = cell.data[0].clone();
    let selector = match selector {
        Value::Tensor(tensor) => tensor,
        Value::Num(value) => Tensor::new(vec![value], vec![1, 1])
            .map_err(|err| duration_error(format!("duration.subsref: {err}")))?,
        Value::Int(value) => Tensor::new_integer(
            runmat_builtins::IntegerStorage::from_scalar(value),
            vec![1, 1],
        )
        .map_err(|err| duration_error(format!("duration.subsref: {err}")))?,
        Value::LogicalArray(logical) => tensor::logical_to_tensor(&logical)
            .map_err(|err| duration_error(format!("duration.subsref: {err}")))?,
        other => {
            return Err(duration_error(format!(
                "duration.subsref: unsupported index value {other:?}"
            )))
        }
    };
    let indexed =
        crate::perform_indexing(&Value::Tensor(days), &tensor::tensor_values_f64(&selector))
            .await
            .map_err(|err| duration_error(format!("duration.subsref: {}", err.message())))?;
    let indexed_days = match indexed {
        Value::Num(value) => Tensor::new(vec![value], vec![1, 1])
            .map_err(|err| duration_error(format!("duration.subsref: {err}")))?,
        Value::Tensor(tensor) => tensor,
        other => {
            return Err(duration_error(format!(
                "duration.subsref: unexpected indexing result {other:?}"
            )))
        }
    };
    duration_object_from_days_tensor(indexed_days, format)
}

#[runmat_macros::runtime_builtin(
    name = "duration",
    descriptor(crate::builtins::duration::DURATION_DESCRIPTOR),
    builtin_path = "crate::builtins::duration",
    category = "datetime",
    summary = "Create duration arrays from hour, minute, and second components.",
    keywords = "duration,time span,elapsed time,Format",
    related = "datetime,string,char,disp",
    examples = "t = duration(1, 30, 45);",
    extensions(DURATION_EXTENSIONS),
    integer_capabilities(DURATION_INTEGER_CAPABILITIES)
)]
async fn duration_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    ensure_duration_class_registered();
    let (raw_positional_end, _) = parse_trailing_format(&args)?;
    let raw_positional = &args[..raw_positional_end];
    if args.iter().any(crate::value_contains_gpu) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DURATION_GPU_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let short_numeric_form = match raw_positional {
        [value] => is_numeric_duration_input(value) && !is_public_duration_matrix(value),
        [first, second] => is_numeric_duration_input(first) && is_numeric_duration_input(second),
        _ => false,
    };
    if short_numeric_form {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DURATION_SHORT_COMPONENT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let args = gather_args(&args).await?;
    let (positional_end, format) = parse_trailing_format(&args)?;
    let mut positional = args[..positional_end].to_vec();

    match positional.len() {
        1 if is_public_duration_matrix(&positional[0]) => {
            build_from_matrix(positional.remove(0), format)
        }
        1..=4 if positional.iter().all(is_numeric_duration_input) => {
            build_from_components(positional, format)
        }
        _ => Err(duration_error(
            "duration: unsupported argument pattern; use X or H/MI/S/MS numeric inputs",
        )),
    }
}

#[runmat_macros::runtime_builtin(
    name = "days",
    builtin_path = "crate::builtins::duration",
    category = "datetime",
    summary = "Create duration values from days or convert duration values to day counts.",
    keywords = "days,duration,datetime"
)]
async fn days_builtin(value: Value) -> crate::BuiltinResult<Value> {
    duration_unit_value(value, "days", 1.0).await
}

#[runmat_macros::runtime_builtin(
    name = "hours",
    builtin_path = "crate::builtins::duration",
    category = "datetime",
    summary = "Create duration values from hours or convert duration values to hour counts.",
    keywords = "hours,duration,datetime"
)]
async fn hours_builtin(value: Value) -> crate::BuiltinResult<Value> {
    duration_unit_value(value, "hours", 1.0 / 24.0).await
}

#[runmat_macros::runtime_builtin(
    name = "minutes",
    builtin_path = "crate::builtins::duration",
    category = "datetime",
    summary = "Create duration values from minutes or convert duration values to minute counts.",
    keywords = "minutes,duration,datetime"
)]
async fn minutes_builtin(value: Value) -> crate::BuiltinResult<Value> {
    duration_unit_value(value, "minutes", 1.0 / (24.0 * 60.0)).await
}

#[runmat_macros::runtime_builtin(
    name = "seconds",
    builtin_path = "crate::builtins::duration",
    category = "datetime",
    summary = "Create duration values from seconds or convert duration values to second counts.",
    keywords = "seconds,duration,datetime"
)]
async fn seconds_builtin(value: Value) -> crate::BuiltinResult<Value> {
    duration_unit_value(value, "seconds", 1.0 / SECONDS_PER_DAY).await
}

#[runmat_macros::runtime_builtin(
    name = "milliseconds",
    builtin_path = "crate::builtins::duration",
    category = "datetime",
    summary = "Create duration values from milliseconds or convert duration values to millisecond counts.",
    keywords = "milliseconds,duration,datetime"
)]
async fn milliseconds_builtin(value: Value) -> crate::BuiltinResult<Value> {
    duration_unit_value(value, "milliseconds", 1.0 / (SECONDS_PER_DAY * 1000.0)).await
}

#[runmat_macros::runtime_builtin(
    name = "years",
    builtin_path = "crate::builtins::duration",
    category = "datetime",
    summary = "Create fixed-length duration values from years or convert durations to fixed-length years.",
    keywords = "years,duration,datetime"
)]
async fn years_builtin(value: Value) -> crate::BuiltinResult<Value> {
    duration_unit_value(value, "years", 365.2425).await
}

#[runmat_macros::runtime_builtin(
    name = "isduration",
    builtin_path = "crate::builtins::duration",
    category = "datetime",
    summary = "Return true for duration values.",
    keywords = "isduration,duration,predicate"
)]
fn isduration_builtin(value: Value) -> crate::BuiltinResult<Value> {
    Ok(Value::Bool(is_duration_object(&value)))
}

#[runmat_macros::runtime_builtin(
    name = "duration.subsref",
    descriptor(crate::builtins::duration::DURATION_SUBSREF_DESCRIPTOR),
    builtin_path = "crate::builtins::duration"
)]
async fn duration_subsref(obj: Value, kind: String, payload: Value) -> crate::BuiltinResult<Value> {
    match kind.as_str() {
        OBJECT_INDEX_PAREN => duration_indexing(obj, payload).await,
        OBJECT_INDEX_MEMBER => {
            let Value::Object(object) = obj else {
                return Err(duration_error(
                    "duration.subsref: receiver must be a duration object",
                ));
            };
            let field = scalar_text(&payload, "field selector")?;
            match field.as_str() {
                FORMAT_FIELD => Ok(Value::String(format_for_object(&object))),
                _ => Err(duration_error(format!(
                    "duration.subsref: unsupported duration property '{field}'"
                ))),
            }
        }
        other => Err(duration_error(format!(
            "duration.subsref: unsupported indexing kind '{other}'"
        ))),
    }
}

#[runmat_macros::runtime_builtin(
    name = "duration.subsasgn",
    descriptor(crate::builtins::duration::DURATION_SUBSASGN_DESCRIPTOR),
    builtin_path = "crate::builtins::duration"
)]
async fn duration_subsasgn(
    obj: Value,
    kind: String,
    payload: Value,
    rhs: Value,
) -> crate::BuiltinResult<Value> {
    let Value::Object(mut object) = obj else {
        return Err(duration_error(
            "duration.subsasgn: receiver must be a duration object",
        ));
    };
    match kind.as_str() {
        OBJECT_INDEX_MEMBER => {
            let field = scalar_text(&payload, "field selector")?;
            match field.as_str() {
                FORMAT_FIELD => {
                    let text = scalar_text(&rhs, "Format value")?;
                    object
                        .properties
                        .insert(FORMAT_FIELD.to_string(), Value::String(text));
                    Ok(Value::Object(object))
                }
                _ => Err(duration_error(format!(
                    "duration.subsasgn: unsupported duration property '{field}'"
                ))),
            }
        }
        _ => Err(duration_error(format!(
            "duration.subsasgn: unsupported indexing kind '{kind}'"
        ))),
    }
}

#[runmat_macros::runtime_builtin(
    name = "duration.eq",
    descriptor(crate::builtins::duration::DURATION_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::duration"
)]
async fn duration_eq(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    compare_duration(lhs, rhs, "eq", |a, b| (a - b).abs() <= 1e-12)
}

#[runmat_macros::runtime_builtin(
    name = "duration.ne",
    descriptor(crate::builtins::duration::DURATION_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::duration"
)]
async fn duration_ne(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    compare_duration(lhs, rhs, "ne", |a, b| (a - b).abs() > 1e-12)
}

#[runmat_macros::runtime_builtin(
    name = "duration.lt",
    descriptor(crate::builtins::duration::DURATION_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::duration"
)]
async fn duration_lt(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    compare_duration(lhs, rhs, "lt", |a, b| a < b)
}

#[runmat_macros::runtime_builtin(
    name = "duration.le",
    descriptor(crate::builtins::duration::DURATION_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::duration"
)]
async fn duration_le(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    compare_duration(lhs, rhs, "le", |a, b| a <= b)
}

#[runmat_macros::runtime_builtin(
    name = "duration.gt",
    descriptor(crate::builtins::duration::DURATION_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::duration"
)]
async fn duration_gt(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    compare_duration(lhs, rhs, "gt", |a, b| a > b)
}

#[runmat_macros::runtime_builtin(
    name = "duration.ge",
    descriptor(crate::builtins::duration::DURATION_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::duration"
)]
async fn duration_ge(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    compare_duration(lhs, rhs, "ge", |a, b| a >= b)
}

#[runmat_macros::runtime_builtin(
    name = "duration.plus",
    descriptor(crate::builtins::duration::DURATION_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::duration"
)]
async fn duration_plus(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let lhs_days = duration_tensor_from_duration_value(&lhs)?;
    if crate::builtins::datetime::is_datetime_object(&rhs) {
        let rhs_serials = crate::builtins::datetime::serials_from_datetime_value(&rhs)?;
        let (left, right, shape) =
            tensor::binary_numeric_tensors(&lhs_days, &rhs_serials, "plus", BUILTIN_NAME)?;
        let serials = left
            .iter()
            .zip(right.iter())
            .map(|(a, b)| a + b)
            .collect::<Vec<_>>();
        let tensor =
            Tensor::new(serials, shape).map_err(|err| duration_error(format!("plus: {err}")))?;
        return crate::builtins::datetime::datetime_object_from_serial_tensor(
            tensor,
            crate::builtins::datetime::datetime_format_from_value(&rhs),
        );
    }

    let rhs_days = duration_tensor_from_duration_value(&rhs)?;
    let (left, right, shape) =
        tensor::binary_numeric_tensors(&lhs_days, &rhs_days, "plus", BUILTIN_NAME)?;
    let days = left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| a + b)
        .collect::<Vec<_>>();
    duration_object_from_days(days, shape, duration_format_from_value(&lhs))
}

#[runmat_macros::runtime_builtin(
    name = "duration.minus",
    descriptor(crate::builtins::duration::DURATION_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::duration"
)]
async fn duration_minus(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let lhs_days = duration_tensor_from_duration_value(&lhs)?;
    let rhs_days = duration_tensor_from_duration_value(&rhs)?;
    let (left, right, shape) =
        tensor::binary_numeric_tensors(&lhs_days, &rhs_days, "minus", BUILTIN_NAME)?;
    let days = left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| a - b)
        .collect::<Vec<_>>();
    duration_object_from_days(days, shape, duration_format_from_value(&lhs))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_duration(args: Vec<Value>) -> Value {
        futures::executor::block_on(duration_builtin(args)).expect("duration")
    }

    fn integer_tensor(storage: runmat_builtins::IntegerStorage, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new_integer(storage, shape).expect("integer tensor"))
    }

    #[test]
    fn duration_descriptor_signatures_cover_constructor_and_methods() {
        let labels: Vec<&str> = DURATION_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"t = duration(X)"));
        assert!(labels.contains(&"t = duration(hours, minutes, seconds)"));
        assert!(labels.contains(&"t = duration(hours, minutes, seconds, milliseconds)"));
        assert!(labels.contains(&"t = duration(___, \"Format\", format)"));

        let four_component = DURATION_DESCRIPTOR
            .signatures
            .iter()
            .find(|signature| {
                signature.label == "t = duration(hours, minutes, seconds, milliseconds)"
            })
            .expect("four-component duration signature");
        assert_eq!(
            four_component
                .inputs
                .iter()
                .map(|input| input.name)
                .collect::<Vec<_>>(),
            ["hours", "minutes", "seconds", "milliseconds"]
        );
        assert!(four_component.inputs.iter().all(|input| {
            matches!(input.ty, BuiltinParamType::NumericArray)
                && matches!(input.arity, BuiltinParamArity::Required)
        }));

        assert_eq!(
            DURATION_SUBSREF_DESCRIPTOR.signatures[0].label,
            "out = duration.subsref(obj, kind, payload)"
        );
        assert_eq!(
            DURATION_BINARY_DESCRIPTOR.signatures[0].label,
            "out = duration.op(lhs, rhs)"
        );
    }

    #[test]
    fn duration_builds_from_components() {
        let value = run_duration(vec![Value::Num(1.0), Value::Num(30.0), Value::Num(45.0)]);
        let rendered = duration_display_text(&value)
            .expect("display")
            .expect("duration text");
        assert_eq!(rendered, "01:30:45");
    }

    #[test]
    fn duration_formats_arrays() {
        let hours = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
        let minutes = Value::Tensor(Tensor::new(vec![15.0, 45.0], vec![1, 2]).unwrap());
        let value = run_duration(vec![hours, minutes, Value::Num(0.0)]);
        let rendered = duration_display_text(&value)
            .expect("display")
            .expect("duration text");
        assert!(rendered.contains("01:15:00"));
        assert!(rendered.contains("02:45:00"));
    }

    #[test]
    fn duration_typed_integer_components_cross_double_boundary_exactly() {
        let hours = integer_tensor(runmat_builtins::IntegerStorage::U8(vec![1, 2]), vec![1, 2]);
        let minutes = integer_tensor(
            runmat_builtins::IntegerStorage::U16(vec![15, 45]),
            vec![1, 2],
        );
        let seconds = integer_tensor(
            runmat_builtins::IntegerStorage::I16(vec![0, 30]),
            vec![1, 2],
        );
        let value = run_duration(vec![hours, minutes, seconds]);
        let rendered = duration_display_text(&value)
            .expect("display")
            .expect("duration text");
        assert!(rendered.contains("01:15:00"));
        assert!(rendered.contains("02:45:30"));
    }

    #[test]
    fn duration_integer_matrix_form_supports_all_classes_and_returns_column() {
        let storages = [
            runmat_builtins::IntegerStorage::I8(vec![1, 2, 15, 45, 0, 30]),
            runmat_builtins::IntegerStorage::I16(vec![1, 2, 15, 45, 0, 30]),
            runmat_builtins::IntegerStorage::I32(vec![1, 2, 15, 45, 0, 30]),
            runmat_builtins::IntegerStorage::I64(vec![1, 2, 15, 45, 0, 30]),
            runmat_builtins::IntegerStorage::U8(vec![1, 2, 15, 45, 0, 30]),
            runmat_builtins::IntegerStorage::U16(vec![1, 2, 15, 45, 0, 30]),
            runmat_builtins::IntegerStorage::U32(vec![1, 2, 15, 45, 0, 30]),
            runmat_builtins::IntegerStorage::U64(vec![1, 2, 15, 45, 0, 30]),
        ];
        for storage in storages {
            let value = run_duration(vec![integer_tensor(storage, vec![2, 3])]);
            let days = duration_tensor_from_duration_value(&value).expect("duration days");
            assert_eq!(days.shape, vec![2, 1]);
            let rendered = duration_display_text(&value)
                .expect("display")
                .expect("duration text");
            assert!(rendered.contains("01:15:00"));
            assert!(rendered.contains("02:45:30"));
        }
    }

    #[test]
    fn duration_four_component_form_adds_milliseconds() {
        let value = run_duration(vec![
            Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap()),
            Value::Num(0.0),
            Value::Num(1.0),
            Value::Int(runmat_builtins::IntValue::U16(250)),
        ]);
        let days = duration_tensor_from_duration_value(&value).expect("duration days");
        assert_eq!(days.shape, vec![1, 2]);
        let seconds: Vec<f64> = days
            .materialize_f64()
            .into_iter()
            .map(|days| days * SECONDS_PER_DAY)
            .collect();
        assert!((seconds[0] - 1.25).abs() < 1.0e-12);
        assert!((seconds[1] - 3601.25).abs() < 1.0e-9);
    }

    #[test]
    fn duration_public_components_preserve_nan_and_infinity() {
        let value = run_duration(vec![Value::Num(f64::NAN), Value::Num(0.0), Value::Num(0.0)]);
        assert!(duration_tensor_from_duration_value(&value)
            .unwrap()
            .materialize_f64()[0]
            .is_nan());
        for infinite in [f64::INFINITY, f64::NEG_INFINITY] {
            let value = run_duration(vec![Value::Num(infinite), Value::Num(0.0), Value::Num(0.0)]);
            assert_eq!(
                duration_tensor_from_duration_value(&value)
                    .unwrap()
                    .materialize_f64()[0],
                infinite
            );
            let expected = if infinite.is_sign_negative() {
                "-Inf"
            } else {
                "Inf"
            };
            assert_eq!(
                duration_display_text(&value).expect("display"),
                Some(expected.to_string())
            );
            assert_eq!(
                duration_string_array(&value)
                    .expect("string conversion")
                    .expect("duration string array")
                    .data,
                vec![expected.to_string()]
            );
            let chars = duration_char_array(&value)
                .expect("char conversion")
                .expect("duration char array");
            assert_eq!(chars.data.iter().collect::<String>(), expected);
        }
    }

    #[test]
    fn duration_short_numeric_forms_are_extension_gated_but_matrix_is_public() {
        let strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = futures::executor::block_on(duration_builtin(vec![Value::Num(1.0)]))
            .expect_err("one-component hour extension");
        assert_eq!(
            error.identifier(),
            DURATION_SHORT_COMPONENT_EXTENSION.error_identifier
        );
        let matrix = integer_tensor(
            runmat_builtins::IntegerStorage::U8(vec![1, 30, 0]),
            vec![1, 3],
        );
        futures::executor::block_on(duration_builtin(vec![matrix]))
            .expect("documented matrix form remains public");
        drop(strict);

        let extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        futures::executor::block_on(duration_builtin(vec![Value::Num(1.0)]))
            .expect("one-component extension in RunMat mode");
        futures::executor::block_on(duration_builtin(vec![Value::Num(1.0), Value::Num(30.0)]))
            .expect("two-component extension in RunMat mode");
        drop(extensions);
    }

    #[test]
    fn duration_gpu_extension_rejects_before_provider_access() {
        let strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 3],
            device_id: 0,
            buffer_id: 9_399_002,
        });
        let error = futures::executor::block_on(duration_builtin(vec![resident]))
            .expect_err("GPU extension gate");
        assert_eq!(
            error.identifier(),
            DURATION_GPU_INPUT_EXTENSION.error_identifier
        );
        drop(strict);
    }

    #[test]
    fn duration_missing_days_render_without_error() {
        let value = duration_object_from_days_tensor(
            Tensor::new(vec![f64::NAN], vec![1, 1]).unwrap(),
            DEFAULT_DURATION_FORMAT,
        )
        .expect("duration object");
        let rendered = duration_string_array(&value)
            .expect("string array")
            .expect("duration strings");
        assert_eq!(rendered.data, vec!["NaN".to_string()]);
        assert_eq!(
            duration_display_text(&value).expect("display"),
            Some("NaN".to_string())
        );
    }

    #[test]
    fn duration_unit_helpers_create_and_convert_values() {
        let one_day = futures::executor::block_on(days_builtin(Value::Num(1.0))).expect("days");
        assert!(is_duration_object(&one_day));
        let as_hours = futures::executor::block_on(hours_builtin(one_day.clone())).expect("hours");
        assert_eq!(as_hours, Value::Num(24.0));
        let as_minutes =
            futures::executor::block_on(minutes_builtin(one_day.clone())).expect("minutes");
        assert_eq!(as_minutes, Value::Num(1440.0));
        let as_seconds =
            futures::executor::block_on(seconds_builtin(one_day.clone())).expect("seconds");
        assert_eq!(as_seconds, Value::Num(86_400.0));
        let as_millis =
            futures::executor::block_on(milliseconds_builtin(one_day.clone())).expect("millis");
        assert_eq!(as_millis, Value::Num(86_400_000.0));

        let two_hours = futures::executor::block_on(hours_builtin(Value::Num(2.0))).expect("hours");
        let rendered = duration_display_text(&two_hours)
            .expect("display")
            .expect("duration text");
        assert_eq!(rendered, "02:00:00");

        let year = futures::executor::block_on(years_builtin(Value::Num(1.0))).expect("years");
        let year_days = duration_tensor_from_duration_value(&year).expect("duration tensor");
        assert!((tensor::tensor_value_f64(&year_days, 0) - 365.2425).abs() < 1e-9);
        assert_eq!(
            isduration_builtin(year).expect("isduration"),
            Value::Bool(true)
        );
        assert_eq!(
            isduration_builtin(Value::Num(1.0)).expect("isduration"),
            Value::Bool(false)
        );
        assert!(futures::executor::block_on(years_builtin(Value::Num(f64::MAX))).is_err());
    }

    #[test]
    fn duration_unit_helpers_read_typed_integer_days_exactly() {
        let days =
            Tensor::new_integer(runmat_builtins::IntegerStorage::I16(vec![1, 2]), vec![1, 2])
                .expect("integer tensor");
        let value = duration_object_from_days_tensor(days, DEFAULT_DURATION_FORMAT)
            .expect("duration object");

        let hours = futures::executor::block_on(hours_builtin(value.clone())).expect("hours");
        assert_eq!(
            hours,
            Value::Tensor(Tensor::new(vec![24.0, 48.0], vec![1, 2]).unwrap())
        );

        let rendered = duration_display_text(&value)
            .expect("display")
            .expect("duration text");
        assert!(rendered.contains("24:00:00"));
        assert!(rendered.contains("48:00:00"));
        assert_eq!(
            duration_summary(&value).expect("summary"),
            Some("[1x2 duration]".to_string())
        );
    }

    #[test]
    fn duration_supports_format_assignment_and_indexing() {
        let value = run_duration(vec![Value::Num(1.0), Value::Num(5.0), Value::Num(0.0)]);
        let updated = futures::executor::block_on(duration_subsasgn(
            value.clone(),
            ".".to_string(),
            Value::String(FORMAT_FIELD.to_string()),
            Value::String("hh:mm".to_string()),
        ))
        .expect("subsasgn");
        let rendered = duration_display_text(&updated)
            .expect("display")
            .expect("duration text");
        assert_eq!(rendered, "01:05");

        let array = run_duration(vec![
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            Value::Num(0.0),
            Value::Num(0.0),
        ]);
        let payload =
            Value::Cell(runmat_builtins::CellArray::new(vec![Value::Num(2.0)], 1, 1).unwrap());
        let indexed =
            futures::executor::block_on(duration_subsref(array, "()".to_string(), payload))
                .expect("subsref");
        let text = duration_display_text(&indexed)
            .expect("display")
            .expect("duration text");
        assert_eq!(text, "02:00:00");
    }

    #[test]
    fn duration_typed_integer_index_selectors_are_exact() {
        let array = run_duration(vec![
            integer_tensor(runmat_builtins::IntegerStorage::U8(vec![1, 2]), vec![1, 2]),
            Value::Num(0.0),
            Value::Num(0.0),
        ]);
        let payload = Value::Cell(
            runmat_builtins::CellArray::new(
                vec![integer_tensor(
                    runmat_builtins::IntegerStorage::U64(vec![2]),
                    vec![1, 1],
                )],
                1,
                1,
            )
            .unwrap(),
        );
        let indexed =
            futures::executor::block_on(duration_subsref(array, "()".to_string(), payload))
                .expect("subsref");
        let text = duration_display_text(&indexed)
            .expect("display")
            .expect("duration text");
        assert_eq!(text, "02:00:00");
    }
}
