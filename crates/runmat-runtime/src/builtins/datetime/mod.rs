use std::collections::HashMap;
use std::sync::OnceLock;

use chrono::{DateTime, Datelike, Duration, Local, NaiveDate, NaiveDateTime, Timelike, Weekday};
use runmat_builtins::{
    Access, BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, ClassDef, MethodDef, ObjectInstance, PropertyDef, StringArray, Tensor, Value,
};

use crate::builtins::common::tensor;
use crate::{
    build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError, OBJECT_INDEX_MEMBER,
    OBJECT_INDEX_PAREN, OBJECT_SUBSASGN_METHOD, OBJECT_SUBSREF_METHOD,
};

const BUILTIN_NAME: &str = "datetime";
const DATETIME_CLASS: &str = "datetime";
const SERIAL_FIELD: &str = "__serial";
const FORMAT_FIELD: &str = "Format";
const DEFAULT_DATE_FORMAT: &str = "dd-MMM-yyyy";
const DEFAULT_DATETIME_FORMAT: &str = "dd-MMM-yyyy HH:mm:ss";
const UNIX_DATENUM: f64 = 719_529.0;
const SECONDS_PER_DAY: f64 = 86_400.0;

static DATETIME_CLASS_REGISTERED: OnceLock<()> = OnceLock::new();

const DATETIME_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DATETIME.INVALID_ARGUMENT",
    identifier: Some("RunMat:datetime:InvalidArgument"),
    when: "Arguments or option grammar do not match supported datetime forms.",
    message: "datetime: invalid argument",
};
const DATETIME_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DATETIME.INVALID_INPUT",
    identifier: Some("RunMat:datetime:InvalidInput"),
    when: "Input values cannot be parsed/converted/broadcast to a valid datetime result.",
    message: "datetime: invalid input",
};
const DATETIME_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DATETIME.INTERNAL",
    identifier: Some("RunMat:datetime:Internal"),
    when: "Internal datetime state or indexing/evaluation failed unexpectedly.",
    message: "datetime: internal operation failed",
};
const DATETIME_ERRORS: [BuiltinErrorDescriptor; 3] = [
    DATETIME_ERROR_INVALID_ARGUMENT,
    DATETIME_ERROR_INVALID_INPUT,
    DATETIME_ERROR_INTERNAL,
];

const OUT_DATETIME: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "t",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Datetime object result.",
}];
const OUT_NUMERIC: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric scalar/tensor result.",
}];
const OUT_ANY: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Method result.",
}];
const DATETIME_ARGS_ONLY: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Datetime constructor arguments.",
}];
const DATETIME_SINGLE_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Datetime input.",
}];
const DATETIME_BINARY_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "lhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Left datetime operand.",
    },
    BuiltinParamDescriptor {
        name: "rhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right datetime/numeric/duration operand.",
    },
];
const DATESHIFT_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "t",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Datetime input.",
    },
    BuiltinParamDescriptor {
        name: "boundary",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Shift boundary: 'start', 'end', or 'nearest'.",
    },
    BuiltinParamDescriptor {
        name: "unit",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Calendar/time unit.",
    },
    BuiltinParamDescriptor {
        name: "weekdayOrOption",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional weekday for week-based shifts.",
    },
];
const DATETIME_SUBSREF_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Datetime receiver object.",
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
const DATETIME_SUBSASGN_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Datetime receiver object.",
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

const DATETIME_SIGNATURES: [BuiltinSignatureDescriptor; 11] = [
    BuiltinSignatureDescriptor {
        label: "t = datetime()",
        inputs: &[],
        outputs: &OUT_DATETIME,
    },
    BuiltinSignatureDescriptor {
        label: "t = datetime(textOrArray)",
        inputs: &[BuiltinParamDescriptor {
            name: "textOrArray",
            ty: BuiltinParamType::Any,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "String/char/date text input.",
        }],
        outputs: &OUT_DATETIME,
    },
    BuiltinSignatureDescriptor {
        label: "t = datetime(serialDateNumbers)",
        inputs: &[BuiltinParamDescriptor {
            name: "serialDateNumbers",
            ty: BuiltinParamType::NumericArray,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "Numeric serial date input.",
        }],
        outputs: &OUT_DATETIME,
    },
    BuiltinSignatureDescriptor {
        label: "t = datetime(year, month, day)",
        inputs: &[
            BuiltinParamDescriptor {
                name: "year",
                ty: BuiltinParamType::NumericArray,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Year component.",
            },
            BuiltinParamDescriptor {
                name: "month",
                ty: BuiltinParamType::NumericArray,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Month component.",
            },
            BuiltinParamDescriptor {
                name: "day",
                ty: BuiltinParamType::NumericArray,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Day component.",
            },
        ],
        outputs: &OUT_DATETIME,
    },
    BuiltinSignatureDescriptor {
        label: "t = datetime(year, month, day, hour)",
        inputs: &[
            BuiltinParamDescriptor {
                name: "year",
                ty: BuiltinParamType::NumericArray,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Year component.",
            },
            BuiltinParamDescriptor {
                name: "month",
                ty: BuiltinParamType::NumericArray,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Month component.",
            },
            BuiltinParamDescriptor {
                name: "day",
                ty: BuiltinParamType::NumericArray,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Day component.",
            },
            BuiltinParamDescriptor {
                name: "hour",
                ty: BuiltinParamType::NumericArray,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Hour component.",
            },
        ],
        outputs: &OUT_DATETIME,
    },
    BuiltinSignatureDescriptor {
        label: "t = datetime(year, month, day, hour, minute)",
        inputs: &[
            BuiltinParamDescriptor {
                name: "year",
                ty: BuiltinParamType::NumericArray,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Year component.",
            },
            BuiltinParamDescriptor {
                name: "month",
                ty: BuiltinParamType::NumericArray,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Month component.",
            },
            BuiltinParamDescriptor {
                name: "day",
                ty: BuiltinParamType::NumericArray,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Day component.",
            },
            BuiltinParamDescriptor {
                name: "hour",
                ty: BuiltinParamType::NumericArray,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Hour component.",
            },
            BuiltinParamDescriptor {
                name: "minute",
                ty: BuiltinParamType::NumericArray,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Minute component.",
            },
        ],
        outputs: &OUT_DATETIME,
    },
    BuiltinSignatureDescriptor {
        label: "t = datetime(year, month, day, hour, minute, second)",
        inputs: &[
            BuiltinParamDescriptor {
                name: "year",
                ty: BuiltinParamType::NumericArray,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Year component.",
            },
            BuiltinParamDescriptor {
                name: "month",
                ty: BuiltinParamType::NumericArray,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Month component.",
            },
            BuiltinParamDescriptor {
                name: "day",
                ty: BuiltinParamType::NumericArray,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Day component.",
            },
            BuiltinParamDescriptor {
                name: "hour",
                ty: BuiltinParamType::NumericArray,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Hour component.",
            },
            BuiltinParamDescriptor {
                name: "minute",
                ty: BuiltinParamType::NumericArray,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Minute component.",
            },
            BuiltinParamDescriptor {
                name: "second",
                ty: BuiltinParamType::NumericArray,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Second component.",
            },
        ],
        outputs: &OUT_DATETIME,
    },
    BuiltinSignatureDescriptor {
        label: "t = datetime(serialDateNumbers, \"ConvertFrom\", \"datenum\")",
        inputs: &[BuiltinParamDescriptor {
            name: "args",
            ty: BuiltinParamType::Any,
            arity: BuiltinParamArity::Variadic,
            default: None,
            description: "Numeric serial input with ConvertFrom option.",
        }],
        outputs: &OUT_DATETIME,
    },
    BuiltinSignatureDescriptor {
        label: "t = datetime(___, \"Format\", format)",
        inputs: &DATETIME_ARGS_ONLY,
        outputs: &OUT_DATETIME,
    },
    BuiltinSignatureDescriptor {
        label: "t = datetime(textOrArray, \"InputFormat\", inputFormat)",
        inputs: &DATETIME_ARGS_ONLY,
        outputs: &OUT_DATETIME,
    },
    BuiltinSignatureDescriptor {
        label: "t = datetime(___, Name, Value, ...)",
        inputs: &DATETIME_ARGS_ONLY,
        outputs: &OUT_DATETIME,
    },
];

const DATETIME_YEAR_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "X = year(t)",
    inputs: &DATETIME_SINGLE_INPUT,
    outputs: &OUT_NUMERIC,
}];
const DATETIME_MONTH_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "X = month(t)",
    inputs: &DATETIME_SINGLE_INPUT,
    outputs: &OUT_NUMERIC,
}];
const DATETIME_DAY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "X = day(t)",
    inputs: &DATETIME_SINGLE_INPUT,
    outputs: &OUT_NUMERIC,
}];
const DATETIME_HOUR_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "X = hour(t)",
    inputs: &DATETIME_SINGLE_INPUT,
    outputs: &OUT_NUMERIC,
}];
const DATETIME_MINUTE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "X = minute(t)",
    inputs: &DATETIME_SINGLE_INPUT,
    outputs: &OUT_NUMERIC,
}];
const DATETIME_SECOND_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "X = second(t)",
    inputs: &DATETIME_SINGLE_INPUT,
    outputs: &OUT_NUMERIC,
}];
const DATETIME_SUBSREF_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = datetime.subsref(obj, kind, payload)",
    inputs: &DATETIME_SUBSREF_INPUTS,
    outputs: &OUT_ANY,
}];
const DATETIME_SUBSASGN_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "out = datetime.subsasgn(obj, kind, payload, rhs)",
        inputs: &DATETIME_SUBSASGN_INPUTS,
        outputs: &OUT_ANY,
    }];
const DATETIME_BINARY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = datetime.op(lhs, rhs)",
    inputs: &DATETIME_BINARY_INPUTS,
    outputs: &OUT_ANY,
}];
const DATESHIFT_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "t2 = dateshift(t, boundary, unit)",
        inputs: &DATESHIFT_INPUTS,
        outputs: &OUT_DATETIME,
    },
    BuiltinSignatureDescriptor {
        label: "t2 = dateshift(t, boundary, \"week\", weekday)",
        inputs: &DATESHIFT_INPUTS,
        outputs: &OUT_DATETIME,
    },
    BuiltinSignatureDescriptor {
        label: "t2 = dateshift(t, \"dayofweek\", weekday)",
        inputs: &DATESHIFT_INPUTS,
        outputs: &OUT_DATETIME,
    },
];

pub const DATETIME_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DATETIME_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DATETIME_ERRORS,
};
pub const DATETIME_YEAR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DATETIME_YEAR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DATETIME_ERRORS,
};
pub const DATETIME_MONTH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DATETIME_MONTH_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DATETIME_ERRORS,
};
pub const DATETIME_DAY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DATETIME_DAY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DATETIME_ERRORS,
};
pub const DATETIME_HOUR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DATETIME_HOUR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DATETIME_ERRORS,
};
pub const DATETIME_MINUTE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DATETIME_MINUTE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DATETIME_ERRORS,
};
pub const DATETIME_SECOND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DATETIME_SECOND_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DATETIME_ERRORS,
};
pub const DATETIME_SUBSREF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DATETIME_SUBSREF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &DATETIME_ERRORS,
};
pub const DATETIME_SUBSASGN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DATETIME_SUBSASGN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &DATETIME_ERRORS,
};
pub const DATETIME_BINARY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DATETIME_BINARY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &DATETIME_ERRORS,
};
pub const DATESHIFT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DATESHIFT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DATETIME_ERRORS,
};

fn datetime_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn ensure_datetime_class_registered() {
    DATETIME_CLASS_REGISTERED.get_or_init(|| {
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
                default_value: Some(Value::String(DEFAULT_DATETIME_FORMAT.to_string())),
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
                    function_name: format!("{DATETIME_CLASS}.{name}"),
                    implicit_class_argument: None,
                },
            );
        }

        runmat_builtins::register_class(ClassDef {
            name: DATETIME_CLASS.to_string(),
            parent: None,
            properties,
            methods,
        });
    });
}

async fn gather_args(args: &[Value]) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(args.len());
    for arg in args {
        out.push(
            gather_if_needed_async(arg)
                .await
                .map_err(|err| datetime_error(format!("datetime: {}", err.message())))?,
        );
    }
    Ok(out)
}

fn scalar_text(value: &Value, context: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        Value::CharArray(array) if array.rows == 1 => Ok(array.data.iter().collect()),
        _ => Err(datetime_error(format!(
            "datetime: {context} must be a string scalar or character vector"
        ))),
    }
}

#[derive(Default)]
struct DatetimeOptions {
    format: Option<String>,
    convert_from: Option<String>,
    input_format: Option<String>,
}

fn parse_trailing_options(args: &[Value]) -> BuiltinResult<(usize, DatetimeOptions)> {
    let mut positional_end = args.len();
    let mut options = DatetimeOptions::default();

    while positional_end >= 2 {
        let name = match scalar_text(&args[positional_end - 2], "option name") {
            Ok(text) => text,
            Err(_) => break,
        };
        let lowered = name.trim().to_ascii_lowercase();
        let value = scalar_text(&args[positional_end - 1], &format!("{name} option"))?;
        match lowered.as_str() {
            "format" => options.format = Some(value),
            "convertfrom" => options.convert_from = Some(value),
            "inputformat" => options.input_format = Some(value),
            _ => break,
        }
        positional_end -= 2;
    }

    Ok((positional_end, options))
}

fn tensor_from_numeric(value: Value, context: &str) -> BuiltinResult<Tensor> {
    tensor::value_into_tensor_for(context, value)
        .map_err(|message| datetime_error(format!("datetime: {message}")))
}

fn serial_tensor_from_value(value: Value, context: &str) -> BuiltinResult<Tensor> {
    let tensor = tensor_from_numeric(value, context)?;
    Tensor::new(
        tensor.data.clone(),
        tensor::default_shape_for(&tensor.shape, tensor.data.len()),
    )
    .map_err(|err| datetime_error(format!("datetime: {err}")))
}

fn format_for_object(obj: &ObjectInstance) -> String {
    match obj.properties.get(FORMAT_FIELD) {
        Some(Value::String(text)) => text.clone(),
        Some(Value::StringArray(array)) if array.data.len() == 1 => array.data[0].clone(),
        Some(Value::CharArray(array)) if array.rows == 1 => array.data.iter().collect(),
        _ => DEFAULT_DATETIME_FORMAT.to_string(),
    }
}

fn serial_tensor_for_object(obj: &ObjectInstance) -> BuiltinResult<Tensor> {
    match obj.properties.get(SERIAL_FIELD) {
        Some(Value::Tensor(tensor)) => Ok(tensor.clone()),
        Some(Value::Num(value)) => Tensor::new(vec![*value], vec![1, 1])
            .map_err(|err| datetime_error(format!("datetime: {err}"))),
        Some(other) => Err(datetime_error(format!(
            "datetime: invalid internal serial storage {other:?}"
        ))),
        None => Err(datetime_error("datetime: missing internal serial storage")),
    }
}

pub(crate) fn datetime_object_from_serial_tensor(
    serials: Tensor,
    format: impl Into<String>,
) -> BuiltinResult<Value> {
    ensure_datetime_class_registered();
    let mut object = ObjectInstance::new(DATETIME_CLASS.to_string());
    object
        .properties
        .insert(SERIAL_FIELD.to_string(), Value::Tensor(serials));
    object
        .properties
        .insert(FORMAT_FIELD.to_string(), Value::String(format.into()));
    Ok(Value::Object(object))
}

fn datetime_object_from_serials(
    serials: Vec<f64>,
    shape: Vec<usize>,
    format: impl Into<String>,
) -> BuiltinResult<Value> {
    let tensor =
        Tensor::new(serials, shape).map_err(|err| datetime_error(format!("datetime: {err}")))?;
    datetime_object_from_serial_tensor(tensor, format)
}

fn format_token_to_strftime(format: &str) -> String {
    let mut out = format.to_string();
    for (src, dst) in [
        ("yyyy", "%Y"),
        ("MMM", "%b"),
        ("MM", "%m"),
        ("dd", "%d"),
        ("HH", "%H"),
        ("mm", "%M"),
        ("ss", "%S"),
    ] {
        out = out.replace(src, dst);
    }
    out
}

pub(crate) fn datenum_from_naive(datetime: NaiveDateTime) -> f64 {
    let base = NaiveDate::from_ymd_opt(1970, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    let duration = datetime - base;
    let seconds = duration.num_seconds();
    let nanos = (duration - Duration::seconds(seconds))
        .num_nanoseconds()
        .unwrap_or(0);
    let total_seconds = seconds as f64 + nanos as f64 / 1_000_000_000.0;
    total_seconds / SECONDS_PER_DAY + UNIX_DATENUM
}

fn naive_from_datenum(serial: f64) -> BuiltinResult<NaiveDateTime> {
    if !serial.is_finite() {
        return Err(datetime_error(
            "datetime: serial date numbers must be finite",
        ));
    }
    let total_nanos = ((serial - UNIX_DATENUM) * SECONDS_PER_DAY * 1_000_000_000.0).round() as i128;
    let seconds = total_nanos.div_euclid(1_000_000_000) as i64;
    let nanos = total_nanos.rem_euclid(1_000_000_000) as i64;
    let base = NaiveDate::from_ymd_opt(1970, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    Ok(base + Duration::seconds(seconds) + Duration::nanoseconds(nanos))
}

fn format_serial(serial: f64, format: &str) -> BuiltinResult<String> {
    let naive = naive_from_datenum(serial)?;
    let chrono_format = format_token_to_strftime(format);
    Ok(naive.format(&chrono_format).to_string())
}

fn parse_datetime_text(text: &str) -> Option<(NaiveDateTime, bool)> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }

    if let Ok(value) = DateTime::parse_from_rfc3339(trimmed) {
        return Some((value.with_timezone(&Local).naive_local(), true));
    }

    for (pattern, has_time) in [
        ("%Y-%m-%d %H:%M:%S", true),
        ("%Y-%m-%d", false),
        ("%d-%b-%Y %H:%M:%S", true),
        ("%d-%b-%Y", false),
        ("%m/%d/%Y %H:%M:%S", true),
        ("%m/%d/%Y", false),
    ] {
        if has_time {
            if let Ok(value) = NaiveDateTime::parse_from_str(trimmed, pattern) {
                return Some((value, true));
            }
        } else if let Ok(value) = NaiveDate::parse_from_str(trimmed, pattern) {
            return Some((value.and_hms_opt(0, 0, 0).unwrap(), false));
        }
    }

    None
}

fn parse_datetime_text_with_input_format(
    text: &str,
    input_format: Option<&str>,
) -> Option<(NaiveDateTime, bool)> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    let Some(input_format) = input_format else {
        return parse_datetime_text(trimmed);
    };
    let chrono_format = format_token_to_strftime(input_format);
    if let Ok(value) = NaiveDateTime::parse_from_str(trimmed, &chrono_format) {
        return Some((value, true));
    }
    if let Ok(value) = NaiveDate::parse_from_str(trimmed, &chrono_format) {
        return Some((value.and_hms_opt(0, 0, 0).unwrap(), false));
    }
    None
}

fn parse_text_input(
    value: Value,
    input_format: Option<&str>,
) -> BuiltinResult<(Vec<f64>, Vec<usize>, String)> {
    match value {
        Value::String(text) => {
            if text.trim().eq_ignore_ascii_case("now") {
                let now = Local::now().naive_local();
                return Ok((
                    vec![datenum_from_naive(now)],
                    vec![1, 1],
                    DEFAULT_DATETIME_FORMAT.to_string(),
                ));
            }
            let (naive, has_time) = parse_datetime_text_with_input_format(&text, input_format)
                .ok_or_else(|| {
                    datetime_error(format!("datetime: unable to parse date/time text '{text}'"))
                })?;
            Ok((
                vec![datenum_from_naive(naive)],
                vec![1, 1],
                if has_time {
                    DEFAULT_DATETIME_FORMAT.to_string()
                } else {
                    DEFAULT_DATE_FORMAT.to_string()
                },
            ))
        }
        Value::StringArray(array) => {
            let mut serials = Vec::with_capacity(array.data.len());
            let mut has_time = false;
            for text in &array.data {
                let (naive, parsed_has_time) =
                    parse_datetime_text_with_input_format(text, input_format).ok_or_else(|| {
                        datetime_error(format!("datetime: unable to parse date/time text '{text}'"))
                    })?;
                serials.push(datenum_from_naive(naive));
                has_time |= parsed_has_time;
            }
            Ok((
                serials,
                tensor::default_shape_for(&array.shape, array.data.len()),
                if has_time {
                    DEFAULT_DATETIME_FORMAT.to_string()
                } else {
                    DEFAULT_DATE_FORMAT.to_string()
                },
            ))
        }
        Value::CharArray(array) => {
            let mut texts = Vec::with_capacity(array.rows);
            for row in 0..array.rows {
                let start = row * array.cols;
                let end = start + array.cols;
                texts.push(
                    array.data[start..end]
                        .iter()
                        .collect::<String>()
                        .trim_end()
                        .to_string(),
                );
            }
            parse_text_input(
                Value::StringArray(
                    StringArray::new(texts, vec![array.rows, 1])
                        .map_err(|err| datetime_error(format!("datetime: {err}")))?,
                ),
                input_format,
            )
        }
        _ => Err(datetime_error(
            "datetime: text input must be a string scalar, string array, or character array",
        )),
    }
}

fn round_component(value: f64, label: &str, min: i64, max: i64) -> BuiltinResult<i64> {
    if !value.is_finite() {
        return Err(datetime_error(format!(
            "datetime: {label} values must be finite"
        )));
    }
    let rounded = value.round();
    if (rounded - value).abs() > 1e-9 {
        return Err(datetime_error(format!(
            "datetime: {label} values must be integers"
        )));
    }
    let integer = rounded as i64;
    if integer < min || integer > max {
        return Err(datetime_error(format!(
            "datetime: {label} values must be in the range [{min}, {max}]"
        )));
    }
    Ok(integer)
}

fn naive_from_components(
    year: f64,
    month: f64,
    day: f64,
    hour: f64,
    minute: f64,
    second: f64,
) -> BuiltinResult<NaiveDateTime> {
    let year = round_component(year, "year", -262_000, 262_000)? as i32;
    let month = round_component(month, "month", 1, 12)? as u32;
    let day = round_component(day, "day", 1, 31)? as u32;
    let hour = round_component(hour, "hour", 0, 23)? as u32;
    let minute = round_component(minute, "minute", 0, 59)? as u32;
    if !second.is_finite() {
        return Err(datetime_error("datetime: second values must be finite"));
    }
    if !(0.0..60.0).contains(&second) {
        return Err(datetime_error(
            "datetime: second values must be in the range [0, 60)",
        ));
    }

    let base_date = NaiveDate::from_ymd_opt(year, month, day)
        .ok_or_else(|| datetime_error("datetime: invalid calendar date"))?;
    let whole_second = second.floor();
    let mut nanos = ((second - whole_second) * 1_000_000_000.0).round() as u32;
    let mut secs = whole_second as u32;
    if nanos == 1_000_000_000 {
        secs += 1;
        nanos = 0;
    }
    let time = base_date
        .and_hms_nano_opt(hour, minute, secs, nanos)
        .ok_or_else(|| datetime_error("datetime: invalid time components"))?;
    Ok(time)
}

fn broadcast_component_data(
    arrays: &[Tensor],
    labels: &[&str],
) -> BuiltinResult<(Vec<Vec<f64>>, Vec<usize>)> {
    let mut target_shape = vec![1, 1];
    let mut target_len = 1usize;

    for array in arrays {
        let len = array.data.len();
        if len > 1 {
            let shape = tensor::default_shape_for(&array.shape, len);
            if target_len == 1 {
                target_len = len;
                target_shape = shape;
            } else if len != target_len || shape != target_shape {
                return Err(datetime_error(
                    "datetime: non-scalar component inputs must have matching sizes",
                ));
            }
        }
    }

    let mut broadcasted = Vec::with_capacity(arrays.len());
    for (idx, array) in arrays.iter().enumerate() {
        if array.data.len() == 1 {
            broadcasted.push(vec![array.data[0]; target_len]);
        } else if array.data.len() == target_len {
            broadcasted.push(array.data.clone());
        } else {
            return Err(datetime_error(format!(
                "datetime: {} input size does not match the other components",
                labels[idx]
            )));
        }
    }

    Ok((broadcasted, target_shape))
}

fn component_tensor(value: Value, context: &str) -> BuiltinResult<Tensor> {
    let tensor = tensor_from_numeric(value, context)?;
    Tensor::new(
        tensor.data.clone(),
        tensor::default_shape_for(&tensor.shape, tensor.data.len()),
    )
    .map_err(|err| datetime_error(format!("datetime: {err}")))
}

fn build_from_components(args: Vec<Value>, format: Option<String>) -> BuiltinResult<Value> {
    let labels = ["year", "month", "day", "hour", "minute", "second"];
    let input_count = args.len();
    let mut arrays = Vec::with_capacity(args.len());
    for (idx, arg) in args.into_iter().enumerate() {
        arrays.push(component_tensor(arg, labels[idx])?);
    }
    while arrays.len() < 6 {
        arrays.push(Tensor::new(vec![0.0], vec![1, 1]).unwrap());
    }

    let (broadcasted, shape) = broadcast_component_data(&arrays, &labels)?;
    let len = broadcasted[0].len();
    let mut serials = Vec::with_capacity(len);
    for idx in 0..len {
        let naive = naive_from_components(
            broadcasted[0][idx],
            broadcasted[1][idx],
            broadcasted[2][idx],
            broadcasted[3][idx],
            broadcasted[4][idx],
            broadcasted[5][idx],
        )?;
        serials.push(datenum_from_naive(naive));
    }

    let default_format = if let Some(format) = format {
        format
    } else if input_count > 3 {
        DEFAULT_DATETIME_FORMAT.to_string()
    } else {
        DEFAULT_DATE_FORMAT.to_string()
    };
    datetime_object_from_serials(serials, shape, default_format)
}

fn numeric_value_to_datetime(value: Value, format: Option<String>) -> BuiltinResult<Value> {
    let serials = serial_tensor_from_value(value, "datetime")?;
    datetime_object_from_serial_tensor(
        serials,
        format.unwrap_or_else(|| DEFAULT_DATETIME_FORMAT.to_string()),
    )
}

pub fn is_datetime_object(value: &Value) -> bool {
    matches!(value, Value::Object(obj) if obj.is_class(DATETIME_CLASS))
}

pub(crate) fn serials_from_datetime_value(value: &Value) -> BuiltinResult<Tensor> {
    match value {
        Value::Object(obj) if obj.is_class(DATETIME_CLASS) => serial_tensor_for_object(obj),
        _ => Err(datetime_error("datetime: expected a datetime value")),
    }
}

pub(crate) fn datetime_format_from_value(value: &Value) -> String {
    match value {
        Value::Object(obj) if obj.is_class(DATETIME_CLASS) => format_for_object(obj),
        _ => DEFAULT_DATETIME_FORMAT.to_string(),
    }
}

pub fn datetime_string_array(value: &Value) -> BuiltinResult<Option<StringArray>> {
    let Value::Object(obj) = value else {
        return Ok(None);
    };
    if !obj.is_class(DATETIME_CLASS) {
        return Ok(None);
    }
    let serials = serial_tensor_for_object(obj)?;
    let format = format_for_object(obj);
    let mut strings = Vec::with_capacity(serials.data.len());
    for serial in &serials.data {
        strings.push(format_serial(*serial, &format)?);
    }
    let shape = tensor::default_shape_for(&serials.shape, serials.data.len());
    let array = StringArray::new(strings, shape)
        .map_err(|err| datetime_error(format!("datetime: {err}")))?;
    Ok(Some(array))
}

pub fn datetime_display_text(value: &Value) -> BuiltinResult<Option<String>> {
    let Some(array) = datetime_string_array(value)? else {
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
            widths[col] = widths[col].max(array.data[idx].chars().count());
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
            let padding = widths[col].saturating_sub(text.chars().count());
            if padding > 0 {
                line.push_str(&" ".repeat(padding));
            }
        }
        lines.push(line);
    }
    Ok(Some(lines.join("\n")))
}

pub fn datetime_summary(value: &Value) -> BuiltinResult<Option<String>> {
    let Value::Object(obj) = value else {
        return Ok(None);
    };
    if !obj.is_class(DATETIME_CLASS) {
        return Ok(None);
    }
    let serials = serial_tensor_for_object(obj)?;
    if serials.data.len() == 1 {
        return datetime_display_text(value);
    }
    let shape = tensor::default_shape_for(&serials.shape, serials.data.len());
    Ok(Some(format!(
        "[{} datetime]",
        shape
            .iter()
            .map(|dim| dim.to_string())
            .collect::<Vec<_>>()
            .join("x")
    )))
}

fn component_tensor_from_datetime(
    value: &Value,
    label: &str,
    extractor: impl Fn(&NaiveDateTime) -> f64,
) -> BuiltinResult<Value> {
    let serials = serials_from_datetime_value(value)?;
    let mut out = Vec::with_capacity(serials.data.len());
    for serial in &serials.data {
        let naive = naive_from_datenum(*serial)?;
        out.push(extractor(&naive));
    }
    if out.len() == 1 {
        Ok(Value::Num(out[0]))
    } else {
        let shape = tensor::default_shape_for(&serials.shape, serials.data.len());
        let tensor =
            Tensor::new(out, shape).map_err(|err| datetime_error(format!("{label}: {err}")))?;
        Ok(Value::Tensor(tensor))
    }
}

fn tensor_or_scalar(data: Vec<f64>, shape: Vec<usize>) -> BuiltinResult<Value> {
    if data.len() == 1 {
        Ok(Value::Num(data[0]))
    } else {
        Ok(Value::Tensor(Tensor::new(data, shape).map_err(|err| {
            datetime_error(format!("datetime: {err}"))
        })?))
    }
}

async fn datetime_indexing(obj: Value, payload: Value) -> BuiltinResult<Value> {
    let Value::Object(object) = obj else {
        return Err(datetime_error(
            "datetime.subsref: receiver must be a datetime object",
        ));
    };
    let format = format_for_object(&object);
    let serials = serial_tensor_for_object(&object)?;

    let Value::Cell(cell) = payload else {
        return Err(datetime_error(
            "datetime.subsref: indexing payload must be a cell array",
        ));
    };
    if cell.data.is_empty() {
        return datetime_object_from_serial_tensor(serials, format);
    }
    if cell.data.len() != 1 {
        return Err(datetime_error(
            "datetime.subsref: only linear datetime indexing is currently supported",
        ));
    }
    let selector = (*cell.data[0]).clone();
    let selector = match selector {
        Value::Tensor(tensor) => tensor,
        Value::Num(value) => Tensor::new(vec![value], vec![1, 1])
            .map_err(|err| datetime_error(format!("datetime.subsref: {err}")))?,
        Value::Int(value) => Tensor::new(vec![value.to_f64()], vec![1, 1])
            .map_err(|err| datetime_error(format!("datetime.subsref: {err}")))?,
        Value::LogicalArray(logical) => tensor::logical_to_tensor(&logical)
            .map_err(|err| datetime_error(format!("datetime.subsref: {err}")))?,
        other => {
            return Err(datetime_error(format!(
                "datetime.subsref: unsupported index value {other:?}"
            )))
        }
    };
    let indexed = crate::perform_indexing(&Value::Tensor(serials), &selector.data)
        .await
        .map_err(|err| datetime_error(format!("datetime.subsref: {}", err.message())))?;
    let indexed_serials = match indexed {
        Value::Num(value) => Tensor::new(vec![value], vec![1, 1])
            .map_err(|err| datetime_error(format!("datetime.subsref: {err}")))?,
        Value::Tensor(tensor) => tensor,
        other => {
            return Err(datetime_error(format!(
                "datetime.subsref: unexpected indexing result {other:?}"
            )))
        }
    };
    datetime_object_from_serial_tensor(indexed_serials, format)
}

#[runmat_macros::runtime_builtin(
    name = "datetime",
    descriptor(crate::builtins::datetime::DATETIME_DESCRIPTOR),
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Create datetime arrays from text, components, or serial date numbers.",
    keywords = "datetime,date,time,datenum,Format",
    related = "year,month,day,hour,minute,second,string,char,disp",
    examples = "t = datetime(2024, 4, 9, 13, 30, 0);"
)]
async fn datetime_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    ensure_datetime_class_registered();
    let args = gather_args(&args).await?;
    let (positional_end, options) = parse_trailing_options(&args)?;
    let positional = args[..positional_end].to_vec();

    if let Some(convert_from) = options.convert_from {
        if !convert_from.eq_ignore_ascii_case("datenum") {
            return Err(datetime_error(format!(
                "datetime: unsupported ConvertFrom value '{convert_from}'"
            )));
        }
        if positional.len() != 1 {
            return Err(datetime_error(
                "datetime: ConvertFrom='datenum' expects exactly one numeric input",
            ));
        }
        return numeric_value_to_datetime(positional[0].clone(), options.format);
    }

    match positional.len() {
        0 => {
            let now = Local::now().naive_local();
            datetime_object_from_serials(
                vec![datenum_from_naive(now)],
                vec![1, 1],
                options
                    .format
                    .unwrap_or_else(|| DEFAULT_DATETIME_FORMAT.to_string()),
            )
        }
        1 => match &positional[0] {
            Value::Object(obj) if obj.is_class(DATETIME_CLASS) => {
                let serials = serials_from_datetime_value(&positional[0])?;
                let format = options
                    .format
                    .unwrap_or_else(|| datetime_format_from_value(&positional[0]));
                datetime_object_from_serial_tensor(serials, format)
            }
            Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
                let (serials, shape, inferred_format) =
                    parse_text_input(positional[0].clone(), options.input_format.as_deref())?;
                datetime_object_from_serials(
                    serials,
                    shape,
                    options.format.unwrap_or(inferred_format),
                )
            }
            _ => numeric_value_to_datetime(positional[0].clone(), options.format),
        },
        3..=6 => build_from_components(positional, options.format),
        _ => Err(datetime_error(
            "datetime: unsupported argument pattern; use text, serial dates, or Y/M/D component inputs",
        )),
    }
}

#[runmat_macros::runtime_builtin(
    name = "year",
    descriptor(crate::builtins::datetime::DATETIME_YEAR_DESCRIPTOR),
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Extract calendar year components from datetime values.",
    keywords = "year,datetime,date component"
)]
async fn year_builtin(value: Value) -> crate::BuiltinResult<Value> {
    component_tensor_from_datetime(&value, "year", |naive| naive.year() as f64)
}

#[runmat_macros::runtime_builtin(
    name = "month",
    descriptor(crate::builtins::datetime::DATETIME_MONTH_DESCRIPTOR),
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Extract month numbers from datetime arrays.",
    keywords = "month,datetime,date component"
)]
async fn month_builtin(value: Value) -> crate::BuiltinResult<Value> {
    component_tensor_from_datetime(&value, "month", |naive| naive.month() as f64)
}

#[runmat_macros::runtime_builtin(
    name = "day",
    descriptor(crate::builtins::datetime::DATETIME_DAY_DESCRIPTOR),
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Extract day-of-month numbers from datetime values.",
    keywords = "day,datetime,date component"
)]
async fn day_builtin(value: Value) -> crate::BuiltinResult<Value> {
    component_tensor_from_datetime(&value, "day", |naive| naive.day() as f64)
}

#[runmat_macros::runtime_builtin(
    name = "hour",
    descriptor(crate::builtins::datetime::DATETIME_HOUR_DESCRIPTOR),
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Extract hour components from datetime values.",
    keywords = "hour,datetime,time component"
)]
async fn hour_builtin(value: Value) -> crate::BuiltinResult<Value> {
    component_tensor_from_datetime(&value, "hour", |naive| naive.hour() as f64)
}

#[runmat_macros::runtime_builtin(
    name = "minute",
    descriptor(crate::builtins::datetime::DATETIME_MINUTE_DESCRIPTOR),
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Extract minute numbers from datetime arrays.",
    keywords = "minute,datetime,time component"
)]
async fn minute_builtin(value: Value) -> crate::BuiltinResult<Value> {
    component_tensor_from_datetime(&value, "minute", |naive| naive.minute() as f64)
}

#[runmat_macros::runtime_builtin(
    name = "second",
    descriptor(crate::builtins::datetime::DATETIME_SECOND_DESCRIPTOR),
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Extract second components from datetime values.",
    keywords = "second,datetime,time component"
)]
async fn second_builtin(value: Value) -> crate::BuiltinResult<Value> {
    component_tensor_from_datetime(&value, "second", |naive| {
        naive.second() as f64 + f64::from(naive.nanosecond()) / 1_000_000_000.0
    })
}

#[runmat_macros::runtime_builtin(
    name = "datetime.subsref",
    descriptor(crate::builtins::datetime::DATETIME_SUBSREF_DESCRIPTOR),
    builtin_path = "crate::builtins::datetime"
)]
async fn datetime_subsref(obj: Value, kind: String, payload: Value) -> crate::BuiltinResult<Value> {
    match kind.as_str() {
        OBJECT_INDEX_PAREN => datetime_indexing(obj, payload).await,
        OBJECT_INDEX_MEMBER => {
            let Value::Object(object) = obj else {
                return Err(datetime_error(
                    "datetime.subsref: receiver must be a datetime object",
                ));
            };
            let field = scalar_text(&payload, "field selector")?;
            match field.as_str() {
                FORMAT_FIELD => Ok(Value::String(format_for_object(&object))),
                _ => Err(datetime_error(format!(
                    "datetime.subsref: unsupported datetime property '{field}'"
                ))),
            }
        }
        other => Err(datetime_error(format!(
            "datetime.subsref: unsupported indexing kind '{other}'"
        ))),
    }
}

#[runmat_macros::runtime_builtin(
    name = "datetime.subsasgn",
    descriptor(crate::builtins::datetime::DATETIME_SUBSASGN_DESCRIPTOR),
    builtin_path = "crate::builtins::datetime"
)]
async fn datetime_subsasgn(
    obj: Value,
    kind: String,
    payload: Value,
    rhs: Value,
) -> crate::BuiltinResult<Value> {
    let Value::Object(mut object) = obj else {
        return Err(datetime_error(
            "datetime.subsasgn: receiver must be a datetime object",
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
                _ => Err(datetime_error(format!(
                    "datetime.subsasgn: unsupported datetime property '{field}'"
                ))),
            }
        }
        _ => Err(datetime_error(format!(
            "datetime.subsasgn: unsupported indexing kind '{kind}'"
        ))),
    }
}

fn datetime_binary_serials(
    lhs: Value,
    rhs: Value,
    context: &str,
) -> BuiltinResult<(Tensor, Tensor, Vec<usize>, String)> {
    let lhs_serials = serials_from_datetime_value(&lhs)?;
    let rhs_serials = match &rhs {
        Value::Object(obj) if obj.is_class(DATETIME_CLASS) => serial_tensor_for_object(obj)?,
        _ => serial_tensor_from_value(rhs, context)?,
    };
    let (left, right, shape) =
        tensor::binary_numeric_tensors(&lhs_serials, &rhs_serials, context, BUILTIN_NAME)?;
    let left_tensor = Tensor::new(left, shape.clone())
        .map_err(|err| datetime_error(format!("{context}: {err}")))?;
    let right_tensor = Tensor::new(right, shape.clone())
        .map_err(|err| datetime_error(format!("{context}: {err}")))?;
    Ok((
        left_tensor,
        right_tensor,
        shape,
        datetime_format_from_value(&lhs),
    ))
}

fn compare_datetime(
    lhs: Value,
    rhs: Value,
    op: &str,
    cmp: impl Fn(f64, f64) -> bool,
) -> BuiltinResult<Value> {
    let (left, right, shape, _) = datetime_binary_serials(lhs, rhs, op)?;
    let out = left
        .data
        .iter()
        .zip(right.data.iter())
        .map(|(a, b)| if cmp(*a, *b) { 1.0 } else { 0.0 })
        .collect::<Vec<_>>();
    tensor_or_scalar(out, shape)
}

#[runmat_macros::runtime_builtin(
    name = "datetime.eq",
    descriptor(crate::builtins::datetime::DATETIME_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::datetime"
)]
async fn datetime_eq(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    compare_datetime(lhs, rhs, "eq", |a, b| (a - b).abs() <= 1e-12)
}

#[runmat_macros::runtime_builtin(
    name = "datetime.ne",
    descriptor(crate::builtins::datetime::DATETIME_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::datetime"
)]
async fn datetime_ne(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    compare_datetime(lhs, rhs, "ne", |a, b| (a - b).abs() > 1e-12)
}

#[runmat_macros::runtime_builtin(
    name = "datetime.lt",
    descriptor(crate::builtins::datetime::DATETIME_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::datetime"
)]
async fn datetime_lt(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    compare_datetime(lhs, rhs, "lt", |a, b| a < b)
}

#[runmat_macros::runtime_builtin(
    name = "datetime.le",
    descriptor(crate::builtins::datetime::DATETIME_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::datetime"
)]
async fn datetime_le(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    compare_datetime(lhs, rhs, "le", |a, b| a <= b)
}

#[runmat_macros::runtime_builtin(
    name = "datetime.gt",
    descriptor(crate::builtins::datetime::DATETIME_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::datetime"
)]
async fn datetime_gt(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    compare_datetime(lhs, rhs, "gt", |a, b| a > b)
}

#[runmat_macros::runtime_builtin(
    name = "datetime.ge",
    descriptor(crate::builtins::datetime::DATETIME_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::datetime"
)]
async fn datetime_ge(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    compare_datetime(lhs, rhs, "ge", |a, b| a >= b)
}

#[runmat_macros::runtime_builtin(
    name = "datetime.plus",
    descriptor(crate::builtins::datetime::DATETIME_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::datetime"
)]
async fn datetime_plus(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let lhs_serials = serials_from_datetime_value(&lhs)?;
    let rhs_numeric = if crate::builtins::duration::is_duration_object(&rhs) {
        crate::builtins::duration::duration_tensor_from_duration_value(&rhs)?
    } else {
        serial_tensor_from_value(rhs, "plus")?
    };
    let (left, right, shape) =
        tensor::binary_numeric_tensors(&lhs_serials, &rhs_numeric, "plus", BUILTIN_NAME)?;
    let serials = left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| a + b)
        .collect::<Vec<_>>();
    datetime_object_from_serials(serials, shape, datetime_format_from_value(&lhs))
}

#[runmat_macros::runtime_builtin(
    name = "datetime.minus",
    descriptor(crate::builtins::datetime::DATETIME_BINARY_DESCRIPTOR),
    builtin_path = "crate::builtins::datetime"
)]
async fn datetime_minus(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let lhs_serials = serials_from_datetime_value(&lhs)?;
    match &rhs {
        _ if crate::builtins::duration::is_duration_object(&rhs) => {
            let rhs_days = crate::builtins::duration::duration_tensor_from_duration_value(&rhs)?;
            let (left, right, shape) =
                tensor::binary_numeric_tensors(&lhs_serials, &rhs_days, "minus", BUILTIN_NAME)?;
            let serials = left
                .iter()
                .zip(right.iter())
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>();
            datetime_object_from_serials(serials, shape, datetime_format_from_value(&lhs))
        }
        Value::Object(obj) if obj.is_class(DATETIME_CLASS) => {
            let rhs_serials = serial_tensor_for_object(obj)?;
            let (left, right, shape) =
                tensor::binary_numeric_tensors(&lhs_serials, &rhs_serials, "minus", BUILTIN_NAME)?;
            let deltas = left
                .iter()
                .zip(right.iter())
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>();
            tensor_or_scalar(deltas, shape)
        }
        _ => {
            let rhs_numeric = serial_tensor_from_value(rhs, "minus")?;
            let (left, right, shape) =
                tensor::binary_numeric_tensors(&lhs_serials, &rhs_numeric, "minus", BUILTIN_NAME)?;
            let serials = left
                .iter()
                .zip(right.iter())
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>();
            datetime_object_from_serials(serials, shape, datetime_format_from_value(&lhs))
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DateShiftBoundary {
    Start,
    End,
    Nearest,
    DayOfWeek,
}

impl DateShiftBoundary {
    fn parse(value: &Value) -> BuiltinResult<Self> {
        let text = scalar_text(value, "dateshift boundary")?;
        match text.trim().to_ascii_lowercase().as_str() {
            "start" => Ok(Self::Start),
            "end" => Ok(Self::End),
            "nearest" => Ok(Self::Nearest),
            "dayofweek" => Ok(Self::DayOfWeek),
            other => Err(datetime_error(format!(
                "dateshift: unsupported boundary '{other}'"
            ))),
        }
    }
}

#[derive(Clone, Copy)]
enum DateShiftUnit {
    Year,
    Quarter,
    Month,
    Week,
    Day,
    Hour,
    Minute,
    Second,
}

impl DateShiftUnit {
    fn parse(value: &Value) -> BuiltinResult<Self> {
        let text = scalar_text(value, "dateshift unit")?;
        match text.trim().to_ascii_lowercase().as_str() {
            "year" | "years" => Ok(Self::Year),
            "quarter" | "quarters" => Ok(Self::Quarter),
            "month" | "months" => Ok(Self::Month),
            "week" | "weeks" => Ok(Self::Week),
            "day" | "days" => Ok(Self::Day),
            "hour" | "hours" => Ok(Self::Hour),
            "minute" | "minutes" => Ok(Self::Minute),
            "second" | "seconds" => Ok(Self::Second),
            other => Err(datetime_error(format!(
                "dateshift: unsupported unit '{other}'"
            ))),
        }
    }
}

fn parse_weekday(value: &Value) -> BuiltinResult<Weekday> {
    match value {
        Value::Num(n) if n.is_finite() && (*n - n.round()).abs() <= f64::EPSILON => {
            weekday_from_matlab_index(n.round() as i64)
        }
        Value::Int(i) => weekday_from_matlab_index(i.to_i64()),
        _ => {
            let text = scalar_text(value, "weekday")?;
            match text.trim().to_ascii_lowercase().as_str() {
                "sun" | "sunday" => Ok(Weekday::Sun),
                "mon" | "monday" => Ok(Weekday::Mon),
                "tue" | "tues" | "tuesday" => Ok(Weekday::Tue),
                "wed" | "wednesday" => Ok(Weekday::Wed),
                "thu" | "thur" | "thurs" | "thursday" => Ok(Weekday::Thu),
                "fri" | "friday" => Ok(Weekday::Fri),
                "sat" | "saturday" => Ok(Weekday::Sat),
                other => Err(datetime_error(format!(
                    "dateshift: unsupported weekday '{other}'"
                ))),
            }
        }
    }
}

fn weekday_from_matlab_index(index: i64) -> BuiltinResult<Weekday> {
    match index {
        1 => Ok(Weekday::Sun),
        2 => Ok(Weekday::Mon),
        3 => Ok(Weekday::Tue),
        4 => Ok(Weekday::Wed),
        5 => Ok(Weekday::Thu),
        6 => Ok(Weekday::Fri),
        7 => Ok(Weekday::Sat),
        _ => Err(datetime_error(
            "dateshift: numeric weekdays must be in the range 1..7",
        )),
    }
}

fn midnight(date: NaiveDate) -> NaiveDateTime {
    date.and_hms_opt(0, 0, 0).unwrap()
}

fn start_of_week(value: NaiveDateTime, week_start: Weekday) -> NaiveDateTime {
    let current = value.weekday().num_days_from_monday() as i64;
    let start = week_start.num_days_from_monday() as i64;
    let delta = (current - start).rem_euclid(7);
    midnight(value.date() - Duration::days(delta))
}

fn start_of_unit(value: NaiveDateTime, unit: DateShiftUnit, week_start: Weekday) -> NaiveDateTime {
    match unit {
        DateShiftUnit::Year => midnight(NaiveDate::from_ymd_opt(value.year(), 1, 1).unwrap()),
        DateShiftUnit::Quarter => {
            let month = ((value.month() - 1) / 3) * 3 + 1;
            midnight(NaiveDate::from_ymd_opt(value.year(), month, 1).unwrap())
        }
        DateShiftUnit::Month => {
            midnight(NaiveDate::from_ymd_opt(value.year(), value.month(), 1).unwrap())
        }
        DateShiftUnit::Week => start_of_week(value, week_start),
        DateShiftUnit::Day => midnight(value.date()),
        DateShiftUnit::Hour => value
            .date()
            .and_hms_nano_opt(value.hour(), 0, 0, 0)
            .unwrap(),
        DateShiftUnit::Minute => value
            .date()
            .and_hms_nano_opt(value.hour(), value.minute(), 0, 0)
            .unwrap(),
        DateShiftUnit::Second => value
            .date()
            .and_hms_nano_opt(value.hour(), value.minute(), value.second(), 0)
            .unwrap(),
    }
}

fn add_months(year: i32, month: u32, delta: u32) -> (i32, u32) {
    let zero_based = year as i64 * 12 + i64::from(month - 1) + i64::from(delta);
    let out_year = zero_based.div_euclid(12) as i32;
    let out_month = zero_based.rem_euclid(12) as u32 + 1;
    (out_year, out_month)
}

fn next_unit_start(start: NaiveDateTime, unit: DateShiftUnit) -> NaiveDateTime {
    match unit {
        DateShiftUnit::Year => midnight(NaiveDate::from_ymd_opt(start.year() + 1, 1, 1).unwrap()),
        DateShiftUnit::Quarter => {
            let (year, month) = add_months(start.year(), start.month(), 3);
            midnight(NaiveDate::from_ymd_opt(year, month, 1).unwrap())
        }
        DateShiftUnit::Month => {
            let (year, month) = add_months(start.year(), start.month(), 1);
            midnight(NaiveDate::from_ymd_opt(year, month, 1).unwrap())
        }
        DateShiftUnit::Week => start + Duration::days(7),
        DateShiftUnit::Day => start + Duration::days(1),
        DateShiftUnit::Hour => start + Duration::hours(1),
        DateShiftUnit::Minute => start + Duration::minutes(1),
        DateShiftUnit::Second => start + Duration::seconds(1),
    }
}

fn shift_naive_datetime(
    value: NaiveDateTime,
    boundary: DateShiftBoundary,
    unit: DateShiftUnit,
    week_start: Weekday,
) -> NaiveDateTime {
    let start = start_of_unit(value, unit, week_start);
    match boundary {
        DateShiftBoundary::Start => start,
        DateShiftBoundary::End => next_unit_start(start, unit) - Duration::milliseconds(1),
        DateShiftBoundary::Nearest => {
            let next = next_unit_start(start, unit);
            if value - start <= next - value {
                start
            } else {
                next
            }
        }
        DateShiftBoundary::DayOfWeek => value,
    }
}

fn shift_to_dayofweek(value: NaiveDateTime, weekday: Weekday) -> NaiveDateTime {
    let current = value.weekday().num_days_from_monday() as i64;
    let target = weekday.num_days_from_monday() as i64;
    let delta = (target - current).rem_euclid(7);
    midnight(value.date() + Duration::days(delta))
}

#[runmat_macros::runtime_builtin(
    name = "dateshift",
    descriptor(crate::builtins::datetime::DATESHIFT_DESCRIPTOR),
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Shift datetime values to calendar or clock boundaries.",
    keywords = "dateshift,datetime,start,end,nearest,week,month,year",
    related = "datetime,year,month,day"
)]
async fn dateshift_builtin(
    value: Value,
    boundary: Value,
    unit: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| datetime_error(format!("dateshift: {}", err.message())))?;
    let boundary = gather_if_needed_async(&boundary)
        .await
        .map_err(|err| datetime_error(format!("dateshift: {}", err.message())))?;
    let unit = gather_if_needed_async(&unit)
        .await
        .map_err(|err| datetime_error(format!("dateshift: {}", err.message())))?;
    let rest = gather_args(&rest).await?;
    let serials = serials_from_datetime_value(&value)?;
    let format = datetime_format_from_value(&value);
    let boundary = DateShiftBoundary::parse(&boundary)?;

    let mut out = Vec::with_capacity(serials.data.len());
    if boundary == DateShiftBoundary::DayOfWeek {
        if !rest.is_empty() {
            return Err(datetime_error(
                "dateshift: dayofweek boundary does not accept extra arguments",
            ));
        }
        let weekday = parse_weekday(&unit)?;
        for serial in &serials.data {
            out.push(datenum_from_naive(shift_to_dayofweek(
                naive_from_datenum(*serial)?,
                weekday,
            )));
        }
    } else {
        let unit = DateShiftUnit::parse(&unit)?;
        let week_start = if matches!(unit, DateShiftUnit::Week) {
            if rest.len() > 1 {
                return Err(datetime_error(
                    "dateshift: week unit accepts at most one weekday argument",
                ));
            }
            rest.first()
                .map(parse_weekday)
                .transpose()?
                .unwrap_or(Weekday::Mon)
        } else {
            if !rest.is_empty() {
                return Err(datetime_error(
                    "dateshift: extra arguments are only supported for week units",
                ));
            }
            Weekday::Mon
        };
        for serial in &serials.data {
            out.push(datenum_from_naive(shift_naive_datetime(
                naive_from_datenum(*serial)?,
                boundary,
                unit,
                week_start,
            )));
        }
    }

    let shape = tensor::default_shape_for(&serials.shape, serials.data.len());
    datetime_object_from_serials(out, shape, format)
}

pub fn datetime_char_array(value: &Value) -> BuiltinResult<Option<CharArray>> {
    let Some(array) = datetime_string_array(value)? else {
        return Ok(None);
    };
    let width = array
        .data
        .iter()
        .map(|s| s.chars().count())
        .max()
        .unwrap_or(0);
    let rows = array.data.len();
    let mut data = vec![' '; rows * width];
    for (row, text) in array.data.iter().enumerate() {
        for (col, ch) in text.chars().enumerate() {
            data[row * width + col] = ch;
        }
    }
    let out = CharArray::new(data, rows, width)
        .map_err(|err| datetime_error(format!("datetime: {err}")))?;
    Ok(Some(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_datetime(args: Vec<Value>) -> Value {
        futures::executor::block_on(datetime_builtin(args)).expect("datetime")
    }

    fn as_datetime(value: Value) -> ObjectInstance {
        match value {
            Value::Object(object) => object,
            other => panic!("expected datetime object, got {other:?}"),
        }
    }

    #[test]
    fn datetime_descriptor_signatures_cover_constructor_and_methods() {
        let labels: Vec<&str> = DATETIME_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"t = datetime()"));
        assert!(labels.contains(&"t = datetime(year, month, day, hour, minute, second)"));
        assert!(labels.contains(&"t = datetime(serialDateNumbers, \"ConvertFrom\", \"datenum\")"));

        assert_eq!(DATETIME_YEAR_DESCRIPTOR.signatures[0].label, "X = year(t)");
        assert_eq!(
            DATETIME_SUBSREF_DESCRIPTOR.signatures[0].label,
            "out = datetime.subsref(obj, kind, payload)"
        );
        assert_eq!(
            DATETIME_BINARY_DESCRIPTOR.signatures[0].label,
            "out = datetime.op(lhs, rhs)"
        );
    }

    #[test]
    fn datetime_builds_from_components() {
        let value = run_datetime(vec![Value::Num(2024.0), Value::Num(3.0), Value::Num(14.0)]);
        let object = as_datetime(value);
        assert_eq!(object.class_name, DATETIME_CLASS);
        assert_eq!(format_for_object(&object), DEFAULT_DATE_FORMAT);
        let serials = serial_tensor_for_object(&object).expect("serials");
        assert_eq!(serials.data.len(), 1);
        let year =
            futures::executor::block_on(year_builtin(Value::Object(object.clone()))).expect("year");
        assert_eq!(year, Value::Num(2024.0));
    }

    #[test]
    fn datetime_builds_arrays_from_component_vectors() {
        let years = Value::Tensor(Tensor::new(vec![2024.0, 2025.0], vec![1, 2]).unwrap());
        let months = Value::Tensor(Tensor::new(vec![1.0, 6.0], vec![1, 2]).unwrap());
        let days = Value::Tensor(Tensor::new(vec![15.0, 20.0], vec![1, 2]).unwrap());
        let value = run_datetime(vec![years, months, days]);
        let object = as_datetime(value.clone());
        let serials = serial_tensor_for_object(&object).expect("serials");
        assert_eq!(serials.shape, vec![1, 2]);
        let rendered = datetime_display_text(&value)
            .expect("display")
            .expect("datetime text");
        assert!(rendered.contains("15-Jan-2024"));
        assert!(rendered.contains("20-Jun-2025"));
    }

    #[test]
    fn datetime_parses_text_and_converts_to_strings() {
        let value = run_datetime(vec![Value::String("2024-03-14 09:26:53".to_string())]);
        let rendered = datetime_string_array(&value)
            .expect("string array")
            .expect("datetime strings");
        assert_eq!(rendered.data, vec!["14-Mar-2024 09:26:53".to_string()]);
    }

    #[test]
    fn datetime_accepts_existing_datetime_input() {
        let value = run_datetime(vec![Value::String("2024-03-14".to_string())]);
        let converted = run_datetime(vec![
            value.clone(),
            Value::from("InputFormat"),
            Value::from("yyyy-MM-dd"),
        ]);
        assert_eq!(
            serials_from_datetime_value(&converted).unwrap().data,
            serials_from_datetime_value(&value).unwrap().data
        );
    }

    #[test]
    fn datetime_parses_text_with_input_format() {
        let input = Value::StringArray(
            StringArray::new(
                vec!["2024/03/14".to_string(), "2024/03/15".to_string()],
                vec![2, 1],
            )
            .unwrap(),
        );
        let value = run_datetime(vec![
            input,
            Value::from("InputFormat"),
            Value::from("yyyy/MM/dd"),
            Value::from("Format"),
            Value::from("yyyy-MM-dd"),
        ]);
        let rendered = datetime_string_array(&value)
            .expect("string array")
            .expect("datetime strings");
        assert_eq!(
            rendered.data,
            vec!["2024-03-14".to_string(), "2024-03-15".to_string()]
        );
    }

    #[test]
    fn dateshift_supports_start_of_week_and_month_end() {
        let input = run_datetime(vec![
            Value::StringArray(
                StringArray::new(
                    vec!["2024-03-14".to_string(), "2024-03-18".to_string()],
                    vec![2, 1],
                )
                .unwrap(),
            ),
            Value::from("Format"),
            Value::from("yyyy-MM-dd"),
        ]);
        let shifted = futures::executor::block_on(dateshift_builtin(
            input,
            Value::from("start"),
            Value::from("week"),
            Vec::new(),
        ))
        .expect("dateshift start week");
        let rendered = datetime_string_array(&shifted)
            .expect("string array")
            .expect("datetime strings");
        assert_eq!(
            rendered.data,
            vec!["2024-03-11".to_string(), "2024-03-18".to_string()]
        );

        let month_end = futures::executor::block_on(dateshift_builtin(
            run_datetime(vec![
                Value::from("2024-02-10"),
                Value::from("Format"),
                Value::from("yyyy-MM-dd HH:mm:ss"),
            ]),
            Value::from("end"),
            Value::from("month"),
            Vec::new(),
        ))
        .expect("dateshift end month");
        let rendered = datetime_string_array(&month_end)
            .expect("string array")
            .expect("datetime strings");
        assert_eq!(rendered.data, vec!["2024-02-29 23:59:59".to_string()]);
    }

    #[test]
    fn dateshift_rejects_unsupported_extra_arguments() {
        let input = run_datetime(vec![Value::from("2024-03-14")]);
        let err = futures::executor::block_on(dateshift_builtin(
            input.clone(),
            Value::from("dayofweek"),
            Value::from("monday"),
            vec![Value::from("extra")],
        ))
        .expect_err("dayofweek extra argument should fail");
        assert!(err.message().contains("does not accept extra arguments"));

        let err = futures::executor::block_on(dateshift_builtin(
            input.clone(),
            Value::from("start"),
            Value::from("week"),
            vec![Value::from("monday"), Value::from("extra")],
        ))
        .expect_err("week second extra argument should fail");
        assert!(err.message().contains("at most one weekday argument"));

        let err = futures::executor::block_on(dateshift_builtin(
            input,
            Value::from("start"),
            Value::from("month"),
            vec![Value::from("monday")],
        ))
        .expect_err("non-week extra argument should fail");
        assert!(err
            .message()
            .contains("extra arguments are only supported for week units"));
    }

    #[test]
    fn datetime_supports_format_assignment() {
        let value = run_datetime(vec![Value::Num(2024.0), Value::Num(3.0), Value::Num(14.0)]);
        let updated = futures::executor::block_on(datetime_subsasgn(
            value,
            ".".to_string(),
            Value::String(FORMAT_FIELD.to_string()),
            Value::String("yyyy-MM-dd".to_string()),
        ))
        .expect("subsasgn");
        let rendered = datetime_display_text(&updated)
            .expect("display")
            .expect("datetime text");
        assert_eq!(rendered, "2024-03-14");
    }

    #[test]
    fn datetime_supports_indexing_and_comparison() {
        let years = Value::Tensor(Tensor::new(vec![2024.0, 2025.0], vec![1, 2]).unwrap());
        let months = Value::Tensor(Tensor::new(vec![1.0, 6.0], vec![1, 2]).unwrap());
        let days = Value::Tensor(Tensor::new(vec![15.0, 20.0], vec![1, 2]).unwrap());
        let value = run_datetime(vec![years, months, days]);
        let payload =
            Value::Cell(runmat_builtins::CellArray::new(vec![Value::Num(2.0)], 1, 1).unwrap());
        let indexed =
            futures::executor::block_on(datetime_subsref(value.clone(), "()".to_string(), payload))
                .expect("subsref");
        let year = futures::executor::block_on(year_builtin(indexed)).expect("year");
        assert_eq!(year, Value::Num(2025.0));

        let lhs = run_datetime(vec![Value::Num(2024.0), Value::Num(1.0), Value::Num(1.0)]);
        let rhs = run_datetime(vec![Value::Num(2024.0), Value::Num(1.0), Value::Num(2.0)]);
        let cmp = futures::executor::block_on(datetime_lt(lhs, rhs)).expect("lt");
        assert_eq!(cmp, Value::Num(1.0));
    }

    #[test]
    fn datetime_and_duration_interoperate() {
        let lhs = run_datetime(vec![Value::Num(2024.0), Value::Num(1.0), Value::Num(1.0)]);
        let rhs = run_datetime(vec![Value::Num(2024.0), Value::Num(1.0), Value::Num(2.0)]);
        let delta = futures::executor::block_on(datetime_minus(rhs.clone(), lhs.clone()))
            .expect("datetime minus datetime");
        assert_eq!(delta, Value::Num(1.0));

        let duration = crate::builtins::duration::duration_object_from_days_tensor(
            Tensor::new(vec![1.0], vec![1, 1]).unwrap(),
            crate::builtins::duration::DEFAULT_DURATION_FORMAT,
        )
        .expect("duration");

        let round_trip = futures::executor::block_on(datetime_plus(lhs.clone(), duration.clone()))
            .expect("plus");
        let round_trip_text = datetime_display_text(&round_trip)
            .expect("datetime display")
            .expect("datetime text");
        assert_eq!(round_trip_text, "02-Jan-2024");

        let restored =
            futures::executor::block_on(datetime_minus(rhs, duration)).expect("minus duration");
        let restored_text = datetime_display_text(&restored)
            .expect("datetime display")
            .expect("datetime text");
        assert_eq!(restored_text, "01-Jan-2024");
    }
}
