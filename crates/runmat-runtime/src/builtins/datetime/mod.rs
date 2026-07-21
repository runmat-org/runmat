use std::cell::Cell;
use std::collections::{HashMap, HashSet};

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
const CALENDAR_DURATION_CLASS: &str = "calendarDuration";
const SERIAL_FIELD: &str = "__serial";
const CALENDAR_MONTHS_FIELD: &str = "__months";
const CALENDAR_DAYS_FIELD: &str = "__days";
const FORMAT_FIELD: &str = "Format";
const DEFAULT_DATE_FORMAT: &str = "dd-MMM-yyyy";
const DEFAULT_DATETIME_FORMAT: &str = "dd-MMM-yyyy HH:mm:ss";
const UNIX_DATENUM: f64 = 719_529.0;
const SECONDS_PER_DAY: f64 = 86_400.0;
const MAX_HOLIDAY_YEAR_SPAN: i32 = 1_000;
const MAX_BUSDAYS_OUTPUT_LEN: i64 = 1_000_000;

type Broadcast3 = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>);

thread_local! {
    static DATETIME_CLASS_REGISTERED: Cell<bool> = const { Cell::new(false) };
    static CALENDAR_DURATION_CLASS_REGISTERED: Cell<bool> = const { Cell::new(false) };
}

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
    DATETIME_CLASS_REGISTERED.with(|registered| {
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
        registered.set(true);
    });
}

fn ensure_calendar_duration_class_registered() {
    CALENDAR_DURATION_CLASS_REGISTERED.with(|registered| {
        if registered.get() {
            return;
        }

        let mut properties = HashMap::new();
        for name in [CALENDAR_MONTHS_FIELD, CALENDAR_DAYS_FIELD] {
            properties.insert(
                name.to_string(),
                PropertyDef {
                    name: name.to_string(),
                    is_static: false,
                    is_constant: false,
                    is_dependent: false,
                    get_access: Access::Public,
                    set_access: Access::Public,
                    default_value: Some(Value::Num(0.0)),
                },
            );
        }

        let mut methods = HashMap::new();
        for name in ["plus", "minus", "eq", "ne"] {
            methods.insert(
                name.to_string(),
                MethodDef {
                    name: name.to_string(),
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: Access::Public,
                    function_name: format!("{CALENDAR_DURATION_CLASS}.{name}"),
                    implicit_class_argument: None,
                },
            );
        }

        runmat_builtins::register_class(ClassDef {
            name: CALENDAR_DURATION_CLASS.to_string(),
            parent: None,
            properties,
            methods,
        });
        registered.set(true);
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
    if serial.is_nan() {
        return Ok("NaT".to_string());
    }
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

pub fn is_calendar_duration_object(value: &Value) -> bool {
    matches!(value, Value::Object(obj) if obj.is_class(CALENDAR_DURATION_CLASS))
}

fn calendar_duration_tensor_for_object(obj: &ObjectInstance, field: &str) -> BuiltinResult<Tensor> {
    match obj.properties.get(field) {
        Some(Value::Tensor(tensor)) => Ok(tensor.clone()),
        Some(Value::Num(value)) => Tensor::new(vec![*value], vec![1, 1])
            .map_err(|err| datetime_error(format!("calendarDuration: {err}"))),
        Some(other) => Err(datetime_error(format!(
            "calendarDuration: invalid internal {field} storage {other:?}"
        ))),
        None => Err(datetime_error(format!(
            "calendarDuration: missing internal {field} storage"
        ))),
    }
}

fn calendar_duration_tensors_from_value(value: &Value) -> BuiltinResult<(Tensor, Tensor)> {
    match value {
        Value::Object(obj) if obj.is_class(CALENDAR_DURATION_CLASS) => Ok((
            calendar_duration_tensor_for_object(obj, CALENDAR_MONTHS_FIELD)?,
            calendar_duration_tensor_for_object(obj, CALENDAR_DAYS_FIELD)?,
        )),
        _ => Err(datetime_error(
            "calendarDuration: expected a calendarDuration value",
        )),
    }
}

fn calendar_duration_object_from_tensors(months: Tensor, days: Tensor) -> BuiltinResult<Value> {
    ensure_calendar_duration_class_registered();
    let mut object = ObjectInstance::new(CALENDAR_DURATION_CLASS.to_string());
    object
        .properties
        .insert(CALENDAR_MONTHS_FIELD.to_string(), Value::Tensor(months));
    object
        .properties
        .insert(CALENDAR_DAYS_FIELD.to_string(), Value::Tensor(days));
    Ok(Value::Object(object))
}

fn calendar_duration_object_from_components(
    months: Vec<f64>,
    days: Vec<f64>,
    shape: Vec<usize>,
) -> BuiltinResult<Value> {
    let month_tensor = Tensor::new(months, shape.clone())
        .map_err(|err| datetime_error(format!("calendarDuration: {err}")))?;
    let day_tensor = Tensor::new(days, shape)
        .map_err(|err| datetime_error(format!("calendarDuration: {err}")))?;
    calendar_duration_object_from_tensors(month_tensor, day_tensor)
}

fn calendar_duration_unit_value(
    value: Value,
    unit_name: &str,
    months_per_unit: f64,
    days_per_unit: f64,
) -> BuiltinResult<Value> {
    if is_calendar_duration_object(&value) {
        let (months, days) = calendar_duration_tensors_from_value(&value)?;
        let (month_data, day_data, shape) =
            tensor::binary_numeric_tensors(&months, &days, unit_name, BUILTIN_NAME)?;
        let data = month_data
            .iter()
            .zip(day_data.iter())
            .map(|(months, days)| {
                if months_per_unit != 0.0 {
                    months / months_per_unit + days / 30.436875 / months_per_unit
                } else {
                    days / days_per_unit
                }
            })
            .collect::<Vec<_>>();
        return tensor_or_scalar(data, shape);
    }

    let numeric = component_tensor(value, unit_name)?;
    let shape = tensor::default_shape_for(&numeric.shape, numeric.data.len());
    let mut months = Vec::with_capacity(numeric.data.len());
    let mut days = Vec::with_capacity(numeric.data.len());
    for value in &numeric.data {
        if !value.is_finite() {
            return Err(datetime_error(format!(
                "{unit_name}: values must be finite"
            )));
        }
        let month_value = value * months_per_unit;
        let day_value = value * days_per_unit;
        if !month_value.is_finite() || !day_value.is_finite() {
            return Err(datetime_error(format!(
                "{unit_name}: resulting calendar duration is outside supported range"
            )));
        }
        months.push(month_value);
        days.push(day_value);
    }
    calendar_duration_object_from_components(months, days, shape)
}

fn add_months_clamped(value: NaiveDateTime, month_delta: i64) -> BuiltinResult<NaiveDateTime> {
    let current_month = i64::from(value.year())
        .checked_mul(12)
        .and_then(|base| base.checked_add(i64::from(value.month() - 1)))
        .ok_or_else(|| datetime_error("calendarDuration: result date is out of range"))?;
    let zero_based = current_month
        .checked_add(month_delta)
        .ok_or_else(|| datetime_error("calendarDuration: result date is out of range"))?;
    let year_i64 = zero_based.div_euclid(12);
    let year = i32::try_from(year_i64)
        .map_err(|_| datetime_error("calendarDuration: result date is out of range"))?;
    let month = zero_based.rem_euclid(12) as u32 + 1;
    let day = value.day().min(days_in_month(year, month)?);
    NaiveDate::from_ymd_opt(year, month, day)
        .and_then(|date| {
            date.and_hms_nano_opt(
                value.hour(),
                value.minute(),
                value.second(),
                value.nanosecond(),
            )
        })
        .ok_or_else(|| datetime_error("calendarDuration: result date is out of range"))
}

fn add_fractional_days(value: NaiveDateTime, days: f64) -> BuiltinResult<NaiveDateTime> {
    if !days.is_finite() {
        return Err(datetime_error(
            "calendarDuration: day components must be finite",
        ));
    }
    let nanos = (days * SECONDS_PER_DAY * 1_000_000_000.0).round();
    if !nanos.is_finite() || nanos < i64::MIN as f64 || nanos > i64::MAX as f64 {
        return Err(datetime_error(
            "calendarDuration: day component is outside supported range",
        ));
    }
    Ok(value + Duration::nanoseconds(nanos as i64))
}

fn apply_calendar_duration_to_serials(
    serials: &Tensor,
    months: &Tensor,
    days: &Tensor,
    sign: f64,
    context: &str,
) -> BuiltinResult<(Vec<f64>, Vec<usize>)> {
    let (serial_data, month_data, day_data, shape) =
        broadcast_three_numeric_tensors(serials, months, days, context)?;
    let mut out = Vec::with_capacity(serial_data.len());
    for ((serial, months), days) in serial_data
        .iter()
        .zip(month_data.iter())
        .zip(day_data.iter())
    {
        if !months.is_finite() {
            return Err(datetime_error(format!(
                "{context}: month components must be finite"
            )));
        }
        let signed_months = months * sign;
        let rounded_months = signed_months.round();
        if (rounded_months - signed_months).abs() > 1e-9 {
            return Err(datetime_error(format!(
                "{context}: calendar month components must be integers for datetime arithmetic"
            )));
        }
        if rounded_months < i64::MIN as f64 || rounded_months > i64::MAX as f64 {
            return Err(datetime_error(format!(
                "{context}: calendar month component is outside supported range"
            )));
        }
        let shifted = add_months_clamped(naive_from_datenum(*serial)?, rounded_months as i64)?;
        out.push(datenum_from_naive(add_fractional_days(
            shifted,
            days * sign,
        )?));
    }
    Ok((out, shape))
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

fn numeric_or_datetime_serial_tensor(value: Value, context: &str) -> BuiltinResult<Tensor> {
    match &value {
        Value::Object(obj) if obj.is_class(DATETIME_CLASS) => serial_tensor_for_object(obj),
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
            let (serials, shape, _) = parse_text_input(value, None)?;
            Tensor::new(serials, shape).map_err(|err| datetime_error(format!("{context}: {err}")))
        }
        _ => serial_tensor_from_value(value, context),
    }
}

fn datevec_components_from_serial(serial: f64) -> BuiltinResult<[f64; 6]> {
    let naive = naive_from_datenum(serial)?;
    Ok([
        naive.year() as f64,
        naive.month() as f64,
        naive.day() as f64,
        naive.hour() as f64,
        naive.minute() as f64,
        naive.second() as f64 + f64::from(naive.nanosecond()) / 1_000_000_000.0,
    ])
}

fn datevec_matrix_from_serial_tensor(serials: &Tensor) -> BuiltinResult<Tensor> {
    let rows = serials.data.len();
    let mut data = vec![0.0; rows.saturating_mul(6)];
    for (row, serial) in serials.data.iter().enumerate() {
        let components = datevec_components_from_serial(*serial)?;
        for col in 0..6 {
            data[col * rows + row] = components[col];
        }
    }
    Tensor::new(data, vec![rows, 6]).map_err(|err| datetime_error(format!("datevec: {err}")))
}

fn datetime_from_date_only(
    naive: NaiveDateTime,
    format: impl Into<String>,
) -> BuiltinResult<Value> {
    datetime_object_from_serials(vec![datenum_from_naive(naive)], vec![1, 1], format)
}

fn current_naive_local() -> NaiveDateTime {
    Local::now().naive_local()
}

fn days_in_month(year: i32, month: u32) -> BuiltinResult<u32> {
    let _ = NaiveDate::from_ymd_opt(year, month, 1)
        .ok_or_else(|| datetime_error("eomday: invalid year/month"))?;
    let (next_year, next_month) = if month == 12 {
        (year + 1, 1)
    } else {
        (year, month + 1)
    };
    let next = NaiveDate::from_ymd_opt(next_year, next_month, 1)
        .ok_or_else(|| datetime_error("eomday: invalid year/month"))?;
    Ok((next - Duration::days(1)).day())
}

fn tensor_from_datevec_like(value: Value, context: &str) -> BuiltinResult<Tensor> {
    let tensor = tensor_from_numeric(value, context)?;
    let shape = tensor::default_shape_for(&tensor.shape, tensor.data.len());
    let normalize = |rows: usize, cols: usize, data: Vec<f64>| -> BuiltinResult<Tensor> {
        if cols == 6 {
            return Tensor::new(data, vec![rows, 6])
                .map_err(|err| datetime_error(format!("{context}: {err}")));
        }
        let mut padded = vec![0.0; rows.saturating_mul(6)];
        for col in 0..3 {
            for row in 0..rows {
                padded[col * rows + row] = data[col * rows + row];
            }
        }
        Tensor::new(padded, vec![rows, 6])
            .map_err(|err| datetime_error(format!("{context}: {err}")))
    };
    if tensor.data.len() == 3 {
        return normalize(1, 3, tensor.data);
    }
    if tensor.data.len() == 6 {
        return normalize(1, 6, tensor.data);
    }
    if shape.len() >= 2 && (shape[1] == 3 || shape[1] == 6) {
        return normalize(shape[0], shape[1], tensor.data);
    }
    Err(datetime_error(format!(
        "{context}: expected a date vector with three or six columns"
    )))
}

fn datenum_from_datevec_tensor(tensor: &Tensor, context: &str) -> BuiltinResult<Tensor> {
    let rows = tensor.rows;
    let cols = tensor.cols;
    if cols != 6 {
        return Err(datetime_error(format!(
            "{context}: date vectors must have six columns"
        )));
    }
    let mut out = Vec::with_capacity(rows);
    for row in 0..rows {
        let component = |col: usize| tensor.data[col * rows + row];
        let naive = naive_from_components(
            component(0),
            component(1),
            component(2),
            component(3),
            component(4),
            component(5),
        )?;
        out.push(datenum_from_naive(naive));
    }
    Tensor::new(out, vec![rows, 1]).map_err(|err| datetime_error(format!("{context}: {err}")))
}

fn char_array_from_rows(rows: &[String], context: &str) -> BuiltinResult<CharArray> {
    let width = rows
        .iter()
        .map(|row| row.chars().count())
        .max()
        .unwrap_or(0);
    let mut data = vec![' '; rows.len().saturating_mul(width)];
    for (row_idx, row) in rows.iter().enumerate() {
        for (col, ch) in row.chars().enumerate() {
            data[row_idx * width + col] = ch;
        }
    }
    CharArray::new(data, rows.len(), width)
        .map_err(|err| datetime_error(format!("{context}: {err}")))
}

fn broadcast_three_numeric_tensors(
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    context: &str,
) -> BuiltinResult<Broadcast3> {
    let mut output_shape = Vec::new();
    let mut output_len = 1usize;
    for operand in [a, b, c] {
        if operand.data.len() == 1 {
            continue;
        }
        let shape = tensor::default_shape_for(&operand.shape, operand.data.len());
        if output_shape.is_empty() {
            output_len = operand.data.len();
            output_shape = shape;
        } else if operand.data.len() != output_len || shape != output_shape {
            return Err(datetime_error(format!(
                "{context}: operands must be scalar or have matching sizes"
            )));
        }
    }
    if output_shape.is_empty() {
        output_shape = vec![1, 1];
        output_len = 1;
    }

    let expand = |operand: &Tensor| -> BuiltinResult<Vec<f64>> {
        match operand.data.len() {
            1 => Ok(vec![operand.data[0]; output_len]),
            len if len == output_len => Ok(operand.data.clone()),
            _ => Err(datetime_error(format!(
                "{context}: operands must be scalar or have matching sizes"
            ))),
        }
    };

    Ok((expand(a)?, expand(b)?, expand(c)?, output_shape))
}

fn serial_date_key(serial: f64) -> BuiltinResult<i64> {
    if !serial.is_finite() {
        return Err(datetime_error("date values must be finite"));
    }
    let key = serial.floor();
    if key < i64::MIN as f64 || key > i64::MAX as f64 {
        return Err(datetime_error("date value is outside supported range"));
    }
    Ok(key as i64)
}

fn date_from_key(key: i64) -> BuiltinResult<NaiveDate> {
    Ok(naive_from_datenum(key as f64)?.date())
}

fn key_from_date(date: NaiveDate) -> i64 {
    datenum_from_naive(midnight(date)).floor() as i64
}

fn observed_fixed_holiday(year: i32, month: u32, day: u32) -> BuiltinResult<i64> {
    let date = NaiveDate::from_ymd_opt(year, month, day)
        .ok_or_else(|| datetime_error("holidays: invalid fixed holiday date"))?;
    let observed = match date.weekday() {
        Weekday::Sat => date - Duration::days(1),
        Weekday::Sun => date + Duration::days(1),
        _ => date,
    };
    Ok(key_from_date(observed))
}

fn nth_weekday(year: i32, month: u32, weekday: Weekday, n: u32) -> BuiltinResult<i64> {
    let mut date = NaiveDate::from_ymd_opt(year, month, 1)
        .ok_or_else(|| datetime_error("holidays: invalid nth weekday month"))?;
    while date.weekday() != weekday {
        date += Duration::days(1);
    }
    date += Duration::days(i64::from(n.saturating_sub(1)) * 7);
    Ok(key_from_date(date))
}

fn last_weekday(year: i32, month: u32, weekday: Weekday) -> BuiltinResult<i64> {
    let last_day = days_in_month(year, month)?;
    let mut date = NaiveDate::from_ymd_opt(year, month, last_day)
        .ok_or_else(|| datetime_error("holidays: invalid last weekday month"))?;
    while date.weekday() != weekday {
        date -= Duration::days(1);
    }
    Ok(key_from_date(date))
}

fn easter_sunday(year: i32) -> BuiltinResult<NaiveDate> {
    let a = year.rem_euclid(19);
    let b = year.div_euclid(100);
    let c = year.rem_euclid(100);
    let d = b.div_euclid(4);
    let e = b.rem_euclid(4);
    let f = (b + 8).div_euclid(25);
    let g = (b - f + 1).div_euclid(3);
    let h = (19 * a + b - d - g + 15).rem_euclid(30);
    let i = c.div_euclid(4);
    let k = c.rem_euclid(4);
    let l = (32 + 2 * e + 2 * i - h - k).rem_euclid(7);
    let m = (a + 11 * h + 22 * l).div_euclid(451);
    let month = (h + l - 7 * m + 114).div_euclid(31) as u32;
    let day = ((h + l - 7 * m + 114).rem_euclid(31) + 1) as u32;
    NaiveDate::from_ymd_opt(year, month, day)
        .ok_or_else(|| datetime_error("holidays: invalid computed Easter date"))
}

fn market_holiday_keys_for_year(year: i32) -> BuiltinResult<Vec<i64>> {
    let mut keys = vec![
        observed_fixed_holiday(year, 1, 1)?,
        nth_weekday(year, 1, Weekday::Mon, 3)?,
        nth_weekday(year, 2, Weekday::Mon, 3)?,
        key_from_date(easter_sunday(year)? - Duration::days(2)),
        last_weekday(year, 5, Weekday::Mon)?,
        observed_fixed_holiday(year, 6, 19)?,
        observed_fixed_holiday(year, 7, 4)?,
        nth_weekday(year, 9, Weekday::Mon, 1)?,
        nth_weekday(year, 11, Weekday::Thu, 4)?,
        observed_fixed_holiday(year, 12, 25)?,
    ];
    keys.sort_unstable();
    keys.dedup();
    Ok(keys)
}

fn holiday_keys_between(start_key: i64, end_key: i64) -> BuiltinResult<Vec<i64>> {
    let start_year = date_from_key(start_key.min(end_key))?
        .year()
        .checked_sub(1)
        .ok_or_else(|| datetime_error("holidays: date range is outside supported range"))?;
    let end_year = date_from_key(start_key.max(end_key))?
        .year()
        .checked_add(1)
        .ok_or_else(|| datetime_error("holidays: date range is outside supported range"))?;
    if end_year - start_year > MAX_HOLIDAY_YEAR_SPAN {
        return Err(datetime_error(format!(
            "holidays: date range spans more than {MAX_HOLIDAY_YEAR_SPAN} years"
        )));
    }
    let mut keys = Vec::new();
    for year in start_year..=end_year {
        keys.extend(market_holiday_keys_for_year(year)?);
    }
    keys.sort_unstable();
    keys.dedup();
    Ok(keys
        .into_iter()
        .filter(|key| *key >= start_key.min(end_key) && *key <= start_key.max(end_key))
        .collect())
}

fn holiday_set_for_range(start_key: i64, end_key: i64) -> BuiltinResult<HashSet<i64>> {
    Ok(holiday_keys_between(start_key, end_key)?
        .into_iter()
        .collect())
}

fn holiday_set_from_optional_or_default(
    value: Option<Value>,
    context: &str,
    start_key: i64,
    end_key: i64,
) -> BuiltinResult<HashSet<i64>> {
    if let Some(value) = value {
        let serials = numeric_or_datetime_serial_tensor(value, context)?;
        return serials
            .data
            .iter()
            .map(|serial| serial_date_key(*serial))
            .collect::<BuiltinResult<HashSet<_>>>();
    }
    holiday_set_for_range(start_key, end_key)
}

fn date_key_range(tensors: &[&Tensor]) -> BuiltinResult<(i64, i64)> {
    let mut min_key = i64::MAX;
    let mut max_key = i64::MIN;
    let mut found = false;
    for tensor in tensors {
        for serial in &tensor.data {
            let key = serial_date_key(*serial)?;
            min_key = min_key.min(key);
            max_key = max_key.max(key);
            found = true;
        }
    }
    if found {
        Ok((min_key, max_key))
    } else {
        Ok((0, 0))
    }
}

fn is_business_day_key(key: i64, holidays: &HashSet<i64>) -> BuiltinResult<bool> {
    let date = date_from_key(key)?;
    Ok(!matches!(date.weekday(), Weekday::Sat | Weekday::Sun) && !holidays.contains(&key))
}

fn count_weekdays_forward(start_key: i64, end_key: i64) -> BuiltinResult<i64> {
    let total_days = end_key
        .checked_sub(start_key)
        .and_then(|delta| delta.checked_add(1))
        .ok_or_else(|| datetime_error("business-day date range is outside supported range"))?;
    let full_weeks = total_days / 7;
    let mut count = full_weeks * 5;
    let remainder = total_days % 7;
    for offset in 0..remainder {
        let key = start_key
            .checked_add(offset)
            .ok_or_else(|| datetime_error("business-day date range is outside supported range"))?;
        if !matches!(date_from_key(key)?.weekday(), Weekday::Sat | Weekday::Sun) {
            count += 1;
        }
    }
    Ok(count)
}

fn count_business_days(
    start_key: i64,
    end_key: i64,
    holidays: &HashSet<i64>,
) -> BuiltinResult<i64> {
    if start_key > end_key {
        return Ok(-count_business_days(end_key, start_key, holidays)?);
    }
    let mut count = count_weekdays_forward(start_key, end_key)?;
    for holiday in holidays {
        if *holiday >= start_key
            && *holiday <= end_key
            && !matches!(
                date_from_key(*holiday)?.weekday(),
                Weekday::Sat | Weekday::Sun
            )
        {
            count -= 1;
        }
    }
    Ok(count)
}

fn first_business_day_key(year: i32, month: u32, holidays: &HashSet<i64>) -> BuiltinResult<i64> {
    let mut date = NaiveDate::from_ymd_opt(year, month, 1)
        .ok_or_else(|| datetime_error("fbusdate: invalid year/month"))?;
    loop {
        let key = key_from_date(date);
        if is_business_day_key(key, holidays)? {
            return Ok(key);
        }
        date += Duration::days(1);
    }
}

fn last_business_day_key(year: i32, month: u32, holidays: &HashSet<i64>) -> BuiltinResult<i64> {
    let mut date = NaiveDate::from_ymd_opt(year, month, days_in_month(year, month)?)
        .ok_or_else(|| datetime_error("lbusdate: invalid year/month"))?;
    loop {
        let key = key_from_date(date);
        if is_business_day_key(key, holidays)? {
            return Ok(key);
        }
        date -= Duration::days(1);
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
    let selector = cell.data[0].clone();
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
    name = "isdatetime",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Return true for datetime values.",
    keywords = "isdatetime,datetime,predicate"
)]
fn isdatetime_builtin(value: Value) -> crate::BuiltinResult<Value> {
    Ok(Value::Bool(is_datetime_object(&value)))
}

#[runmat_macros::runtime_builtin(
    name = "now",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Return the current local date and time as a MATLAB serial date number.",
    keywords = "now,datenum,current time"
)]
fn now_builtin() -> crate::BuiltinResult<Value> {
    Ok(Value::Num(datenum_from_naive(current_naive_local())))
}

#[runmat_macros::runtime_builtin(
    name = "today",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Return the current local date as a datetime scalar.",
    keywords = "today,datetime,current date"
)]
fn today_builtin() -> crate::BuiltinResult<Value> {
    let today = Local::now().date_naive().and_hms_opt(0, 0, 0).unwrap();
    datetime_from_date_only(today, DEFAULT_DATE_FORMAT)
}

#[runmat_macros::runtime_builtin(
    name = "clock",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Return the current local date and time as a date vector.",
    keywords = "clock,datevec,current time"
)]
fn clock_builtin() -> crate::BuiltinResult<Value> {
    let components = datevec_components_from_serial(datenum_from_naive(current_naive_local()))?;
    Ok(Value::Tensor(
        Tensor::new(components.to_vec(), vec![1, 6])
            .map_err(|err| datetime_error(format!("clock: {err}")))?,
    ))
}

#[runmat_macros::runtime_builtin(
    name = "datenum",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Convert date/time inputs to MATLAB serial date numbers.",
    keywords = "datenum,datetime,datevec,serial date"
)]
async fn datenum_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let args = gather_args(&args).await?;
    let tensor = match args.len() {
        0 => Tensor::new(vec![datenum_from_naive(current_naive_local())], vec![1, 1])
            .map_err(|err| datetime_error(format!("datenum: {err}")))?,
        1 => match &args[0] {
            Value::Object(obj) if obj.is_class(DATETIME_CLASS) => serial_tensor_for_object(obj)?,
            Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
                let (serials, shape, _) = parse_text_input(args[0].clone(), None)?;
                Tensor::new(serials, shape)
                    .map_err(|err| datetime_error(format!("datenum: {err}")))?
            }
            Value::Tensor(_) => {
                if let Ok(datevec) = tensor_from_datevec_like(args[0].clone(), "datenum") {
                    datenum_from_datevec_tensor(&datevec, "datenum")?
                } else {
                    tensor_from_numeric(args[0].clone(), "datenum")?
                }
            }
            _ => tensor_from_numeric(args[0].clone(), "datenum")?,
        },
        3..=6 => {
            let datetime = build_from_components(args, None)?;
            serials_from_datetime_value(&datetime)?
        }
        _ => {
            return Err(datetime_error(
                "datenum: expected datetime, text, date vector, or Y/M/D components",
            ))
        }
    };
    if tensor.data.len() == 1 {
        Ok(Value::Num(tensor.data[0]))
    } else {
        Ok(Value::Tensor(tensor))
    }
}

#[runmat_macros::runtime_builtin(
    name = "datevec",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Convert date/time inputs to date vectors.",
    keywords = "datevec,datetime,datenum"
)]
async fn datevec_builtin(value: Value) -> crate::BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| datetime_error(format!("datevec: {}", err.message())))?;
    let serials = numeric_or_datetime_serial_tensor(value, "datevec")?;
    let matrix = datevec_matrix_from_serial_tensor(&serials)?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        let mut outputs = Vec::with_capacity(out_count.min(6));
        for col in 0..6.min(out_count) {
            let mut data = Vec::with_capacity(matrix.rows);
            for row in 0..matrix.rows {
                data.push(matrix.data[col * matrix.rows + row]);
            }
            outputs.push(if data.len() == 1 {
                Value::Num(data[0])
            } else {
                Value::Tensor(
                    Tensor::new(data, vec![matrix.rows, 1])
                        .map_err(|err| datetime_error(format!("datevec: {err}")))?,
                )
            });
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count, outputs,
        ));
    }
    Ok(Value::Tensor(matrix))
}

#[runmat_macros::runtime_builtin(
    name = "datestr",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Format date/time inputs as character rows.",
    keywords = "datestr,datetime,datenum,date formatting"
)]
async fn datestr_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| datetime_error(format!("datestr: {}", err.message())))?;
    let rest = gather_args(&rest).await?;
    if rest.len() > 1 {
        return Err(datetime_error(
            "datestr: expected at most one format argument",
        ));
    }
    let format = rest
        .first()
        .map(|value| scalar_text(value, "datestr format"))
        .transpose()?
        .unwrap_or_else(|| DEFAULT_DATETIME_FORMAT.to_string());
    let serials = match &value {
        Value::Tensor(_) => {
            if let Ok(datevec) = tensor_from_datevec_like(value.clone(), "datestr") {
                datenum_from_datevec_tensor(&datevec, "datestr")?
            } else {
                numeric_or_datetime_serial_tensor(value, "datestr")?
            }
        }
        _ => numeric_or_datetime_serial_tensor(value, "datestr")?,
    };
    let mut rows = Vec::with_capacity(serials.data.len());
    for serial in &serials.data {
        rows.push(format_serial(*serial, &format)?);
    }
    Ok(Value::CharArray(char_array_from_rows(&rows, "datestr")?))
}

#[runmat_macros::runtime_builtin(
    name = "weekday",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Return weekday numbers and names for date/time inputs.",
    keywords = "weekday,datetime,datenum"
)]
async fn weekday_builtin(value: Value) -> crate::BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| datetime_error(format!("weekday: {}", err.message())))?;
    let serials = numeric_or_datetime_serial_tensor(value, "weekday")?;
    let mut nums = Vec::with_capacity(serials.data.len());
    let mut names = Vec::with_capacity(serials.data.len());
    for serial in &serials.data {
        let weekday = naive_from_datenum(*serial)?.weekday();
        nums.push(f64::from(weekday.num_days_from_sunday()) + 1.0);
        names.push(
            match weekday {
                Weekday::Sun => "Sunday",
                Weekday::Mon => "Monday",
                Weekday::Tue => "Tuesday",
                Weekday::Wed => "Wednesday",
                Weekday::Thu => "Thursday",
                Weekday::Fri => "Friday",
                Weekday::Sat => "Saturday",
            }
            .to_string(),
        );
    }
    let shape = tensor::default_shape_for(&serials.shape, serials.data.len());
    let num_value = tensor_or_scalar(nums, shape.clone())?;
    let name_value = Value::StringArray(
        StringArray::new(names, shape).map_err(|err| datetime_error(format!("weekday: {err}")))?,
    );
    if let Some(out_count) = crate::output_count::current_output_count() {
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![num_value, name_value],
        ));
    }
    Ok(num_value)
}

#[runmat_macros::runtime_builtin(
    name = "eomday",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Return the last day number for month/year pairs.",
    keywords = "eomday,end of month,calendar"
)]
async fn eomday_builtin(year: Value, month: Value) -> crate::BuiltinResult<Value> {
    let year = gather_if_needed_async(&year)
        .await
        .map_err(|err| datetime_error(format!("eomday: {}", err.message())))?;
    let month = gather_if_needed_async(&month)
        .await
        .map_err(|err| datetime_error(format!("eomday: {}", err.message())))?;
    let years = component_tensor(year, "year")?;
    let months = component_tensor(month, "month")?;
    let (year_data, month_data, shape) =
        tensor::binary_numeric_tensors(&years, &months, "eomday", BUILTIN_NAME)?;
    let mut out = Vec::with_capacity(year_data.len());
    for (year, month) in year_data.iter().zip(month_data.iter()) {
        let year = round_component(*year, "year", -262_000, 262_000)? as i32;
        let month = round_component(*month, "month", 1, 12)? as u32;
        out.push(f64::from(days_in_month(year, month)?));
    }
    tensor_or_scalar(out, shape)
}

#[runmat_macros::runtime_builtin(
    name = "etime",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Return elapsed seconds between date vectors.",
    keywords = "etime,datevec,elapsed time"
)]
async fn etime_builtin(t2: Value, t1: Value) -> crate::BuiltinResult<Value> {
    let t2 = gather_if_needed_async(&t2)
        .await
        .map_err(|err| datetime_error(format!("etime: {}", err.message())))?;
    let t1 = gather_if_needed_async(&t1)
        .await
        .map_err(|err| datetime_error(format!("etime: {}", err.message())))?;
    let t2 = datenum_from_datevec_tensor(&tensor_from_datevec_like(t2, "etime")?, "etime")?;
    let t1 = datenum_from_datevec_tensor(&tensor_from_datevec_like(t1, "etime")?, "etime")?;
    let (left, right, shape) = tensor::binary_numeric_tensors(&t2, &t1, "etime", BUILTIN_NAME)?;
    let out = left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| (a - b) * SECONDS_PER_DAY)
        .collect::<Vec<_>>();
    tensor_or_scalar(out, shape)
}

#[runmat_macros::runtime_builtin(
    name = "isbetween",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Return true where values fall between lower and upper bounds.",
    keywords = "isbetween,datetime,duration,comparison"
)]
async fn isbetween_builtin(
    value: Value,
    lower: Value,
    upper: Value,
) -> crate::BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| datetime_error(format!("isbetween: {}", err.message())))?;
    let lower = gather_if_needed_async(&lower)
        .await
        .map_err(|err| datetime_error(format!("isbetween: {}", err.message())))?;
    let upper = gather_if_needed_async(&upper)
        .await
        .map_err(|err| datetime_error(format!("isbetween: {}", err.message())))?;
    let values = numeric_or_datetime_serial_tensor(value, "isbetween")?;
    let lower = numeric_or_datetime_serial_tensor(lower, "isbetween")?;
    let upper = numeric_or_datetime_serial_tensor(upper, "isbetween")?;
    let (values_data, lower_data, upper_data, shape) =
        broadcast_three_numeric_tensors(&values, &lower, &upper, "isbetween")?;
    let out = values_data
        .iter()
        .zip(lower_data.iter())
        .zip(upper_data.iter())
        .map(|((value, lower), upper)| {
            if value >= lower && value <= upper {
                1.0
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    tensor_or_scalar(out, shape)
}

#[runmat_macros::runtime_builtin(
    name = "calendarDuration",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Create calendar duration values from calendar components.",
    keywords = "calendarDuration,caldays,calmonths,calyears,datetime"
)]
async fn calendar_duration_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let args = gather_args(&args).await?;
    if args.is_empty() {
        return calendar_duration_object_from_components(vec![0.0], vec![0.0], vec![1, 1]);
    }
    if args.len() == 1 && is_calendar_duration_object(&args[0]) {
        return Ok(args[0].clone());
    }

    let labels = ["years", "months", "days", "hours", "minutes", "seconds"];
    let positional = match args.len() {
        1 => {
            let days = component_tensor(args[0].clone(), "calendarDuration")?;
            let shape = tensor::default_shape_for(&days.shape, days.data.len());
            return calendar_duration_object_from_components(
                vec![0.0; days.data.len()],
                days.data,
                shape,
            );
        }
        3..=6 => args,
        _ => {
            return Err(datetime_error(
                "calendarDuration: expected no input, days, or Y/M/D[/H/M/S] components",
            ))
        }
    };

    let mut arrays = Vec::with_capacity(6);
    for (idx, arg) in positional.into_iter().enumerate() {
        arrays.push(component_tensor(arg, labels[idx])?);
    }
    while arrays.len() < 6 {
        arrays.push(Tensor::new(vec![0.0], vec![1, 1]).unwrap());
    }
    let (broadcasted, shape) = broadcast_component_data(&arrays, &labels)?;
    let len = broadcasted[0].len();
    let mut months = Vec::with_capacity(len);
    let mut days = Vec::with_capacity(len);
    for idx in 0..len {
        let month_value = broadcasted[0][idx] * 12.0 + broadcasted[1][idx];
        let day_value = broadcasted[2][idx]
            + broadcasted[3][idx] / 24.0
            + broadcasted[4][idx] / (24.0 * 60.0)
            + broadcasted[5][idx] / SECONDS_PER_DAY;
        if !month_value.is_finite() || !day_value.is_finite() {
            return Err(datetime_error(
                "calendarDuration: resulting calendar duration is outside supported range",
            ));
        }
        months.push(month_value);
        days.push(day_value);
    }
    calendar_duration_object_from_components(months, days, shape)
}

#[runmat_macros::runtime_builtin(
    name = "caldays",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Create calendar durations from days or convert calendar durations to day counts.",
    keywords = "caldays,calendarDuration,datetime"
)]
async fn caldays_builtin(value: Value) -> crate::BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| datetime_error(format!("caldays: {}", err.message())))?;
    calendar_duration_unit_value(value, "caldays", 0.0, 1.0)
}

#[runmat_macros::runtime_builtin(
    name = "calweeks",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Create calendar durations from weeks or convert calendar durations to week counts.",
    keywords = "calweeks,calendarDuration,datetime"
)]
async fn calweeks_builtin(value: Value) -> crate::BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| datetime_error(format!("calweeks: {}", err.message())))?;
    calendar_duration_unit_value(value, "calweeks", 0.0, 7.0)
}

#[runmat_macros::runtime_builtin(
    name = "calmonths",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Create calendar durations from months or convert calendar durations to month counts.",
    keywords = "calmonths,calendarDuration,datetime"
)]
async fn calmonths_builtin(value: Value) -> crate::BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| datetime_error(format!("calmonths: {}", err.message())))?;
    calendar_duration_unit_value(value, "calmonths", 1.0, 0.0)
}

#[runmat_macros::runtime_builtin(
    name = "calquarters",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Create calendar durations from quarters or convert calendar durations to quarter counts.",
    keywords = "calquarters,calendarDuration,datetime"
)]
async fn calquarters_builtin(value: Value) -> crate::BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| datetime_error(format!("calquarters: {}", err.message())))?;
    calendar_duration_unit_value(value, "calquarters", 3.0, 0.0)
}

#[runmat_macros::runtime_builtin(
    name = "calyears",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Create calendar durations from years or convert calendar durations to year counts.",
    keywords = "calyears,calendarDuration,datetime"
)]
async fn calyears_builtin(value: Value) -> crate::BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| datetime_error(format!("calyears: {}", err.message())))?;
    calendar_duration_unit_value(value, "calyears", 12.0, 0.0)
}

#[runmat_macros::runtime_builtin(
    name = "iscalendarduration",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Return true for calendarDuration values.",
    keywords = "iscalendarduration,calendarDuration,predicate"
)]
fn iscalendarduration_builtin(value: Value) -> crate::BuiltinResult<Value> {
    Ok(Value::Bool(is_calendar_duration_object(&value)))
}

#[runmat_macros::runtime_builtin(
    name = "isbusday",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Return true where date values are business days.",
    keywords = "isbusday,business day,datetime,financial"
)]
async fn isbusday_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| datetime_error(format!("isbusday: {}", err.message())))?;
    let rest = gather_args(&rest).await?;
    if rest.len() > 1 {
        return Err(datetime_error(
            "isbusday: expected at most one holiday list",
        ));
    }
    let serials = numeric_or_datetime_serial_tensor(value, "isbusday")?;
    let (start_key, end_key) = date_key_range(&[&serials])?;
    let holidays = holiday_set_from_optional_or_default(
        rest.into_iter().next(),
        "isbusday",
        start_key,
        end_key,
    )?;
    let mut out = Vec::with_capacity(serials.data.len());
    for serial in &serials.data {
        out.push(
            if is_business_day_key(serial_date_key(*serial)?, &holidays)? {
                1.0
            } else {
                0.0
            },
        );
    }
    tensor_or_scalar(
        out,
        tensor::default_shape_for(&serials.shape, serials.data.len()),
    )
}

#[runmat_macros::runtime_builtin(
    name = "holidays",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Return exchange-style market holidays for a year or date range.",
    keywords = "holidays,business day,datetime,financial"
)]
async fn holidays_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let args = gather_args(&args).await?;
    let keys = match args.len() {
        0 => {
            let year = current_naive_local().year();
            let start = key_from_date(NaiveDate::from_ymd_opt(year, 1, 1).unwrap());
            let end = key_from_date(NaiveDate::from_ymd_opt(year, 12, 31).unwrap());
            holiday_keys_between(start, end)?
        }
        1 => {
            let tensor = tensor_from_numeric(args[0].clone(), "holidays");
            if let Ok(tensor) = tensor {
                if tensor.data.len() == 1 && (1000.0..=9999.0).contains(&tensor.data[0]) {
                    let keys = market_holiday_keys_for_year(tensor.data[0].round() as i32)?;
                    let len = keys.len();
                    return datetime_object_from_serials(
                        keys.into_iter().map(|key| key as f64).collect(),
                        vec![len, 1],
                        DEFAULT_DATE_FORMAT,
                    );
                }
            }
            let serials = numeric_or_datetime_serial_tensor(args[0].clone(), "holidays")?;
            let year = date_from_key(serial_date_key(serials.data[0])?)?.year();
            let start = key_from_date(NaiveDate::from_ymd_opt(year, 1, 1).unwrap());
            let end = key_from_date(NaiveDate::from_ymd_opt(year, 12, 31).unwrap());
            holiday_keys_between(start, end)?
        }
        2 => {
            let start = numeric_or_datetime_serial_tensor(args[0].clone(), "holidays")?;
            let end = numeric_or_datetime_serial_tensor(args[1].clone(), "holidays")?;
            if start.data.len() != 1 || end.data.len() != 1 {
                return Err(datetime_error(
                    "holidays: start and end dates must be scalar",
                ));
            }
            holiday_keys_between(
                serial_date_key(start.data[0])?,
                serial_date_key(end.data[0])?,
            )?
        }
        _ => {
            return Err(datetime_error(
                "holidays: expected zero, one, or two inputs",
            ))
        }
    };
    let len = keys.len();
    datetime_object_from_serials(
        keys.into_iter().map(|key| key as f64).collect(),
        vec![len, 1],
        DEFAULT_DATE_FORMAT,
    )
}

#[runmat_macros::runtime_builtin(
    name = "busdays",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Return serial date numbers for business days in a scalar date range.",
    keywords = "busdays,business day,datetime,financial"
)]
async fn busdays_builtin(
    start: Value,
    end: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let start = gather_if_needed_async(&start)
        .await
        .map_err(|err| datetime_error(format!("busdays: {}", err.message())))?;
    let end = gather_if_needed_async(&end)
        .await
        .map_err(|err| datetime_error(format!("busdays: {}", err.message())))?;
    let rest = gather_args(&rest).await?;
    if rest.len() > 1 {
        return Err(datetime_error("busdays: expected at most one holiday list"));
    }
    let start = numeric_or_datetime_serial_tensor(start, "busdays")?;
    let end = numeric_or_datetime_serial_tensor(end, "busdays")?;
    if start.data.len() != 1 || end.data.len() != 1 {
        return Err(datetime_error(
            "busdays: start and end dates must be scalar",
        ));
    }
    let mut key = serial_date_key(start.data[0])?;
    let end_key = serial_date_key(end.data[0])?;
    let span = key
        .max(end_key)
        .checked_sub(key.min(end_key))
        .and_then(|delta| delta.checked_add(1))
        .ok_or_else(|| datetime_error("busdays: date range is outside supported range"))?;
    if span > MAX_BUSDAYS_OUTPUT_LEN {
        return Err(datetime_error(format!(
            "busdays: output would exceed {MAX_BUSDAYS_OUTPUT_LEN} dates"
        )));
    }
    let holidays =
        holiday_set_from_optional_or_default(rest.into_iter().next(), "busdays", key, end_key)?;
    let step = if key <= end_key { 1 } else { -1 };
    let mut out = Vec::new();
    loop {
        if is_business_day_key(key, &holidays)? {
            out.push(key as f64);
        }
        if key == end_key {
            break;
        }
        key = key
            .checked_add(step)
            .ok_or_else(|| datetime_error("busdays: date range is outside supported range"))?;
    }
    let len = out.len();
    Tensor::new(out, vec![len, 1])
        .map(Value::Tensor)
        .map_err(|err| datetime_error(format!("busdays: {err}")))
}

#[runmat_macros::runtime_builtin(
    name = "days252bus",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Count business days between date values using a 252-business-day calendar.",
    keywords = "days252bus,business day,datetime,financial"
)]
async fn days252bus_builtin(
    start: Value,
    end: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let start = gather_if_needed_async(&start)
        .await
        .map_err(|err| datetime_error(format!("days252bus: {}", err.message())))?;
    let end = gather_if_needed_async(&end)
        .await
        .map_err(|err| datetime_error(format!("days252bus: {}", err.message())))?;
    let rest = gather_args(&rest).await?;
    if rest.len() > 1 {
        return Err(datetime_error(
            "days252bus: expected at most one holiday list",
        ));
    }
    let starts = numeric_or_datetime_serial_tensor(start, "days252bus")?;
    let ends = numeric_or_datetime_serial_tensor(end, "days252bus")?;
    let (start_key, end_key) = date_key_range(&[&starts, &ends])?;
    let holidays = holiday_set_from_optional_or_default(
        rest.into_iter().next(),
        "days252bus",
        start_key,
        end_key,
    )?;
    let (start_data, end_data, shape) =
        tensor::binary_numeric_tensors(&starts, &ends, "days252bus", BUILTIN_NAME)?;
    let counts = start_data
        .iter()
        .zip(end_data.iter())
        .map(|(start, end)| {
            Ok(
                count_business_days(serial_date_key(*start)?, serial_date_key(*end)?, &holidays)?
                    as f64,
            )
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    tensor_or_scalar(counts, shape)
}

#[runmat_macros::runtime_builtin(
    name = "daysdif",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Return date differences using actual or 30/360 day-count bases.",
    keywords = "daysdif,date difference,datetime,financial"
)]
async fn daysdif_builtin(
    start: Value,
    end: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let start = gather_if_needed_async(&start)
        .await
        .map_err(|err| datetime_error(format!("daysdif: {}", err.message())))?;
    let end = gather_if_needed_async(&end)
        .await
        .map_err(|err| datetime_error(format!("daysdif: {}", err.message())))?;
    let rest = gather_args(&rest).await?;
    if rest.len() > 1 {
        return Err(datetime_error(
            "daysdif: expected at most one basis argument",
        ));
    }
    let basis = rest
        .first()
        .map(|value| tensor_from_numeric(value.clone(), "daysdif"))
        .transpose()?
        .and_then(|tensor| tensor.data.first().copied())
        .unwrap_or(0.0)
        .round() as i64;
    let starts = numeric_or_datetime_serial_tensor(start, "daysdif")?;
    let ends = numeric_or_datetime_serial_tensor(end, "daysdif")?;
    let (start_data, end_data, shape) =
        tensor::binary_numeric_tensors(&starts, &ends, "daysdif", BUILTIN_NAME)?;
    let out = start_data
        .iter()
        .zip(end_data.iter())
        .map(|(start, end)| {
            let start_key = serial_date_key(*start)?;
            let end_key = serial_date_key(*end)?;
            if basis == 1 {
                let s = date_from_key(start_key)?;
                let e = date_from_key(end_key)?;
                let sd = s.day().min(30) as i32;
                let ed = if sd == 30 { e.day().min(30) } else { e.day() } as i32;
                Ok(f64::from(
                    (e.year() - s.year()) * 360
                        + (e.month() as i32 - s.month() as i32) * 30
                        + (ed - sd),
                ))
            } else {
                Ok((end_key - start_key) as f64)
            }
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    tensor_or_scalar(out, shape)
}

#[runmat_macros::runtime_builtin(
    name = "fbusdate",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Return first business day serial date numbers for month/year pairs.",
    keywords = "fbusdate,business day,datetime,financial"
)]
async fn fbusdate_builtin(
    year: Value,
    month: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let year = gather_if_needed_async(&year)
        .await
        .map_err(|err| datetime_error(format!("fbusdate: {}", err.message())))?;
    let month = gather_if_needed_async(&month)
        .await
        .map_err(|err| datetime_error(format!("fbusdate: {}", err.message())))?;
    let rest = gather_args(&rest).await?;
    if rest.len() > 1 {
        return Err(datetime_error(
            "fbusdate: expected at most one holiday list",
        ));
    }
    let years = component_tensor(year, "fbusdate year")?;
    let months = component_tensor(month, "fbusdate month")?;
    let (year_data, month_data, shape) =
        tensor::binary_numeric_tensors(&years, &months, "fbusdate", BUILTIN_NAME)?;
    let mut min_key = i64::MAX;
    let mut max_key = i64::MIN;
    for (year, month) in year_data.iter().zip(month_data.iter()) {
        let year = round_component(*year, "year", -262_000, 262_000)? as i32;
        let month = round_component(*month, "month", 1, 12)? as u32;
        min_key = min_key.min(key_from_date(
            NaiveDate::from_ymd_opt(year, month, 1)
                .ok_or_else(|| datetime_error("fbusdate: invalid year/month"))?,
        ));
        max_key = max_key.max(key_from_date(
            NaiveDate::from_ymd_opt(year, month, days_in_month(year, month)?)
                .ok_or_else(|| datetime_error("fbusdate: invalid year/month"))?,
        ));
    }
    let holidays = holiday_set_from_optional_or_default(
        rest.into_iter().next(),
        "fbusdate",
        min_key,
        max_key,
    )?;
    let out = year_data
        .iter()
        .zip(month_data.iter())
        .map(|(year, month)| {
            Ok(first_business_day_key(
                round_component(*year, "year", -262_000, 262_000)? as i32,
                round_component(*month, "month", 1, 12)? as u32,
                &holidays,
            )? as f64)
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    tensor_or_scalar(out, shape)
}

#[runmat_macros::runtime_builtin(
    name = "lbusdate",
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Return last business day serial date numbers for month/year pairs.",
    keywords = "lbusdate,business day,datetime,financial"
)]
async fn lbusdate_builtin(
    year: Value,
    month: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let year = gather_if_needed_async(&year)
        .await
        .map_err(|err| datetime_error(format!("lbusdate: {}", err.message())))?;
    let month = gather_if_needed_async(&month)
        .await
        .map_err(|err| datetime_error(format!("lbusdate: {}", err.message())))?;
    let rest = gather_args(&rest).await?;
    if rest.len() > 1 {
        return Err(datetime_error(
            "lbusdate: expected at most one holiday list",
        ));
    }
    let years = component_tensor(year, "lbusdate year")?;
    let months = component_tensor(month, "lbusdate month")?;
    let (year_data, month_data, shape) =
        tensor::binary_numeric_tensors(&years, &months, "lbusdate", BUILTIN_NAME)?;
    let mut min_key = i64::MAX;
    let mut max_key = i64::MIN;
    for (year, month) in year_data.iter().zip(month_data.iter()) {
        let year = round_component(*year, "year", -262_000, 262_000)? as i32;
        let month = round_component(*month, "month", 1, 12)? as u32;
        min_key = min_key.min(key_from_date(
            NaiveDate::from_ymd_opt(year, month, 1)
                .ok_or_else(|| datetime_error("lbusdate: invalid year/month"))?,
        ));
        max_key = max_key.max(key_from_date(
            NaiveDate::from_ymd_opt(year, month, days_in_month(year, month)?)
                .ok_or_else(|| datetime_error("lbusdate: invalid year/month"))?,
        ));
    }
    let holidays = holiday_set_from_optional_or_default(
        rest.into_iter().next(),
        "lbusdate",
        min_key,
        max_key,
    )?;
    let out = year_data
        .iter()
        .zip(month_data.iter())
        .map(|(year, month)| {
            Ok(last_business_day_key(
                round_component(*year, "year", -262_000, 262_000)? as i32,
                round_component(*month, "month", 1, 12)? as u32,
                &holidays,
            )? as f64)
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    tensor_or_scalar(out, shape)
}

#[runmat_macros::runtime_builtin(
    name = "datetick",
    builtin_path = "crate::builtins::datetime",
    category = "plotting",
    summary = "Accept MATLAB date-axis formatting calls for compatibility.",
    keywords = "datetick,plot,date axis"
)]
async fn datetick_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let _args = gather_args(&args).await?;
    Ok(Value::Num(0.0))
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
    if is_calendar_duration_object(&rhs) {
        let (months, days) = calendar_duration_tensors_from_value(&rhs)?;
        let (serials, shape) =
            apply_calendar_duration_to_serials(&lhs_serials, &months, &days, 1.0, "plus")?;
        return datetime_object_from_serials(serials, shape, datetime_format_from_value(&lhs));
    }
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
        _ if is_calendar_duration_object(&rhs) => {
            let (months, days) = calendar_duration_tensors_from_value(&rhs)?;
            let (serials, shape) =
                apply_calendar_duration_to_serials(&lhs_serials, &months, &days, -1.0, "minus")?;
            datetime_object_from_serials(serials, shape, datetime_format_from_value(&lhs))
        }
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

fn combine_calendar_durations(
    lhs: &Value,
    rhs: &Value,
    sign: f64,
    context: &str,
) -> BuiltinResult<Value> {
    let (lhs_months, lhs_days) = calendar_duration_tensors_from_value(lhs)?;
    let (rhs_months, rhs_days) = calendar_duration_tensors_from_value(rhs)?;
    let (left_months, right_months, shape) =
        tensor::binary_numeric_tensors(&lhs_months, &rhs_months, context, BUILTIN_NAME)?;
    let lhs_days_shape = tensor::default_shape_for(&lhs_days.shape, lhs_days.data.len());
    let rhs_days_shape = tensor::default_shape_for(&rhs_days.shape, rhs_days.data.len());
    let lhs_day_tensor = Tensor::new(lhs_days.data, lhs_days_shape)
        .map_err(|err| datetime_error(format!("{context}: {err}")))?;
    let rhs_day_tensor = Tensor::new(rhs_days.data, rhs_days_shape)
        .map_err(|err| datetime_error(format!("{context}: {err}")))?;
    let (left_days, right_days, day_shape) =
        tensor::binary_numeric_tensors(&lhs_day_tensor, &rhs_day_tensor, context, BUILTIN_NAME)?;
    if day_shape != shape {
        return Err(datetime_error(format!(
            "{context}: calendarDuration operands must have matching component sizes"
        )));
    }
    let months = left_months
        .iter()
        .zip(right_months.iter())
        .map(|(left, right)| left + sign * right)
        .collect::<Vec<_>>();
    let days = left_days
        .iter()
        .zip(right_days.iter())
        .map(|(left, right)| left + sign * right)
        .collect::<Vec<_>>();
    calendar_duration_object_from_components(months, days, shape)
}

#[runmat_macros::runtime_builtin(
    name = "calendarDuration.plus",
    builtin_path = "crate::builtins::datetime"
)]
async fn calendar_duration_plus(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    if is_datetime_object(&rhs) {
        let (months, days) = calendar_duration_tensors_from_value(&lhs)?;
        let rhs_serials = serials_from_datetime_value(&rhs)?;
        let (serials, shape) =
            apply_calendar_duration_to_serials(&rhs_serials, &months, &days, 1.0, "plus")?;
        return datetime_object_from_serials(serials, shape, datetime_format_from_value(&rhs));
    }
    combine_calendar_durations(&lhs, &rhs, 1.0, "plus")
}

#[runmat_macros::runtime_builtin(
    name = "calendarDuration.minus",
    builtin_path = "crate::builtins::datetime"
)]
async fn calendar_duration_minus(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    combine_calendar_durations(&lhs, &rhs, -1.0, "minus")
}

#[runmat_macros::runtime_builtin(
    name = "calendarDuration.eq",
    builtin_path = "crate::builtins::datetime"
)]
async fn calendar_duration_eq(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let (lhs_months, lhs_days) = calendar_duration_tensors_from_value(&lhs)?;
    let (rhs_months, rhs_days) = calendar_duration_tensors_from_value(&rhs)?;
    let (left_months, right_months, shape) =
        tensor::binary_numeric_tensors(&lhs_months, &rhs_months, "eq", BUILTIN_NAME)?;
    let (left_days, right_days, day_shape) =
        tensor::binary_numeric_tensors(&lhs_days, &rhs_days, "eq", BUILTIN_NAME)?;
    if day_shape != shape {
        return Err(datetime_error(
            "eq: calendarDuration operands must have matching component sizes",
        ));
    }
    let out = left_months
        .iter()
        .zip(right_months.iter())
        .zip(left_days.iter().zip(right_days.iter()))
        .map(|((lm, rm), (ld, rd))| {
            if (lm - rm).abs() <= 1e-12 && (ld - rd).abs() <= 1e-12 {
                1.0
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    tensor_or_scalar(out, shape)
}

#[runmat_macros::runtime_builtin(
    name = "calendarDuration.ne",
    builtin_path = "crate::builtins::datetime"
)]
async fn calendar_duration_ne(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let eq = calendar_duration_eq(lhs, rhs).await?;
    match eq {
        Value::Num(value) => Ok(Value::Num(if value == 0.0 { 1.0 } else { 0.0 })),
        Value::Tensor(tensor) => Ok(Value::Tensor(
            Tensor::new(
                tensor
                    .data
                    .into_iter()
                    .map(|value| if value == 0.0 { 1.0 } else { 0.0 })
                    .collect(),
                tensor.shape,
            )
            .map_err(|err| datetime_error(format!("ne: {err}")))?,
        )),
        other => Ok(other),
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

    fn serial_for_date(year: i32, month: u32, day: u32) -> f64 {
        datenum_from_naive(midnight(NaiveDate::from_ymd_opt(year, month, day).unwrap()))
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
    fn datetime_missing_serial_renders_as_nat() {
        let value = datetime_object_from_serial_tensor(
            Tensor::new(vec![f64::NAN], vec![1, 1]).unwrap(),
            DEFAULT_DATETIME_FORMAT,
        )
        .expect("datetime object");
        let rendered = datetime_string_array(&value)
            .expect("string array")
            .expect("datetime strings");
        assert_eq!(rendered.data, vec!["NaT".to_string()]);
        assert_eq!(
            datetime_display_text(&value).expect("display"),
            Some("NaT".to_string())
        );
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

    #[test]
    fn legacy_date_conversion_and_query_helpers_work() {
        let serial = serial_for_date(2024, 3, 14);
        let date_vector =
            futures::executor::block_on(datevec_builtin(Value::Num(serial))).expect("datevec");
        let Value::Tensor(date_vector) = date_vector else {
            panic!("expected datevec tensor");
        };
        assert_eq!(date_vector.shape, vec![1, 6]);
        assert_eq!(&date_vector.data[..3], &[2024.0, 3.0, 14.0]);

        let round_trip =
            futures::executor::block_on(datenum_builtin(vec![Value::Tensor(date_vector.clone())]))
                .expect("datenum");
        assert_eq!(round_trip, Value::Num(serial));
        let date_only_round_trip =
            futures::executor::block_on(datenum_builtin(vec![Value::Tensor(
                Tensor::new(vec![2024.0, 3.0, 14.0], vec![1, 3]).unwrap(),
            )]))
            .expect("datenum date vector");
        assert_eq!(date_only_round_trip, Value::Num(serial));

        let text = futures::executor::block_on(datestr_builtin(
            Value::Num(serial),
            vec![Value::from("yyyy-MM-dd")],
        ))
        .expect("datestr");
        let Value::CharArray(text) = text else {
            panic!("expected datestr char array");
        };
        assert_eq!(text.data.iter().collect::<String>(), "2024-03-14");
        let text_from_datevec = futures::executor::block_on(datestr_builtin(
            Value::Tensor(Tensor::new(vec![2024.0, 3.0, 14.0], vec![1, 3]).unwrap()),
            vec![Value::from("yyyy-MM-dd")],
        ))
        .expect("datestr date vector");
        let Value::CharArray(text_from_datevec) = text_from_datevec else {
            panic!("expected datestr char array");
        };
        assert_eq!(
            text_from_datevec.data.iter().collect::<String>(),
            "2024-03-14"
        );

        let weekday =
            futures::executor::block_on(weekday_builtin(Value::Num(serial))).expect("weekday");
        assert_eq!(weekday, Value::Num(5.0));
        assert_eq!(
            futures::executor::block_on(eomday_builtin(Value::Num(2024.0), Value::Num(2.0)))
                .expect("eomday"),
            Value::Num(29.0)
        );
        assert_eq!(
            futures::executor::block_on(etime_builtin(
                Value::Tensor(
                    Tensor::new(vec![2024.0, 1.0, 2.0, 0.0, 0.0, 0.0], vec![1, 6]).unwrap(),
                ),
                Value::Tensor(
                    Tensor::new(vec![2024.0, 1.0, 1.0, 0.0, 0.0, 0.0], vec![1, 6]).unwrap(),
                ),
            ))
            .expect("etime"),
            Value::Num(SECONDS_PER_DAY)
        );
        assert_eq!(
            futures::executor::block_on(isbetween_builtin(
                Value::Num(serial),
                Value::Num(serial - 1.0),
                Value::Num(serial + 1.0),
            ))
            .expect("isbetween"),
            Value::Num(1.0)
        );
    }

    #[test]
    fn calendar_duration_helpers_and_datetime_arithmetic_work() {
        let one_month =
            futures::executor::block_on(calmonths_builtin(Value::Num(1.0))).expect("calmonths");
        assert_eq!(
            iscalendarduration_builtin(one_month.clone()).expect("predicate"),
            Value::Bool(true)
        );
        assert_eq!(
            futures::executor::block_on(calmonths_builtin(one_month.clone()))
                .expect("calmonths convert"),
            Value::Num(1.0)
        );

        let jan31 = run_datetime(vec![
            Value::from("2024-01-31"),
            Value::from("Format"),
            Value::from("yyyy-MM-dd"),
        ]);
        let shifted = futures::executor::block_on(datetime_plus(jan31, one_month)).expect("plus");
        let rendered = datetime_string_array(&shifted)
            .expect("string array")
            .expect("datetime strings");
        assert_eq!(rendered.data, vec!["2024-02-29".to_string()]);

        let duration = futures::executor::block_on(calendar_duration_builtin(vec![
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(3.0),
        ]))
        .expect("calendarDuration");
        let (months, days) = calendar_duration_tensors_from_value(&duration).expect("components");
        assert_eq!(months.data, vec![14.0]);
        assert_eq!(days.data, vec![3.0]);

        assert!(futures::executor::block_on(calyears_builtin(Value::Num(f64::MAX))).is_err());
        assert!(futures::executor::block_on(calendar_duration_builtin(vec![
            Value::Num(f64::MAX),
            Value::Num(f64::MAX),
            Value::Num(0.0),
        ]))
        .is_err());
    }

    #[test]
    fn business_day_helpers_use_weekends_and_holidays() {
        let new_year = serial_for_date(2024, 1, 1);
        let friday = serial_for_date(2024, 1, 5);
        let saturday = serial_for_date(2024, 1, 6);
        let mask = futures::executor::block_on(isbusday_builtin(
            Value::Tensor(Tensor::new(vec![new_year, friday, saturday], vec![1, 3]).unwrap()),
            Vec::new(),
        ))
        .expect("isbusday");
        let Value::Tensor(mask) = mask else {
            panic!("expected isbusday tensor");
        };
        assert_eq!(mask.data, vec![0.0, 1.0, 0.0]);

        assert_eq!(
            futures::executor::block_on(isbusday_builtin(
                Value::Num(friday),
                vec![Value::Num(friday)],
            ))
            .expect("custom holiday"),
            Value::Num(0.0)
        );

        let business_days = futures::executor::block_on(busdays_builtin(
            Value::Num(friday),
            Value::Num(friday + 3.0),
            Vec::new(),
        ))
        .expect("busdays");
        let Value::Tensor(business_days) = business_days else {
            panic!("expected busdays tensor");
        };
        assert_eq!(business_days.data, vec![friday, friday + 3.0]);

        assert_eq!(
            futures::executor::block_on(days252bus_builtin(
                Value::Num(friday),
                Value::Num(friday + 3.0),
                Vec::new(),
            ))
            .expect("days252bus"),
            Value::Num(2.0)
        );
        assert_eq!(
            futures::executor::block_on(daysdif_builtin(
                Value::Num(friday),
                Value::Num(friday + 3.0),
                Vec::new(),
            ))
            .expect("daysdif"),
            Value::Num(3.0)
        );
        assert_eq!(
            futures::executor::block_on(fbusdate_builtin(
                Value::Num(2024.0),
                Value::Num(1.0),
                Vec::new(),
            ))
            .expect("fbusdate"),
            Value::Num(serial_for_date(2024, 1, 2))
        );
        assert_eq!(
            futures::executor::block_on(lbusdate_builtin(
                Value::Num(2024.0),
                Value::Num(6.0),
                Vec::new(),
            ))
            .expect("lbusdate"),
            Value::Num(serial_for_date(2024, 6, 28))
        );

        let holidays = futures::executor::block_on(holidays_builtin(vec![Value::Num(2024.0)]))
            .expect("holidays");
        let serials = serials_from_datetime_value(&holidays).expect("holiday serials");
        assert!(serials.data.contains(&serial_for_date(2024, 1, 1)));
    }
}
