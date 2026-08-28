use runmat_builtins::{BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind};
use runmat_types::MemberAccess;
use std::collections::{HashMap, HashSet};

use chrono::{DateTime, Datelike, Duration, Local, NaiveDate, NaiveDateTime, Timelike, Weekday};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_value::{CharArray, ObjectInstance, StringArray, Tensor, Value};

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
// This exceeds the number of any supported target weekdays across Chrono's
// complete NaiveDate range, while keeping the O(1) whole-week offset safely
// representable as a TimeDelta. Larger controls cannot produce a valid date.
const MAX_DATESHIFT_DAY_OCCURRENCE: u64 = 200_000_000;

const DATETIME_RAW_DATENUM_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "datetime-implicit-datenum",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "A one-argument numeric value that is not an m-by-3 or m-by-6 date vector is interpreted as a serial date number only in RunMat compatibility-extension mode",
    error_identifier: Some("RunMat:compatibility:DatetimeImplicitDatenumExtension"),
};
const DATETIME_LEGACY_COMPONENT_ARITY_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "datetime-four-five-components",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "Four- and five-component datetime constructor forms are retained only in RunMat compatibility-extension mode",
        error_identifier: Some("RunMat:compatibility:DatetimeLegacyComponentArityExtension"),
    };
const DATETIME_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "datetime-logical-numeric-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "Logical values in datetime numeric positions are a RunMat-only compatibility extension",
    error_identifier: Some("RunMat:compatibility:DatetimeLogicalInputExtension"),
};
const DATETIME_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "datetime-resident-numeric-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "Gathering resident numeric input into a host datetime object is a RunMat-only extension",
    error_identifier: Some("RunMat:compatibility:DatetimeGpuInputExtension"),
};
const HOUR_TYPED_LEGACY_NUMERIC_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "hour-typed-legacy-serial-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "hour with single-precision or typed-integer legacy serial-date input is a RunMat extension because the public legacy documentation does not enumerate those storage classes",
        error_identifier: Some("RunMat:compatibility:HourTypedLegacySerialExtension"),
    };
const YEAR_TYPED_LEGACY_NUMERIC_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "year-typed-legacy-serial-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "year with single-precision or typed-integer legacy serial-date input is a RunMat extension because the public legacy documentation does not enumerate those storage classes",
        error_identifier: Some("RunMat:compatibility:YearTypedLegacySerialExtension"),
    };
const MINUTE_TYPED_LEGACY_NUMERIC_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "minute-typed-legacy-serial-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "minute with single-precision or typed-integer legacy serial-date input is a RunMat extension because the public legacy documentation does not enumerate those storage classes",
        error_identifier: Some("RunMat:compatibility:MinuteTypedLegacySerialExtension"),
    };
const MONTH_TYPED_LEGACY_NUMERIC_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "month-typed-legacy-serial-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "month with single-precision or typed-integer legacy serial-date input is a RunMat extension because the public legacy documentation does not enumerate those storage classes",
        error_identifier: Some("RunMat:compatibility:MonthTypedLegacySerialExtension"),
    };
pub const DATETIME_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    DATETIME_RAW_DATENUM_EXTENSION,
    DATETIME_LEGACY_COMPONENT_ARITY_EXTENSION,
    DATETIME_LOGICAL_INPUT_EXTENSION,
    DATETIME_GPU_INPUT_EXTENSION,
];
pub const DAY_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    DATETIME_LOGICAL_INPUT_EXTENSION,
    DATETIME_GPU_INPUT_EXTENSION,
];
pub const HOUR_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    HOUR_TYPED_LEGACY_NUMERIC_EXTENSION,
    DATETIME_LOGICAL_INPUT_EXTENSION,
    DATETIME_GPU_INPUT_EXTENSION,
];
pub const YEAR_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    YEAR_TYPED_LEGACY_NUMERIC_EXTENSION,
    DATETIME_LOGICAL_INPUT_EXTENSION,
    DATETIME_GPU_INPUT_EXTENSION,
];
pub const MINUTE_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    MINUTE_TYPED_LEGACY_NUMERIC_EXTENSION,
    DATETIME_LOGICAL_INPUT_EXTENSION,
    DATETIME_GPU_INPUT_EXTENSION,
];
pub const MONTH_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    MONTH_TYPED_LEGACY_NUMERIC_EXTENSION,
    DATETIME_LOGICAL_INPUT_EXTENSION,
    DATETIME_GPU_INPUT_EXTENSION,
];
pub const DATESHIFT_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [DATETIME_GPU_INPUT_EXTENSION];

const DATETIME_INTEGER_COMPONENT_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "date vectors and Y/M/D/H/M/S/MS components",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes are read from authoritative storage and validated before conversion at the internal serial-date boundary.",
    }];
const DATETIME_INTEGER_CONVERT_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X with ConvertFrom",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer conversion input is documented; RunMat currently implements datenum only and reports other conversion epochs explicitly.",
    }];
const DATETIME_INTEGER_RESIDENT_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "resident numeric input",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Resident input is gated before provider access and gathered only when RunMat extensions are enabled.",
    }];
pub const DATETIME_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor { form: "datetime(integer_date_vector_or_components)", inputs: &DATETIME_INTEGER_COMPONENT_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Calendar structure is resolved exactly before the representable instant is stored at RunMat's current floating serial-date boundary." },
    BuiltinIntegerCapabilityDescriptor { form: "datetime(integer_X, 'ConvertFrom', dateType)", inputs: &DATETIME_INTEGER_CONVERT_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::Multiple, notes: "The output is a host datetime object. TT2000 and other conversion epochs remain explicit implementation gaps; no false nanosecond-precision claim is made." },
    BuiltinIntegerCapabilityDescriptor { form: "datetime(resident_integer, ...)", inputs: &DATETIME_INTEGER_RESIDENT_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GpuRestricted, overload: BuiltinIntegerOverloadKind::Multiple, notes: "MATLAB-compatible mode rejects resident numeric input before provider lookup; extension mode gathers and returns a host datetime object." },
];

type Broadcast3 = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>);

static DATETIME_CLASS_REGISTERED: crate::class_registry::ClassRegistration =
    crate::class_registry::ClassRegistration::new(DATETIME_CLASS);
static CALENDAR_DURATION_CLASS_REGISTERED: crate::class_registry::ClassRegistration =
    crate::class_registry::ClassRegistration::new(CALENDAR_DURATION_CLASS);

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
        description: "Shift form: 'start', 'end', or 'dayofweek'.",
    },
    BuiltinParamDescriptor {
        name: "unit",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Calendar/time unit.",
    },
    BuiltinParamDescriptor {
        name: "rule",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional current/next/previous/nearest or integer occurrence rule.",
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

const DATETIME_SIGNATURES: [BuiltinSignatureDescriptor; 10] = [
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
        label: "t = datetime(dateVectors)",
        inputs: &[BuiltinParamDescriptor {
            name: "dateVectors",
            ty: BuiltinParamType::NumericArray,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "An m-by-3 or m-by-6 numeric date-vector matrix.",
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
        label: "t = datetime(year, month, day, hour, minute, second, millisecond)",
        inputs: &DATETIME_ARGS_ONLY,
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

const DATETIME_YEAR_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "X = year(t)",
        inputs: &DATETIME_SINGLE_INPUT,
        outputs: &OUT_NUMERIC,
    },
    BuiltinSignatureDescriptor {
        label: "X = year(t, yearTypeOrFormat)",
        inputs: DATETIME_HOUR_SIGNATURES[1].inputs,
        outputs: &OUT_NUMERIC,
    },
];
const DATETIME_MONTH_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "X = month(t)",
        inputs: &DATETIME_SINGLE_INPUT,
        outputs: &OUT_NUMERIC,
    },
    BuiltinSignatureDescriptor {
        label: "X = month(t, monthTypeOrFormat)",
        inputs: DATETIME_HOUR_SIGNATURES[1].inputs,
        outputs: &OUT_ANY,
    },
];
const DATETIME_DAY_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "X = day(t)",
        inputs: &DATETIME_SINGLE_INPUT,
        outputs: &OUT_NUMERIC,
    },
    BuiltinSignatureDescriptor {
        label: "X = day(t, dayType)",
        inputs: &[
            BuiltinParamDescriptor {
                name: "t",
                ty: BuiltinParamType::Any,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Datetime, legacy serial date number, or date text.",
            },
            BuiltinParamDescriptor {
                name: "dayType",
                ty: BuiltinParamType::StringScalar,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "dayofmonth, dayofweek, iso-dayofweek, dayofyear, name, or shortname.",
            },
        ],
        outputs: &OUT_ANY,
    },
];
const DATETIME_HOUR_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "X = hour(t)",
        inputs: &DATETIME_SINGLE_INPUT,
        outputs: &OUT_NUMERIC,
    },
    BuiltinSignatureDescriptor {
        label: "X = hour(t, F)",
        inputs: &[
            BuiltinParamDescriptor {
                name: "t",
                ty: BuiltinParamType::Any,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Legacy serial date number or date text.",
            },
            BuiltinParamDescriptor {
                name: "F",
                ty: BuiltinParamType::StringScalar,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Legacy datestr input format.",
            },
        ],
        outputs: &OUT_NUMERIC,
    },
];
const DATETIME_MINUTE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "X = minute(t)",
        inputs: &DATETIME_SINGLE_INPUT,
        outputs: &OUT_NUMERIC,
    },
    BuiltinSignatureDescriptor {
        label: "X = minute(t, F)",
        inputs: DATETIME_HOUR_SIGNATURES[1].inputs,
        outputs: &OUT_NUMERIC,
    },
];
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
        label: "t2 = dateshift(t, boundary, unit, rule)",
        inputs: &DATESHIFT_INPUTS,
        outputs: &OUT_DATETIME,
    },
    BuiltinSignatureDescriptor {
        label: "t2 = dateshift(t, \"dayofweek\", weekday)",
        inputs: &DATESHIFT_INPUTS,
        outputs: &OUT_DATETIME,
    },
];

const DAY_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "legacy serial date number",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Legacy numeric input is interpreted at the serial-date boundary and day returns double numeric results or cell-character names.",
}];
pub const DAY_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor { form: "day(integer_serial, dayType)", inputs: &DAY_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::Multiple, notes: "All integer classes enter through authoritative storage; numeric outputs are MATLAB-compatible double arrays. Name modes return cell arrays of character vectors." }];

const HOUR_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "legacy serial date number",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "The merged legacy API documents serial date numbers but does not enumerate typed integer storage. RunMat accepts all eight classes only behind the typed-legacy extension and validates exact storage before conversion.",
}];
pub const HOUR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor { form: "hour(integer_serial, F?)", inputs: &HOUR_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Typed integer serials are a clearly gated RunMat extension because the documented primary input is datetime and legacy examples use binary64 serial dates. Exact native storage is range-checked before one serial-date conversion; results are host double values with the input shape." }];

const YEAR_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "legacy serial date number",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "The retained public legacy API documents serial date numbers but does not enumerate typed integer storage.",
}];
pub const YEAR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor { form: "year(integer_serial, F?)", inputs: &YEAR_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Typed integer serials are gated before one checked serial-date conversion and return host double values with the input shape." }];

const MINUTE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "legacy serial date number",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented primary input is datetime and legacy examples use binary64 serial dates. RunMat gates all eight typed integer classes and validates authoritative storage before conversion.",
    }];
pub const MINUTE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor { form: "minute(integer_serial, F?)", inputs: &MINUTE_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Typed integer legacy serials are RunMat-only, cross one checked serial-date boundary, and return host double values with the input shape." }];

pub const SECOND_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "second accepts a datetime value and returns its numeric second component. Native integer host or resident values are not datetime objects and reject without conversion or provider access; legacy serial-date parsing belongs to the separately documented date conversion APIs.",
};

const MONTH_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "legacy serial date number",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented primary input is datetime and legacy examples use binary64 serial dates. RunMat gates all eight typed integer classes and validates authoritative storage before conversion.",
    }];
pub const MONTH_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor { form: "month(integer_serial, F?)", inputs: &MONTH_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Typed integer legacy serials are RunMat-only, cross one checked serial-date boundary, and return host double month numbers with the input shape." }];

const DATESHIFT_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "integer-valued weekday or occurrence rule",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The public API documents integer-valued numeric controls without a per-storage-class table. RunMat's settled compatibility coverage accepts all eight integer classes, reads them exactly before calendar shifting, and also accepts ordinary integer-valued doubles.",
    }];
pub const DATESHIFT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor { form: "dateshift(t, ..., integer_weekday_or_rule)", inputs: &DATESHIFT_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::Multiple, notes: "The documented contract is the integer-valued numeric control form, not an explicit per-class matrix. RunMat's settled all-eight-class coverage is exact, scalar-expands against datetime arrays, and produces host datetime objects." }];

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
        .with_identifier(
            DATETIME_ERROR_INVALID_INPUT
                .identifier
                .expect("datetime invalid-input descriptor identifier"),
        )
        .build()
}

fn ensure_datetime_class_registered() {
    DATETIME_CLASS_REGISTERED.ensure(|| {
        let mut properties = HashMap::new();
        properties.insert(
            FORMAT_FIELD.to_string(),
            crate::class_registry::RuntimeProperty {
                name: FORMAT_FIELD.to_string(),
                is_static: false,
                is_constant: false,
                is_dependent: false,
                get_access: MemberAccess::Public,
                set_access: MemberAccess::Public,
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
                crate::class_registry::RuntimeMethod {
                    name: name.to_string(),
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: MemberAccess::Public,
                    function_name: format!("{DATETIME_CLASS}.{name}"),
                    implicit_class_argument: None,
                },
            );
        }

        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: DATETIME_CLASS.to_string(),
            parent: None,
            properties,
            methods,
        });
    });
}

fn ensure_calendar_duration_class_registered() {
    CALENDAR_DURATION_CLASS_REGISTERED.ensure(|| {
        let mut properties = HashMap::new();
        for name in [CALENDAR_MONTHS_FIELD, CALENDAR_DAYS_FIELD] {
            properties.insert(
                name.to_string(),
                crate::class_registry::RuntimeProperty {
                    name: name.to_string(),
                    is_static: false,
                    is_constant: false,
                    is_dependent: false,
                    get_access: MemberAccess::Public,
                    set_access: MemberAccess::Public,
                    default_value: Some(Value::Num(0.0)),
                },
            );
        }

        let mut methods = HashMap::new();
        for name in ["plus", "minus", "eq", "ne"] {
            methods.insert(
                name.to_string(),
                crate::class_registry::RuntimeMethod {
                    name: name.to_string(),
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: MemberAccess::Public,
                    function_name: format!("{CALENDAR_DURATION_CLASS}.{name}"),
                    implicit_class_argument: None,
                },
            );
        }

        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: CALENDAR_DURATION_CLASS.to_string(),
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
    if matches!(value, Value::Bool(_) | Value::LogicalArray(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DATETIME_LOGICAL_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    tensor::value_into_tensor_for(context, value)
        .map_err(|message| datetime_error(format!("datetime: {message}")))
}

fn validate_authoritative_integer_storage(tensor: &Tensor, context: &str) -> BuiltinResult<()> {
    let Some(storage) = tensor.integer_storage() else {
        return Ok(());
    };
    for value in storage.exact_values() {
        let Some(value) = value.try_to_i64() else {
            return Err(datetime_error(format!(
                "datetime: {context} integer value is outside the supported calendar range"
            )));
        };
        // Chrono's civil calendar is far narrower than i64. Reject from exact
        // storage while the value is still authoritative; every admitted
        // integer is then exactly representable at the serial-date boundary.
        if !(-1_000_000_000..=1_000_000_000).contains(&value) {
            return Err(datetime_error(format!(
                "datetime: {context} integer value is outside the supported calendar range"
            )));
        }
    }
    Ok(())
}

fn validate_authoritative_serial_storage(tensor: &Tensor, context: &str) -> BuiltinResult<()> {
    let Some(storage) = tensor.integer_storage() else {
        return Ok(());
    };
    for value in storage.exact_values() {
        let Some(value) = value.try_to_i64() else {
            return Err(datetime_error(format!(
                "datetime: {context} integer serial is outside the supported serial-date range"
            )));
        };
        // RunMat's current serial-date representation and Chrono-backed civil
        // calendar are narrower than i64. Establish that boundary from exact
        // storage so no wide I64/U64 is rounded into apparent admissibility.
        if !(-1_000_000_000..=1_000_000_000).contains(&value) {
            return Err(datetime_error(format!(
                "datetime: {context} integer serial is outside the supported serial-date range"
            )));
        }
    }
    Ok(())
}

fn serial_tensor_from_value(value: Value, context: &str) -> BuiltinResult<Tensor> {
    let tensor = tensor_from_numeric(value, context)?;
    validate_authoritative_serial_storage(&tensor, context)?;
    let tensor = tensor::integer_tensor_to_f64(tensor)
        .map_err(|err| datetime_error(format!("datetime: {err}")))?;
    let shape = tensor::default_shape_for(&tensor.shape, tensor::tensor_element_len(&tensor));
    let values = tensor::tensor_into_values_f64(tensor);
    Tensor::new(values, shape).map_err(|err| datetime_error(format!("datetime: {err}")))
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
    let total_seconds = (serial - UNIX_DATENUM) * SECONDS_PER_DAY;
    if !total_seconds.is_finite()
        || total_seconds < i64::MIN as f64
        || total_seconds > i64::MAX as f64
    {
        return Err(datetime_error(
            "datetime: serial date number is outside the supported range",
        ));
    }
    let mut seconds = total_seconds.floor() as i64;
    let mut nanos = ((total_seconds - seconds as f64) * 1_000_000_000.0).round() as i64;
    if nanos == 1_000_000_000 {
        seconds = seconds.checked_add(1).ok_or_else(|| {
            datetime_error("datetime: serial date number is outside the supported range")
        })?;
        nanos = 0;
    }
    let base = NaiveDate::from_ymd_opt(1970, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    let duration = Duration::try_seconds(seconds)
        .and_then(|duration| duration.checked_add(&Duration::nanoseconds(nanos)))
        .ok_or_else(|| {
            datetime_error("datetime: serial date number is outside the supported range")
        })?;
    base.checked_add_signed(duration).ok_or_else(|| {
        datetime_error("datetime: serial date number is outside the supported range")
    })
}

fn format_serial(serial: f64, format: &str) -> BuiltinResult<String> {
    if serial.is_nan() {
        return Ok("NaT".to_string());
    }
    if serial.is_infinite() {
        return Ok(if serial.is_sign_positive() {
            "Inf"
        } else {
            "-Inf"
        }
        .to_string());
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
        ("%Y/%m/%d %H:%M:%S%.f", true),
        ("%Y/%m/%d %H:%M:%S", true),
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
            if let Some(relative) = relative_datetime(text.trim()) {
                let has_time = text.trim().eq_ignore_ascii_case("now");
                return Ok((
                    vec![datenum_from_naive(relative)],
                    vec![1, 1],
                    if has_time {
                        DEFAULT_DATETIME_FORMAT
                    } else {
                        DEFAULT_DATE_FORMAT
                    }
                    .to_string(),
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
                let parsed = relative_datetime(text.trim())
                    .map(|value| (value, text.trim().eq_ignore_ascii_case("now")))
                    .or_else(|| parse_datetime_text_with_input_format(text, input_format));
                let (naive, parsed_has_time) = parsed.ok_or_else(|| {
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

fn relative_datetime(text: &str) -> Option<NaiveDateTime> {
    let now = Local::now().naive_local();
    match text.to_ascii_lowercase().as_str() {
        "now" => Some(now),
        "today" => Some(midnight(now.date())),
        "tomorrow" => Some(midnight(now.date() + Duration::days(1))),
        "yesterday" => Some(midnight(now.date() - Duration::days(1))),
        _ => None,
    }
}

fn legacy_date_format_to_strftime(format: &str) -> String {
    let mut out = format.to_string();
    for (source, target) in [
        (".fff", "%.3f"),
        ("yyyy", "%Y"),
        ("mmmm", "%B"),
        ("mmm", "%b"),
        ("HH", "%H"),
        ("hh", "%H"),
        ("MM", "%M"),
        ("ss", "%S"),
        ("mm", "%m"),
        ("dd", "%d"),
    ] {
        out = out.replace(source, target);
    }
    out
}

fn parse_legacy_component_text(
    value: Value,
    input_format: Option<&str>,
    label: &str,
) -> BuiltinResult<(Vec<f64>, Vec<usize>)> {
    let (texts, shape) = match value {
        Value::String(text) => (vec![text], vec![1, 1]),
        Value::StringArray(array) => {
            let shape = tensor::default_shape_for(&array.shape, array.data.len());
            (array.data, shape)
        }
        Value::CharArray(array) => {
            let texts = (0..array.rows)
                .map(|row| {
                    array.data[row * array.cols..(row + 1) * array.cols]
                        .iter()
                        .collect::<String>()
                        .trim_end()
                        .to_string()
                })
                .collect();
            (texts, vec![array.rows, 1])
        }
        _ => {
            return Err(datetime_error(format!(
                "{label}: expected legacy date text"
            )))
        }
    };
    let legacy_format = input_format.map(legacy_date_format_to_strftime);
    let mut serials = Vec::with_capacity(texts.len());
    for text in texts {
        let parsed = if let Some(format) = legacy_format.as_deref() {
            NaiveDateTime::parse_from_str(text.trim(), format)
                .ok()
                .or_else(|| {
                    NaiveDate::parse_from_str(text.trim(), format)
                        .ok()
                        .map(|date| date.and_hms_opt(0, 0, 0).unwrap())
                })
        } else {
            parse_datetime_text(text.trim()).map(|(value, _)| value)
        }
        .ok_or_else(|| {
            datetime_error(format!(
                "{label}: unable to parse legacy date text '{text}'"
            ))
        })?;
        serials.push(datenum_from_naive(parsed));
    }
    Ok((serials, shape))
}

fn parse_legacy_day_text(
    value: Value,
    input_format: Option<&str>,
) -> BuiltinResult<(Vec<f64>, Vec<usize>)> {
    let (texts, shape) = match value {
        Value::String(text) => (vec![text], vec![1, 1]),
        Value::StringArray(array) => {
            let shape = tensor::default_shape_for(&array.shape, array.data.len());
            (array.data, shape)
        }
        Value::CharArray(array) => {
            let mut texts = Vec::with_capacity(array.rows);
            for row in 0..array.rows {
                texts.push(
                    array.data[row * array.cols..(row + 1) * array.cols]
                        .iter()
                        .collect::<String>()
                        .trim_end()
                        .to_string(),
                );
            }
            (texts, vec![array.rows, 1])
        }
        _ => return Err(datetime_error("day: expected legacy date text")),
    };
    let legacy_format = input_format.map(legacy_date_format_to_strftime);
    let mut serials = Vec::with_capacity(texts.len());
    for text in texts {
        let parsed = if let Some(format) = legacy_format.as_deref() {
            NaiveDate::parse_from_str(text.trim(), format)
                .ok()
                .map(|date| date.and_hms_opt(0, 0, 0).unwrap())
        } else {
            parse_datetime_text(text.trim()).map(|(value, _)| value)
        }
        .ok_or_else(|| datetime_error(format!("day: unable to parse legacy date text '{text}'")))?;
        serials.push(datenum_from_naive(parsed));
    }
    Ok((serials, shape))
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
        let len = tensor::tensor_element_len(array);
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
        let len = tensor::tensor_element_len(array);
        if len == 1 {
            broadcasted.push(vec![tensor::tensor_value_f64(array, 0); target_len]);
        } else if len == target_len {
            broadcasted.push(tensor::tensor_values_f64(array));
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
    validate_authoritative_integer_storage(&tensor, context)?;
    let tensor = tensor::integer_tensor_to_f64(tensor)
        .map_err(|err| datetime_error(format!("datetime: {err}")))?;
    let shape = tensor::default_shape_for(&tensor.shape, tensor::tensor_element_len(&tensor));
    let values = tensor::tensor_into_values_f64(tensor);
    Tensor::new(values, shape).map_err(|err| datetime_error(format!("datetime: {err}")))
}

fn build_from_components(args: Vec<Value>, format: Option<String>) -> BuiltinResult<Value> {
    let labels = [
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "millisecond",
    ];
    let input_count = args.len();
    let mut arrays = Vec::with_capacity(args.len());
    for (idx, arg) in args.into_iter().enumerate() {
        arrays.push(component_tensor(arg, labels[idx])?);
    }
    while arrays.len() < 7 {
        arrays.push(Tensor::new(vec![0.0], vec![1, 1]).unwrap());
    }

    let (broadcasted, shape) = broadcast_component_data(&arrays, &labels)?;
    let len = broadcasted[0].len();
    let mut serials = Vec::with_capacity(len);
    for idx in 0..len {
        if input_count == 7 && broadcasted[5][idx].is_finite() && broadcasted[5][idx].fract() != 0.0
        {
            return Err(datetime_error(
                "datetime: second must be integral when a millisecond component is supplied",
            ));
        }
        let second = broadcasted[5][idx] + broadcasted[6][idx] / 1_000.0;
        let serial = serial_from_normalized_components(
            broadcasted[0][idx],
            broadcasted[1][idx],
            broadcasted[2][idx],
            broadcasted[3][idx],
            broadcasted[4][idx],
            second,
        )?;
        serials.push(serial);
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

fn serial_from_normalized_components(
    year: f64,
    month: f64,
    day: f64,
    hour: f64,
    minute: f64,
    second: f64,
) -> BuiltinResult<f64> {
    let components = [year, month, day, hour, minute, second];
    if components.iter().any(|value| value.is_nan()) {
        return Ok(f64::NAN);
    }
    if let Some(infinite) = components.iter().find(|value| value.is_infinite()) {
        return Ok(*infinite);
    }
    let year = round_integral_component(year, "year")?;
    let month = round_integral_component(month, "month")?;
    let day = round_integral_component(day, "day")?;
    let hour = round_integral_component(hour, "hour")?;
    let minute = round_integral_component(minute, "minute")?;
    if !second.is_finite() {
        return Ok(second);
    }
    if year < i64::from(i32::MIN) || year > i64::from(i32::MAX) {
        return Err(datetime_error(
            "datetime: year is outside the supported range",
        ));
    }
    let month_offset = month.checked_sub(1).ok_or_else(|| {
        datetime_error("datetime: calendar components are outside the supported range")
    })?;
    let month_index = year
        .checked_mul(12)
        .and_then(|value| value.checked_add(month_offset))
        .ok_or_else(|| {
            datetime_error("datetime: calendar components are outside the supported range")
        })?;
    let normalized_year = month_index.div_euclid(12);
    let normalized_month = month_index.rem_euclid(12) as u32 + 1;
    let base = NaiveDate::from_ymd_opt(
        i32::try_from(normalized_year)
            .map_err(|_| datetime_error("datetime: year is outside the supported range"))?,
        normalized_month,
        1,
    )
    .ok_or_else(|| datetime_error("datetime: calendar components are outside the supported range"))?
    .and_hms_opt(0, 0, 0)
    .unwrap();
    let whole_seconds = second.trunc();
    let fractional_seconds = second - whole_seconds;
    if whole_seconds < i64::MIN as f64 || whole_seconds > i64::MAX as f64 {
        return Err(datetime_error(
            "datetime: second is outside the supported range",
        ));
    }
    let nanos = (fractional_seconds * 1_000_000_000.0).round();
    if nanos < i64::MIN as f64 || nanos > i64::MAX as f64 {
        return Err(datetime_error(
            "datetime: fractional second is outside the supported range",
        ));
    }
    let day_offset = day.checked_sub(1).ok_or_else(|| {
        datetime_error("datetime: calendar components are outside the supported range")
    })?;
    let normalized = base
        .checked_add_signed(Duration::days(day_offset))
        .and_then(|value| value.checked_add_signed(Duration::hours(hour)))
        .and_then(|value| value.checked_add_signed(Duration::minutes(minute)))
        .and_then(|value| value.checked_add_signed(Duration::seconds(whole_seconds as i64)))
        .and_then(|value| value.checked_add_signed(Duration::nanoseconds(nanos as i64)))
        .ok_or_else(|| {
            datetime_error("datetime: normalized components are outside the supported range")
        })?;
    Ok(datenum_from_naive(normalized))
}

fn round_integral_component(value: f64, label: &str) -> BuiltinResult<i64> {
    let rounded = value.round();
    if (rounded - value).abs() > 1e-9 || rounded < i64::MIN as f64 || rounded > i64::MAX as f64 {
        return Err(datetime_error(format!(
            "datetime: {label} values must be representable integers"
        )));
    }
    Ok(rounded as i64)
}

fn build_from_date_vectors(value: Value, format: Option<String>) -> BuiltinResult<Value> {
    let tensor = tensor_from_numeric(value, "date vector")?;
    validate_authoritative_integer_storage(&tensor, "date vector")?;
    let shape = tensor.shape.clone();
    if shape.len() != 2 || !matches!(shape[1], 3 | 6) {
        return Err(datetime_error(
            "datetime: a numeric one-input date vector must be an m-by-3 or m-by-6 matrix",
        ));
    }
    let rows = shape[0];
    let cols = shape[1];
    let values = tensor::tensor_values_f64_cow(&tensor);
    let mut serials = Vec::with_capacity(rows);
    for row in 0..rows {
        let at = |col: usize, default: f64| {
            if col < cols {
                values[row + col * rows]
            } else {
                default
            }
        };
        serials.push(serial_from_normalized_components(
            at(0, 0.0),
            at(1, 1.0),
            at(2, 1.0),
            at(3, 0.0),
            at(4, 0.0),
            at(5, 0.0),
        )?);
    }
    datetime_object_from_serials(
        serials,
        vec![rows, 1],
        format.unwrap_or_else(|| {
            if cols == 6 {
                DEFAULT_DATETIME_FORMAT
            } else {
                DEFAULT_DATE_FORMAT
            }
            .to_string()
        }),
    )
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

pub(crate) fn calendar_duration_tensors_from_value(
    value: &Value,
) -> BuiltinResult<(Tensor, Tensor)> {
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
    let values = tensor::tensor_values_f64_cow(&numeric);
    let shape = tensor::default_shape_for(&numeric.shape, values.len());
    let mut months = Vec::with_capacity(values.len());
    let mut days = Vec::with_capacity(values.len());
    for value in values.iter().copied() {
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

pub(crate) fn datetime_row_times_from_calendar_step(
    start: &Value,
    step: &Value,
    count: usize,
) -> BuiltinResult<Value> {
    let start_serials = serials_from_datetime_value(start)?;
    if start_serials.len() != 1 {
        return Err(datetime_error(
            "array2timetable: StartTime must be a scalar",
        ));
    }
    let (months, days) = calendar_duration_tensors_from_value(step)?;
    if months.len() != 1 || days.len() != 1 {
        return Err(datetime_error("array2timetable: TimeStep must be a scalar"));
    }
    let month_step = tensor::tensor_value_f64(&months, 0);
    let day_step = tensor::tensor_value_f64(&days, 0);
    if !month_step.is_finite() || !day_step.is_finite() {
        return Err(datetime_error("array2timetable: TimeStep must be finite"));
    }
    let one_month_step = Tensor::new(vec![month_step], vec![1, 1])
        .map_err(|error| datetime_error(format!("array2timetable: {error}")))?;
    let one_day_step = Tensor::new(vec![day_step], vec![1, 1])
        .map_err(|error| datetime_error(format!("array2timetable: {error}")))?;
    let (next_serial, _) = apply_calendar_duration_to_serials(
        &start_serials,
        &one_month_step,
        &one_day_step,
        1.0,
        "array2timetable",
    )?;
    if next_serial[0] <= tensor::tensor_value_f64(&start_serials, 0) {
        return Err(datetime_error("array2timetable: TimeStep must be positive"));
    }
    let mut serials = Vec::with_capacity(count);
    for index in 0..count {
        let factor = index as f64;
        let month_offset = Tensor::new(vec![month_step * factor], vec![1, 1])
            .map_err(|error| datetime_error(format!("array2timetable: {error}")))?;
        let day_offset = Tensor::new(vec![day_step * factor], vec![1, 1])
            .map_err(|error| datetime_error(format!("array2timetable: {error}")))?;
        let (value, _) = apply_calendar_duration_to_serials(
            &start_serials,
            &month_offset,
            &day_offset,
            1.0,
            "array2timetable",
        )?;
        serials.push(value[0]);
    }
    if count > 1 && serials.windows(2).any(|pair| pair[1] <= pair[0]) {
        return Err(datetime_error(
            "array2timetable: TimeStep must produce increasing row times",
        ));
    }
    datetime_object_from_serial_tensor(
        Tensor::new(serials, vec![count, 1])
            .map_err(|error| datetime_error(format!("array2timetable: {error}")))?,
        datetime_format_from_value(start),
    )
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
    let values = tensor::tensor_values_f64_cow(&serials);
    let mut strings = Vec::with_capacity(values.len());
    for serial in values.iter().copied() {
        strings.push(format_serial(serial, &format)?);
    }
    let shape = tensor::default_shape_for(&serials.shape, values.len());
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
    if tensor::tensor_element_len(&serials) == 1 {
        return datetime_display_text(value);
    }
    let shape = tensor::default_shape_for(&serials.shape, tensor::tensor_element_len(&serials));
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
    let values = tensor::tensor_values_f64_cow(&serials);
    let mut out = Vec::with_capacity(values.len());
    for serial in values.iter().copied() {
        let naive = naive_from_datenum(serial)?;
        out.push(extractor(&naive));
    }
    if out.len() == 1 {
        Ok(Value::Num(out[0]))
    } else {
        let shape = tensor::default_shape_for(&serials.shape, values.len());
        let tensor =
            Tensor::new(out, shape).map_err(|err| datetime_error(format!("{label}: {err}")))?;
        Ok(Value::Tensor(tensor))
    }
}

fn component_tensor_from_serials(
    serials: &Tensor,
    label: &str,
    extractor: impl Fn(&NaiveDateTime) -> f64,
) -> BuiltinResult<Value> {
    let values = tensor::tensor_values_f64_cow(serials);
    let mut out = Vec::with_capacity(values.len());
    for serial in values.iter().copied() {
        out.push(if serial.is_finite() {
            extractor(&naive_from_datenum(serial)?)
        } else {
            f64::NAN
        });
    }
    let shape = tensor::default_shape_for(&serials.shape, values.len());
    tensor_or_scalar(out, shape)
        .map_err(|err| datetime_error(format!("{label}: {}", err.message())))
}

fn is_typed_legacy_numeric(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some() || tensor.as_f32_slice().is_some())
}

async fn prepare_legacy_component_input(
    builtin: &'static str,
    value: Value,
    rest: &[Value],
    typed_extension: &'static BuiltinExtensionDescriptor,
) -> BuiltinResult<(Value, Option<String>)> {
    if rest.len() > 1 {
        return Err(datetime_error(format!(
            "{builtin}: expected at most one component type or legacy date format"
        )));
    }
    if matches!(value, Value::GpuTensor(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DATETIME_GPU_INPUT_EXTENSION,
            builtin,
        )?;
    }
    if matches!(value, Value::Bool(_) | Value::LogicalArray(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DATETIME_LOGICAL_INPUT_EXTENSION,
            builtin,
        )?;
    }
    if is_typed_legacy_numeric(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(typed_extension, builtin)?;
    }
    if rest
        .iter()
        .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        return Err(datetime_error(format!(
            "{builtin}: component type or legacy date format must be host text"
        )));
    }

    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| datetime_error(format!("{builtin}: {}", err.message())))?;
    let option = rest
        .first()
        .map(|value| scalar_text(value, &format!("{builtin} component type or date format")))
        .transpose()?;
    Ok((value, option))
}

fn numeric_component_from_modern_or_legacy(
    builtin: &'static str,
    value: Value,
    legacy_format: Option<&str>,
    extractor: impl Fn(&NaiveDateTime) -> f64,
) -> BuiltinResult<Value> {
    if is_datetime_object(&value) {
        if legacy_format.is_some() {
            return Err(datetime_error(format!(
                "{builtin}: legacy date format is not supported for datetime input"
            )));
        }
        return component_tensor_from_datetime(&value, builtin, extractor);
    }
    let serials = match value {
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
            let (serials, shape) = parse_legacy_component_text(value, legacy_format, builtin)?;
            Tensor::new(serials, shape)
                .map_err(|err| datetime_error(format!("{builtin}: {err}")))?
        }
        value => serial_tensor_from_value(value, &format!("{builtin} legacy serial input"))?,
    };
    component_tensor_from_serials(&serials, builtin, extractor)
}

fn month_name(month: u32, short: bool) -> &'static str {
    const FULL: [&str; 12] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    const SHORT: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let index = month.saturating_sub(1) as usize;
    if short {
        SHORT.get(index).copied().unwrap_or("")
    } else {
        FULL.get(index).copied().unwrap_or("")
    }
}

fn month_names_from_datetime(value: &Value, short: bool) -> BuiltinResult<Value> {
    let serials = serials_from_datetime_value(value)?;
    let values = tensor::tensor_values_f64_cow(&serials);
    let shape = tensor::default_shape_for(&serials.shape, values.len());
    let mut out = Vec::with_capacity(values.len());
    for serial in values.iter().copied() {
        let text = if serial.is_finite() {
            month_name(naive_from_datenum(serial)?.month(), short)
        } else {
            ""
        };
        out.push(Value::CharArray(CharArray::new_row(text)));
    }
    runmat_value::CellArray::new_with_shape(out, shape)
        .map(Value::Cell)
        .map_err(|err| datetime_error(format!("month: {err}")))
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
    let values = tensor::tensor_values_f64_cow(serials);
    let rows = values.len();
    let mut data = vec![0.0; rows.saturating_mul(6)];
    for (row, serial) in values.iter().enumerate() {
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
    let tensor = tensor::integer_tensor_to_f64(tensor_from_numeric(value, context)?)
        .map_err(|err| datetime_error(format!("{context}: {err}")))?;
    let shape = tensor::default_shape_for(&tensor.shape, tensor::tensor_element_len(&tensor));
    let values = tensor::tensor_into_values_f64(tensor);
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
    if values.len() == 3 {
        return normalize(1, 3, values);
    }
    if values.len() == 6 {
        return normalize(1, 6, values);
    }
    if shape.len() >= 2 && (shape[1] == 3 || shape[1] == 6) {
        return normalize(shape[0], shape[1], values);
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
    let values = tensor::tensor_values_f64_cow(tensor);
    let mut out = Vec::with_capacity(rows);
    for row in 0..rows {
        let component = |col: usize| values[col * rows + row];
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
        let len = tensor::tensor_element_len(operand);
        if len == 1 {
            continue;
        }
        let shape = tensor::default_shape_for(&operand.shape, len);
        if output_shape.is_empty() {
            output_len = len;
            output_shape = shape;
        } else if len != output_len || shape != output_shape {
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
        match tensor::tensor_element_len(operand) {
            1 => Ok(vec![tensor::tensor_value_f64(operand, 0); output_len]),
            len if len == output_len => Ok(tensor::tensor_values_f64(operand)),
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
        let values = tensor::tensor_values_f64_cow(&serials);
        return values
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
        let values = tensor::tensor_values_f64_cow(tensor);
        for serial in values.iter() {
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
        Value::Int(value) => {
            Tensor::new_integer(runmat_value::IntegerStorage::from_scalar(value), vec![1, 1])
                .map_err(|err| datetime_error(format!("datetime.subsref: {err}")))?
        }
        Value::LogicalArray(logical) => tensor::logical_to_tensor(&logical)
            .map_err(|err| datetime_error(format!("datetime.subsref: {err}")))?,
        other => {
            return Err(datetime_error(format!(
                "datetime.subsref: unsupported index value {other:?}"
            )))
        }
    };
    let indexed = crate::perform_indexing(
        &Value::Tensor(serials),
        &tensor::tensor_values_f64(&selector),
    )
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
    extensions(crate::builtins::datetime::DATETIME_EXTENSIONS),
    integer_capabilities(crate::builtins::datetime::DATETIME_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Create datetime arrays from text, components, or serial date numbers.",
    keywords = "datetime,date,time,datenum,Format",
    related = "year,month,day,hour,minute,second,string,char,disp",
    examples = "t = datetime(2024, 4, 9, 13, 30, 0);"
)]
async fn datetime_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    ensure_datetime_class_registered();
    if args
        .iter()
        .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DATETIME_GPU_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if args
        .iter()
        .any(|value| matches!(value, Value::Bool(_) | Value::LogicalArray(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DATETIME_LOGICAL_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
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
            _ => {
                let numeric = tensor_from_numeric(positional[0].clone(), "date vector")?;
                if numeric.shape.len() == 2 && matches!(numeric.shape[1], 3 | 6) {
                    build_from_date_vectors(positional[0].clone(), options.format)
                } else {
                    crate::compatibility::ensure_builtin_extension_enabled(
                        &DATETIME_RAW_DATENUM_EXTENSION,
                        BUILTIN_NAME,
                    )?;
                    numeric_value_to_datetime(positional[0].clone(), options.format)
                }
            }
        },
        3 | 6 | 7 => build_from_components(positional, options.format),
        4 | 5 => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &DATETIME_LEGACY_COMPONENT_ARITY_EXTENSION,
                BUILTIN_NAME,
            )?;
            build_from_components(positional, options.format)
        }
        _ => Err(datetime_error(
            "datetime: unsupported argument pattern; use text, serial dates, or Y/M/D component inputs",
        )),
    }
}

#[runmat_macros::runtime_builtin(
    name = "year",
    descriptor(crate::builtins::datetime::DATETIME_YEAR_DESCRIPTOR),
    extensions(crate::builtins::datetime::YEAR_EXTENSIONS),
    integer_capabilities(crate::builtins::datetime::YEAR_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Extract calendar year components from datetime values.",
    keywords = "year,datetime,date component"
)]
async fn year_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let (value, option) =
        prepare_legacy_component_input("year", value, &rest, &YEAR_TYPED_LEGACY_NUMERIC_EXTENSION)
            .await?;
    if is_datetime_object(&value) {
        let mode = option
            .as_deref()
            .unwrap_or("iso")
            .trim()
            .to_ascii_lowercase();
        return match mode.as_str() {
            "iso" => component_tensor_from_datetime(&value, "year", |naive| naive.year() as f64),
            "gregorian" => component_tensor_from_datetime(&value, "year", |naive| {
                let year = naive.year();
                if year > 0 {
                    year as f64
                } else {
                    f64::from(1 - year)
                }
            }),
            _ => Err(datetime_error(format!(
                "year: unsupported year type '{mode}'"
            ))),
        };
    }
    numeric_component_from_modern_or_legacy("year", value, option.as_deref(), |naive| {
        naive.year() as f64
    })
}

#[runmat_macros::runtime_builtin(
    name = "month",
    descriptor(crate::builtins::datetime::DATETIME_MONTH_DESCRIPTOR),
    extensions(crate::builtins::datetime::MONTH_EXTENSIONS),
    integer_capabilities(crate::builtins::datetime::MONTH_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Extract month numbers from datetime arrays.",
    keywords = "month,datetime,date component"
)]
async fn month_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let (value, option) = prepare_legacy_component_input(
        "month",
        value,
        &rest,
        &MONTH_TYPED_LEGACY_NUMERIC_EXTENSION,
    )
    .await?;
    if is_datetime_object(&value) {
        let mode = option
            .as_deref()
            .unwrap_or("monthofyear")
            .trim()
            .to_ascii_lowercase();
        return match mode.as_str() {
            "monthofyear" => {
                component_tensor_from_datetime(&value, "month", |naive| naive.month() as f64)
            }
            "name" => month_names_from_datetime(&value, false),
            "shortname" => month_names_from_datetime(&value, true),
            _ => Err(datetime_error(format!(
                "month: unsupported month type '{mode}'"
            ))),
        };
    }
    numeric_component_from_modern_or_legacy("month", value, option.as_deref(), |naive| {
        naive.month() as f64
    })
}

#[runmat_macros::runtime_builtin(
    name = "day",
    descriptor(crate::builtins::datetime::DATETIME_DAY_DESCRIPTOR),
    extensions(crate::builtins::datetime::DAY_EXTENSIONS),
    integer_capabilities(crate::builtins::datetime::DAY_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Extract day numbers or names from datetime and legacy date inputs.",
    keywords = "day,datetime,date component,dayofweek,dayofyear"
)]
async fn day_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if std::iter::once(&value)
        .chain(rest.iter())
        .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DATETIME_GPU_INPUT_EXTENSION,
            "day",
        )?;
    }
    if std::iter::once(&value)
        .chain(rest.iter())
        .any(|value| matches!(value, Value::Bool(_) | Value::LogicalArray(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DATETIME_LOGICAL_INPUT_EXTENSION,
            "day",
        )?;
    }
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| datetime_error(format!("day: {}", err.message())))?;
    let rest = gather_args(&rest).await?;
    if rest.len() > 1 {
        return Err(datetime_error("day: expected at most one day type"));
    }
    let modern_datetime = is_datetime_object(&value);
    let mode = if modern_datetime {
        rest.first()
            .map(|value| scalar_text(value, "day type"))
            .transpose()?
            .unwrap_or_else(|| "dayofmonth".to_string())
            .trim()
            .to_ascii_lowercase()
    } else {
        "dayofmonth".to_string()
    };
    let legacy_input_format = if modern_datetime {
        None
    } else {
        rest.first()
            .map(|value| scalar_text(value, "legacy input format"))
            .transpose()?
    };
    let serials = match &value {
        Value::Object(obj) if obj.is_class(DATETIME_CLASS) => serial_tensor_for_object(obj)?,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
            let (serials, shape) = parse_legacy_day_text(value, legacy_input_format.as_deref())?;
            Tensor::new(serials, shape).map_err(|err| datetime_error(format!("day: {err}")))?
        }
        _ => serial_tensor_from_value(value, "day legacy serial input")?,
    };
    let shape = tensor::default_shape_for(&serials.shape, tensor::tensor_element_len(&serials));
    let values = tensor::tensor_values_f64_cow(&serials);
    if matches!(mode.as_str(), "name" | "shortname") {
        let short = mode == "shortname";
        let mut out = Vec::with_capacity(values.len());
        for serial in values.iter().copied() {
            let text = if serial.is_finite() {
                let weekday = naive_from_datenum(serial)?.weekday();
                weekday_name(weekday, short).to_string()
            } else {
                String::new()
            };
            out.push(Value::CharArray(
                CharArray::new(text.chars().collect(), 1, text.chars().count())
                    .map_err(|err| datetime_error(format!("day: {err}")))?,
            ));
        }
        return Ok(Value::Cell(
            runmat_value::CellArray::new_with_shape(out, shape)
                .map_err(|err| datetime_error(format!("day: {err}")))?,
        ));
    }
    let extractor: fn(&NaiveDateTime) -> f64 = match mode.as_str() {
        "dayofmonth" => |value| value.day() as f64,
        "dayofweek" => |value| f64::from(value.weekday().num_days_from_sunday() + 1),
        "iso-dayofweek" => |value| f64::from(value.weekday().number_from_monday()),
        "dayofyear" => |value| f64::from(value.ordinal()),
        _ => {
            return Err(datetime_error(format!(
                "day: unsupported day type '{mode}'"
            )))
        }
    };
    let mut out = Vec::with_capacity(values.len());
    for serial in values.iter().copied() {
        out.push(if serial.is_finite() {
            extractor(&naive_from_datenum(serial)?)
        } else {
            f64::NAN
        });
    }
    tensor_or_scalar(out, shape)
}

fn weekday_name(weekday: Weekday, short: bool) -> &'static str {
    match (weekday, short) {
        (Weekday::Mon, false) => "Monday",
        (Weekday::Tue, false) => "Tuesday",
        (Weekday::Wed, false) => "Wednesday",
        (Weekday::Thu, false) => "Thursday",
        (Weekday::Fri, false) => "Friday",
        (Weekday::Sat, false) => "Saturday",
        (Weekday::Sun, false) => "Sunday",
        (Weekday::Mon, true) => "Mon",
        (Weekday::Tue, true) => "Tue",
        (Weekday::Wed, true) => "Wed",
        (Weekday::Thu, true) => "Thu",
        (Weekday::Fri, true) => "Fri",
        (Weekday::Sat, true) => "Sat",
        (Weekday::Sun, true) => "Sun",
    }
}

#[runmat_macros::runtime_builtin(
    name = "hour",
    descriptor(crate::builtins::datetime::DATETIME_HOUR_DESCRIPTOR),
    extensions(crate::builtins::datetime::HOUR_EXTENSIONS),
    integer_capabilities(crate::builtins::datetime::HOUR_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Extract hour components from datetime values.",
    keywords = "hour,datetime,time component"
)]
async fn hour_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let (value, format) =
        prepare_legacy_component_input("hour", value, &rest, &HOUR_TYPED_LEGACY_NUMERIC_EXTENSION)
            .await?;
    numeric_component_from_modern_or_legacy("hour", value, format.as_deref(), |naive| {
        naive.hour() as f64
    })
}

#[runmat_macros::runtime_builtin(
    name = "minute",
    descriptor(crate::builtins::datetime::DATETIME_MINUTE_DESCRIPTOR),
    extensions(crate::builtins::datetime::MINUTE_EXTENSIONS),
    integer_capabilities(crate::builtins::datetime::MINUTE_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Extract minute numbers from datetime arrays.",
    keywords = "minute,datetime,time component"
)]
async fn minute_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let (value, format) = prepare_legacy_component_input(
        "minute",
        value,
        &rest,
        &MINUTE_TYPED_LEGACY_NUMERIC_EXTENSION,
    )
    .await?;
    numeric_component_from_modern_or_legacy("minute", value, format.as_deref(), |naive| {
        naive.minute() as f64
    })
}

#[runmat_macros::runtime_builtin(
    name = "second",
    descriptor(crate::builtins::datetime::DATETIME_SECOND_DESCRIPTOR),
    integer_audit(crate::builtins::datetime::SECOND_INTEGER_AUDIT),
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
                    serial_tensor_from_value(args[0].clone(), "datenum")?
                }
            }
            _ => serial_tensor_from_value(args[0].clone(), "datenum")?,
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
    if tensor::tensor_element_len(&tensor) == 1 {
        Ok(Value::Num(tensor::tensor_value_f64(&tensor, 0)))
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
        let matrix_values = tensor::tensor_values_f64_cow(&matrix);
        for col in 0..6.min(out_count) {
            let mut data = Vec::with_capacity(matrix.rows);
            for row in 0..matrix.rows {
                data.push(matrix_values[col * matrix.rows + row]);
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
    let serial_values = tensor::tensor_values_f64_cow(&serials);
    let mut rows = Vec::with_capacity(serial_values.len());
    for serial in serial_values.iter() {
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
    let serial_values = tensor::tensor_values_f64_cow(&serials);
    let mut nums = Vec::with_capacity(serial_values.len());
    let mut names = Vec::with_capacity(serial_values.len());
    for serial in serial_values.iter() {
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
    let shape = tensor::default_shape_for(&serials.shape, serial_values.len());
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
            let shape = tensor::default_shape_for(&days.shape, tensor::tensor_element_len(&days));
            let day_values = tensor::tensor_into_values_f64(days);
            return calendar_duration_object_from_components(
                vec![0.0; day_values.len()],
                day_values,
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
    let serial_values = tensor::tensor_values_f64_cow(&serials);
    let mut out = Vec::with_capacity(serial_values.len());
    for serial in serial_values.iter() {
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
        tensor::default_shape_for(&serials.shape, serial_values.len()),
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
                let year =
                    tensor::is_scalar_tensor(&tensor).then(|| tensor::tensor_value_f64(&tensor, 0));
                if let Some(year) = year.filter(|year| (1000.0..=9999.0).contains(year)) {
                    let keys = market_holiday_keys_for_year(year.round() as i32)?;
                    let len = keys.len();
                    return datetime_object_from_serials(
                        keys.into_iter().map(|key| key as f64).collect(),
                        vec![len, 1],
                        DEFAULT_DATE_FORMAT,
                    );
                }
            }
            let serials = numeric_or_datetime_serial_tensor(args[0].clone(), "holidays")?;
            let year =
                date_from_key(serial_date_key(tensor::tensor_value_f64(&serials, 0))?)?.year();
            let start = key_from_date(NaiveDate::from_ymd_opt(year, 1, 1).unwrap());
            let end = key_from_date(NaiveDate::from_ymd_opt(year, 12, 31).unwrap());
            holiday_keys_between(start, end)?
        }
        2 => {
            let start = numeric_or_datetime_serial_tensor(args[0].clone(), "holidays")?;
            let end = numeric_or_datetime_serial_tensor(args[1].clone(), "holidays")?;
            if tensor::tensor_element_len(&start) != 1 || tensor::tensor_element_len(&end) != 1 {
                return Err(datetime_error(
                    "holidays: start and end dates must be scalar",
                ));
            }
            holiday_keys_between(
                serial_date_key(tensor::tensor_value_f64(&start, 0))?,
                serial_date_key(tensor::tensor_value_f64(&end, 0))?,
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
    if tensor::tensor_element_len(&start) != 1 || tensor::tensor_element_len(&end) != 1 {
        return Err(datetime_error(
            "busdays: start and end dates must be scalar",
        ));
    }
    let mut key = serial_date_key(tensor::tensor_value_f64(&start, 0))?;
    let end_key = serial_date_key(tensor::tensor_value_f64(&end, 0))?;
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
        .and_then(|tensor| {
            tensor::is_scalar_tensor(&tensor).then(|| tensor::tensor_value_f64(&tensor, 0))
        })
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
    let left_values = tensor::tensor_values_f64_cow(&left);
    let right_values = tensor::tensor_values_f64_cow(&right);
    let out = left_values
        .iter()
        .zip(right_values.iter())
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
    let lhs_days_shape =
        tensor::default_shape_for(&lhs_days.shape, tensor::tensor_element_len(&lhs_days));
    let rhs_days_shape =
        tensor::default_shape_for(&rhs_days.shape, tensor::tensor_element_len(&rhs_days));
    let lhs_day_tensor = Tensor::new(tensor::tensor_into_values_f64(lhs_days), lhs_days_shape)
        .map_err(|err| datetime_error(format!("{context}: {err}")))?;
    let rhs_day_tensor = Tensor::new(tensor::tensor_into_values_f64(rhs_days), rhs_days_shape)
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
        Value::Tensor(tensor) => {
            let shape = tensor.shape.clone();
            let values = tensor::tensor_into_values_f64(tensor);
            Ok(Value::Tensor(
                Tensor::new(
                    values
                        .into_iter()
                        .map(|value| if value == 0.0 { 1.0 } else { 0.0 })
                        .collect(),
                    shape,
                )
                .map_err(|err| datetime_error(format!("ne: {err}")))?,
            ))
        }
        other => Ok(other),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DateShiftBoundary {
    Start,
    End,
    DayOfWeek,
}

impl DateShiftBoundary {
    fn parse(value: &Value) -> BuiltinResult<Self> {
        let text = scalar_text(value, "dateshift boundary")?;
        match text.trim().to_ascii_lowercase().as_str() {
            "start" => Ok(Self::Start),
            "end" => Ok(Self::End),
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
            "year" => Ok(Self::Year),
            "quarter" => Ok(Self::Quarter),
            "month" => Ok(Self::Month),
            "week" => Ok(Self::Week),
            "day" => Ok(Self::Day),
            "hour" => Ok(Self::Hour),
            "minute" => Ok(Self::Minute),
            "second" => Ok(Self::Second),
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

fn start_of_week(value: NaiveDateTime, week_start: Weekday) -> BuiltinResult<NaiveDateTime> {
    let current = value.weekday().num_days_from_monday() as i64;
    let start = week_start.num_days_from_monday() as i64;
    let delta = (current - start).rem_euclid(7);
    value
        .date()
        .checked_sub_signed(Duration::days(delta))
        .map(midnight)
        .ok_or_else(|| datetime_error("dateshift: result is outside the supported range"))
}

fn start_of_unit(
    value: NaiveDateTime,
    unit: DateShiftUnit,
    week_start: Weekday,
) -> BuiltinResult<NaiveDateTime> {
    let start = match unit {
        DateShiftUnit::Year => midnight(NaiveDate::from_ymd_opt(value.year(), 1, 1).unwrap()),
        DateShiftUnit::Quarter => {
            let month = ((value.month() - 1) / 3) * 3 + 1;
            midnight(NaiveDate::from_ymd_opt(value.year(), month, 1).unwrap())
        }
        DateShiftUnit::Month => {
            midnight(NaiveDate::from_ymd_opt(value.year(), value.month(), 1).unwrap())
        }
        DateShiftUnit::Week => return start_of_week(value, week_start),
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
    };
    Ok(start)
}

fn next_unit_start(start: NaiveDateTime, unit: DateShiftUnit) -> BuiltinResult<NaiveDateTime> {
    let out_of_range = || datetime_error("dateshift: result is outside the supported range");
    match unit {
        DateShiftUnit::Year => {
            let year = start.year().checked_add(1).ok_or_else(out_of_range)?;
            return NaiveDate::from_ymd_opt(year, 1, 1)
                .map(midnight)
                .ok_or_else(out_of_range);
        }
        DateShiftUnit::Quarter | DateShiftUnit::Month => {
            let delta = if matches!(unit, DateShiftUnit::Quarter) {
                3
            } else {
                1
            };
            let month_index = i64::from(start.year())
                .checked_mul(12)
                .and_then(|base| base.checked_add(i64::from(start.month0())))
                .and_then(|base| base.checked_add(delta))
                .ok_or_else(out_of_range)?;
            let year = i32::try_from(month_index.div_euclid(12)).map_err(|_| out_of_range())?;
            let month = month_index.rem_euclid(12) as u32 + 1;
            return NaiveDate::from_ymd_opt(year, month, 1)
                .map(midnight)
                .ok_or_else(out_of_range);
        }
        DateShiftUnit::Week => start.checked_add_signed(Duration::days(7)),
        DateShiftUnit::Day => start.checked_add_signed(Duration::days(1)),
        DateShiftUnit::Hour => start.checked_add_signed(Duration::hours(1)),
        DateShiftUnit::Minute => start.checked_add_signed(Duration::minutes(1)),
        DateShiftUnit::Second => start.checked_add_signed(Duration::seconds(1)),
    }
    .ok_or_else(out_of_range)
}

#[derive(Clone, Copy)]
enum DateShiftRule {
    Current,
    Next,
    Previous,
    Nearest,
    Occurrence(i64),
}

fn exact_f64_to_i64(value: f64) -> Option<i64> {
    // i64::MAX rounds upward to 2^63 as f64, so the upper test must be
    // half-open. The lower endpoint -2^63 is exactly representable.
    const I64_LOWER_INCLUSIVE: f64 = -9_223_372_036_854_775_808.0;
    const I64_UPPER_EXCLUSIVE: f64 = 9_223_372_036_854_775_808.0;
    (value.is_finite()
        && value.fract() == 0.0
        && (I64_LOWER_INCLUSIVE..I64_UPPER_EXCLUSIVE).contains(&value))
    .then_some(value as i64)
}

fn exact_integer_values(value: &Value, context: &str) -> BuiltinResult<(Vec<i64>, Vec<usize>)> {
    match value {
        Value::Int(value) => value
            .try_to_i64()
            .map(|value| (vec![value], vec![1, 1]))
            .ok_or_else(|| datetime_error(format!("dateshift: {context} is outside int64 range"))),
        Value::Num(value) => exact_f64_to_i64(*value)
            .map(|value| (vec![value], vec![1, 1]))
            .ok_or_else(|| {
                datetime_error(format!(
                    "dateshift: {context} must be a representable integer"
                ))
            }),
        Value::Tensor(array) => {
            let shape = tensor::default_shape_for(&array.shape, tensor::tensor_element_len(array));
            if let Some(storage) = array.integer_storage() {
                let mut out = Vec::with_capacity(storage.len());
                for value in storage.exact_values() {
                    out.push(value.try_to_i64().ok_or_else(|| {
                        datetime_error(format!("dateshift: {context} is outside int64 range"))
                    })?);
                }
                Ok((out, shape))
            } else {
                let mut out = Vec::with_capacity(tensor::tensor_element_len(array));
                for value in tensor::tensor_values_f64_cow(array).iter().copied() {
                    out.push(exact_f64_to_i64(value).ok_or_else(|| {
                        datetime_error(format!(
                            "dateshift: {context} values must be representable integers"
                        ))
                    })?);
                }
                Ok((out, shape))
            }
        }
        Value::Bool(_) | Value::LogicalArray(_) => Err(datetime_error(format!(
            "dateshift: {context} does not accept logical values"
        ))),
        _ => Err(datetime_error(format!(
            "dateshift: {context} must be an integer scalar or array"
        ))),
    }
}

fn parse_rules(value: Option<&Value>) -> BuiltinResult<(Vec<DateShiftRule>, Vec<usize>)> {
    let Some(value) = value else {
        return Ok((vec![DateShiftRule::Current], vec![1, 1]));
    };
    if matches!(
        value,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
    ) {
        let text = scalar_text(value, "dateshift rule")?;
        let rule = match text.trim().to_ascii_lowercase().as_str() {
            "current" => DateShiftRule::Current,
            "next" => DateShiftRule::Next,
            "previous" => DateShiftRule::Previous,
            "nearest" => DateShiftRule::Nearest,
            other => {
                return Err(datetime_error(format!(
                    "dateshift: unsupported rule '{other}'"
                )))
            }
        };
        return Ok((vec![rule], vec![1, 1]));
    }
    let (values, shape) = exact_integer_values(value, "rule")?;
    Ok((
        values.into_iter().map(DateShiftRule::Occurrence).collect(),
        shape,
    ))
}

fn unit_step(
    value: NaiveDateTime,
    unit: DateShiftUnit,
    steps: i64,
) -> BuiltinResult<NaiveDateTime> {
    match unit {
        DateShiftUnit::Year | DateShiftUnit::Quarter | DateShiftUnit::Month => {
            let factor = match unit {
                DateShiftUnit::Year => 12,
                DateShiftUnit::Quarter => 3,
                _ => 1,
            };
            let delta = steps
                .checked_mul(factor)
                .ok_or_else(|| datetime_error("dateshift: rule is outside the supported range"))?;
            let month_index = i64::from(value.year())
                .checked_mul(12)
                .and_then(|base| base.checked_add(i64::from(value.month0())))
                .and_then(|base| base.checked_add(delta))
                .ok_or_else(|| {
                    datetime_error("dateshift: result is outside the supported range")
                })?;
            let year = i32::try_from(month_index.div_euclid(12))
                .map_err(|_| datetime_error("dateshift: result is outside the supported range"))?;
            let month = month_index.rem_euclid(12) as u32 + 1;
            return Ok(midnight(
                NaiveDate::from_ymd_opt(year, month, 1).ok_or_else(|| {
                    datetime_error("dateshift: result is outside the supported range")
                })?,
            ));
        }
        DateShiftUnit::Week
        | DateShiftUnit::Day
        | DateShiftUnit::Hour
        | DateShiftUnit::Minute
        | DateShiftUnit::Second => {
            let duration = match unit {
                DateShiftUnit::Week => Duration::try_weeks(steps),
                DateShiftUnit::Day => Duration::try_days(steps),
                DateShiftUnit::Hour => Duration::try_hours(steps),
                DateShiftUnit::Minute => Duration::try_minutes(steps),
                DateShiftUnit::Second => Duration::try_seconds(steps),
                _ => unreachable!("calendar units returned above"),
            }
            .ok_or_else(|| datetime_error("dateshift: rule is outside the supported range"))?;
            value.checked_add_signed(duration)
        }
    }
    .ok_or_else(|| datetime_error("dateshift: result is outside the supported range"))
}

fn unit_end(start: NaiveDateTime, unit: DateShiftUnit) -> BuiltinResult<NaiveDateTime> {
    let next = next_unit_start(start, unit)?;
    match unit {
        DateShiftUnit::Year
        | DateShiftUnit::Quarter
        | DateShiftUnit::Month
        | DateShiftUnit::Week => next.checked_sub_signed(Duration::days(1)),
        DateShiftUnit::Day
        | DateShiftUnit::Hour
        | DateShiftUnit::Minute
        | DateShiftUnit::Second => Some(next),
    }
    .ok_or_else(|| datetime_error("dateshift: result is outside the supported range"))
}

fn apply_boundary_rule(
    value: NaiveDateTime,
    boundary: DateShiftBoundary,
    unit: DateShiftUnit,
    rule: DateShiftRule,
) -> BuiltinResult<NaiveDateTime> {
    let start = start_of_unit(value, unit, Weekday::Sun)?;
    let current = if boundary == DateShiftBoundary::Start {
        start
    } else {
        unit_end(start, unit)?
    };
    let shifted = |steps| {
        let shifted_start = unit_step(start, unit, steps)?;
        Ok(if boundary == DateShiftBoundary::Start {
            shifted_start
        } else {
            unit_end(shifted_start, unit)?
        })
    };
    match rule {
        DateShiftRule::Current => Ok(current),
        DateShiftRule::Next => shifted(1),
        DateShiftRule::Previous => shifted(-1),
        DateShiftRule::Nearest => {
            let previous = if value >= current {
                current
            } else {
                shifted(-1)?
            };
            let next = if value <= current {
                current
            } else {
                shifted(1)?
            };
            if value - previous <= next - value {
                Ok(previous)
            } else {
                Ok(next)
            }
        }
        DateShiftRule::Occurrence(0) => Ok(current),
        DateShiftRule::Occurrence(n) => shifted(n),
    }
}

#[derive(Clone, Copy)]
enum DayTarget {
    Exact(Weekday),
    Weekend,
    Weekday,
}

fn target_matches(target: DayTarget, weekday: Weekday) -> bool {
    match target {
        DayTarget::Exact(expected) => weekday == expected,
        DayTarget::Weekend => matches!(weekday, Weekday::Sat | Weekday::Sun),
        DayTarget::Weekday => !matches!(weekday, Weekday::Sat | Weekday::Sun),
    }
}

fn current_week_target(origin: NaiveDateTime, target: DayTarget) -> BuiltinResult<NaiveDateTime> {
    let sunday = start_of_week(origin, Weekday::Sun)?;
    let add_days = |days| {
        sunday
            .checked_add_signed(Duration::days(days))
            .ok_or_else(|| datetime_error("dateshift: result is outside the supported range"))
    };
    match target {
        DayTarget::Exact(weekday) => add_days(i64::from(weekday.num_days_from_sunday())),
        DayTarget::Weekend => {
            if matches!(origin.weekday(), Weekday::Sat | Weekday::Sun) {
                Ok(origin)
            } else {
                add_days(6)
            }
        }
        DayTarget::Weekday => match origin.weekday() {
            Weekday::Sun => add_days(1),
            Weekday::Sat => add_days(5),
            _ => Ok(origin),
        },
    }
}

fn shift_day_target(
    value: NaiveDateTime,
    target: DayTarget,
    rule: DateShiftRule,
) -> BuiltinResult<NaiveDateTime> {
    let origin = midnight(value.date());
    let seek = |direction: i64, occurrence: u64| -> BuiltinResult<NaiveDateTime> {
        if occurrence == 0 || occurrence > MAX_DATESHIFT_DAY_OCCURRENCE {
            return Err(datetime_error(
                "dateshift: day occurrence rule is outside the supported range",
            ));
        }

        let mut date = origin;
        for _ in 0..7 {
            if target_matches(target, date.weekday()) {
                break;
            }
            date = date
                .checked_add_signed(Duration::days(direction))
                .ok_or_else(|| {
                    datetime_error("dateshift: result is outside the supported range")
                })?;
        }
        if !target_matches(target, date.weekday()) {
            return Err(datetime_error(
                "dateshift: result is outside the supported range",
            ));
        }

        let matches_per_week = match target {
            DayTarget::Exact(_) => 1,
            DayTarget::Weekend => 2,
            DayTarget::Weekday => 5,
        };
        let remaining = occurrence - 1;
        let whole_weeks = remaining / matches_per_week;
        let residual = remaining % matches_per_week;
        let whole_week_days = i64::try_from(whole_weeks.checked_mul(7).ok_or_else(|| {
            datetime_error("dateshift: day occurrence rule is outside the supported range")
        })?)
        .map_err(|_| {
            datetime_error("dateshift: day occurrence rule is outside the supported range")
        })?;
        let signed_days = whole_week_days
            .checked_mul(direction)
            .ok_or_else(|| datetime_error("dateshift: rule is outside the supported range"))?;
        let delta = Duration::try_days(signed_days)
            .ok_or_else(|| datetime_error("dateshift: rule is outside the supported range"))?;
        date = date
            .checked_add_signed(delta)
            .ok_or_else(|| datetime_error("dateshift: result is outside the supported range"))?;

        let mut residual_found = 0;
        while residual_found < residual {
            date = date
                .checked_add_signed(Duration::days(direction))
                .ok_or_else(|| {
                    datetime_error("dateshift: result is outside the supported range")
                })?;
            if target_matches(target, date.weekday()) {
                residual_found += 1;
            }
        }
        Ok(date)
    };
    match rule {
        DateShiftRule::Current => current_week_target(origin, target),
        DateShiftRule::Next => seek(1, 1),
        DateShiftRule::Previous => seek(-1, 1),
        DateShiftRule::Nearest => {
            let previous = seek(-1, 1)?;
            let next = seek(1, 1)?;
            if origin - previous <= next - origin {
                Ok(previous)
            } else {
                Ok(next)
            }
        }
        DateShiftRule::Occurrence(0) => current_week_target(origin, target),
        DateShiftRule::Occurrence(n) if n > 0 => seek(1, n as u64),
        DateShiftRule::Occurrence(n) => seek(-1, n.unsigned_abs()),
    }
}

#[runmat_macros::runtime_builtin(
    name = "dateshift",
    descriptor(crate::builtins::datetime::DATESHIFT_DESCRIPTOR),
    extensions(crate::builtins::datetime::DATESHIFT_EXTENSIONS),
    integer_capabilities(crate::builtins::datetime::DATESHIFT_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::datetime",
    category = "datetime",
    summary = "Shift datetime values to calendar or clock boundaries.",
    keywords = "dateshift,datetime,start,end,dayofweek,weekday,weekend,rule",
    related = "datetime,year,month,day"
)]
async fn dateshift_builtin(
    value: Value,
    boundary: Value,
    unit: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    if std::iter::once(&value)
        .chain(std::iter::once(&boundary))
        .chain(std::iter::once(&unit))
        .chain(rest.iter())
        .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DATETIME_GPU_INPUT_EXTENSION,
            "dateshift",
        )?;
    }
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

    let serial_values = tensor::tensor_values_f64_cow(&serials);
    if rest.len() > 1 {
        return Err(datetime_error(
            "dateshift: expected at most one rule argument",
        ));
    }
    let default_rule = if boundary == DateShiftBoundary::DayOfWeek {
        DateShiftRule::Next
    } else {
        DateShiftRule::Current
    };
    let (rules, rule_shape) = if rest.is_empty() {
        (vec![default_rule], vec![1, 1])
    } else {
        parse_rules(rest.first())?
    };
    let mut targets = vec![DayTarget::Exact(Weekday::Sun)];
    let mut target_shape = vec![1, 1];
    if boundary == DateShiftBoundary::DayOfWeek {
        if matches!(
            unit,
            Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
        ) {
            let text = scalar_text(&unit, "dateshift weekday")?;
            targets[0] = match text.trim().to_ascii_lowercase().as_str() {
                "weekend" => DayTarget::Weekend,
                "weekday" => DayTarget::Weekday,
                _ => DayTarget::Exact(parse_weekday(&unit)?),
            };
        } else {
            let (indices, shape) = exact_integer_values(&unit, "weekday")?;
            targets = indices
                .into_iter()
                .map(weekday_from_matlab_index)
                .collect::<BuiltinResult<Vec<_>>>()?
                .into_iter()
                .map(DayTarget::Exact)
                .collect();
            target_shape = shape;
        }
    }
    let lengths = [serial_values.len(), targets.len(), rules.len()];
    let output_len = *lengths.iter().max().unwrap_or(&1);
    if lengths.iter().any(|len| *len != 1 && *len != output_len) {
        return Err(datetime_error(
            "dateshift: datetime, weekday, and rule arrays must have matching sizes or be scalar",
        ));
    }
    let serial_shape = tensor::default_shape_for(&serials.shape, serial_values.len());
    let mut non_scalar_shapes = Vec::new();
    if serial_values.len() > 1 {
        non_scalar_shapes.push(&serial_shape);
    }
    if targets.len() > 1 {
        non_scalar_shapes.push(&target_shape);
    }
    if rules.len() > 1 {
        non_scalar_shapes.push(&rule_shape);
    }
    if non_scalar_shapes.windows(2).any(|pair| pair[0] != pair[1]) {
        return Err(datetime_error(
            "dateshift: non-scalar datetime, weekday, and rule arrays must have matching sizes",
        ));
    }
    let output_shape = if serial_values.len() > 1 {
        serial_shape
    } else if targets.len() > 1 {
        target_shape
    } else if rules.len() > 1 {
        rule_shape
    } else {
        vec![1, 1]
    };
    let parsed_unit = if boundary == DateShiftBoundary::DayOfWeek {
        None
    } else {
        Some(DateShiftUnit::parse(&unit)?)
    };
    let mut out = Vec::with_capacity(output_len);
    for index in 0..output_len {
        let serial = serial_values[if serial_values.len() == 1 { 0 } else { index }];
        if !serial.is_finite() {
            out.push(serial);
            continue;
        }
        let value = naive_from_datenum(serial)?;
        let rule = rules[if rules.len() == 1 { 0 } else { index }];
        let shifted = if boundary == DateShiftBoundary::DayOfWeek {
            let target = targets[if targets.len() == 1 { 0 } else { index }];
            shift_day_target(value, target, rule)?
        } else {
            apply_boundary_rule(value, boundary, parsed_unit.unwrap(), rule)?
        };
        out.push(datenum_from_naive(shifted));
    }
    datetime_object_from_serials(out, output_shape, format)
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

    fn integer_tensor(storage: runmat_value::IntegerStorage, shape: Vec<usize>) -> Value {
        let tensor = Tensor::new_integer(storage, shape).expect("integer tensor");
        Value::Tensor(tensor)
    }

    #[test]
    fn datetime_descriptor_signatures_cover_constructor_and_methods() {
        let labels: Vec<&str> = DATETIME_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"t = datetime()"));
        assert!(labels.contains(&"t = datetime(dateVectors)"));
        assert!(labels.contains(&"t = datetime(year, month, day, hour, minute, second)"));
        assert!(
            labels.contains(&"t = datetime(year, month, day, hour, minute, second, millisecond)")
        );
        assert!(labels.contains(&"t = datetime(serialDateNumbers, \"ConvertFrom\", \"datenum\")"));

        assert_eq!(DATETIME_YEAR_DESCRIPTOR.signatures[0].label, "X = year(t)");
        assert!(DATETIME_HOUR_DESCRIPTOR
            .signatures
            .iter()
            .any(|signature| signature.label == "X = hour(t, F)"));
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
    fn hour_supports_datetime_legacy_serial_and_formatted_text_shapes() {
        let datetime = run_datetime(vec![Value::from("2024-03-14 09:26:53")]);
        assert_eq!(
            futures::executor::block_on(hour_builtin(datetime, Vec::new())).unwrap(),
            Value::Num(9.0)
        );

        let serials = Tensor::new(
            vec![
                serial_for_date(2024, 3, 14) + 3.0 / 24.0,
                serial_for_date(2024, 3, 14) + 17.0 / 24.0,
            ],
            vec![1, 2],
        )
        .unwrap();
        let result =
            futures::executor::block_on(hour_builtin(Value::Tensor(serials), Vec::new())).unwrap();
        let Value::Tensor(result) = result else {
            panic!("expected shaped hour result");
        };
        assert_eq!(result.shape, vec![1, 2]);
        assert_eq!(result.materialize_f64(), vec![3.0, 17.0]);

        let formatted = futures::executor::block_on(hour_builtin(
            Value::from("14/03/2024 21:05:00"),
            vec![Value::from("dd/MM/yyyy HH:mm:ss")],
        ))
        .unwrap();
        assert_eq!(formatted, Value::Num(21.0));

        let documented = futures::executor::block_on(hour_builtin(
            Value::from("2024/14/03 09:26:53.125"),
            vec![Value::from("yyyy/dd/mm hh:MM:ss.fff")],
        ))
        .expect("documented datestr format language");
        assert_eq!(documented, Value::Num(9.0));

        let default_fractional = futures::executor::block_on(hour_builtin(
            Value::from("2024/03/14 17:26:53.125"),
            Vec::new(),
        ))
        .expect("year-first fractional legacy text");
        assert_eq!(default_fractional, Value::Num(17.0));
    }

    #[test]
    fn hour_gates_uncertain_typed_and_resident_legacy_extensions_before_access() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let typed = futures::executor::block_on(hour_builtin(
            integer_tensor(runmat_value::IntegerStorage::U32(vec![739_000]), vec![1, 1]),
            Vec::new(),
        ))
        .expect_err("typed legacy serial must be gated");
        assert_eq!(
            typed.identifier(),
            Some("RunMat:compatibility:HourTypedLegacySerialExtension")
        );

        let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: 0,
            buffer_id: 9_419_002,
            descriptor: Default::default(),
        });
        let resident = futures::executor::block_on(hour_builtin(resident, Vec::new()))
            .expect_err("resident legacy input must gate before provider access");
        assert_eq!(
            resident.identifier(),
            Some("RunMat:compatibility:DatetimeGpuInputExtension")
        );
    }

    #[test]
    fn hour_minute_and_month_typed_legacy_serials_cover_every_integer_class() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let serial = 42;
        let cases = [
            runmat_value::IntegerStorage::I8(vec![serial as i8]),
            runmat_value::IntegerStorage::I16(vec![serial as i16]),
            runmat_value::IntegerStorage::I32(vec![serial]),
            runmat_value::IntegerStorage::I64(vec![i64::from(serial)]),
            runmat_value::IntegerStorage::U8(vec![serial as u8]),
            runmat_value::IntegerStorage::U16(vec![serial as u16]),
            runmat_value::IntegerStorage::U32(vec![serial as u32]),
            runmat_value::IntegerStorage::U64(vec![serial as u64]),
        ];
        let expected_hour =
            futures::executor::block_on(hour_builtin(Value::Num(f64::from(serial)), Vec::new()))
                .expect("double legacy hour");
        let expected_minute =
            futures::executor::block_on(minute_builtin(Value::Num(f64::from(serial)), Vec::new()))
                .expect("double legacy minute");
        let expected_month =
            futures::executor::block_on(month_builtin(Value::Num(f64::from(serial)), Vec::new()))
                .expect("double legacy month");

        for storage in cases {
            let value = integer_tensor(storage.clone(), vec![1, 1]);
            assert_eq!(
                futures::executor::block_on(hour_builtin(value.clone(), Vec::new()))
                    .expect("typed legacy hour"),
                expected_hour
            );
            assert_eq!(
                futures::executor::block_on(minute_builtin(value.clone(), Vec::new()))
                    .expect("typed legacy minute"),
                expected_minute
            );
            assert_eq!(
                futures::executor::block_on(month_builtin(value, Vec::new()))
                    .expect("typed legacy month"),
                expected_month
            );
        }
    }

    #[test]
    fn minute_and_month_gate_typed_and_resident_extensions_before_access() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        for (result, identifier) in [
            (
                futures::executor::block_on(minute_builtin(
                    integer_tensor(runmat_value::IntegerStorage::U16(vec![42]), vec![1, 1]),
                    Vec::new(),
                )),
                "RunMat:compatibility:MinuteTypedLegacySerialExtension",
            ),
            (
                futures::executor::block_on(month_builtin(
                    integer_tensor(runmat_value::IntegerStorage::U16(vec![42]), vec![1, 1]),
                    Vec::new(),
                )),
                "RunMat:compatibility:MonthTypedLegacySerialExtension",
            ),
        ] {
            assert_eq!(
                result
                    .expect_err("typed legacy input must gate")
                    .identifier(),
                Some(identifier)
            );
        }

        for result in [
            futures::executor::block_on(minute_builtin(
                Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
                    shape: vec![1, 1],
                    device_id: 0,
                    buffer_id: 9_419_003,
                    descriptor: Default::default(),
                }),
                Vec::new(),
            )),
            futures::executor::block_on(month_builtin(
                Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
                    shape: vec![1, 1],
                    device_id: 0,
                    buffer_id: 9_419_004,
                    descriptor: Default::default(),
                }),
                Vec::new(),
            )),
        ] {
            assert_eq!(
                result
                    .expect_err("resident legacy input must gate before provider access")
                    .identifier(),
                Some("RunMat:compatibility:DatetimeGpuInputExtension")
            );
        }
    }

    #[test]
    fn month_datetime_name_modes_return_shaped_character_cells() {
        let datetime = run_datetime(vec![Value::from("2024-03-14 09:26:53")]);
        for (mode, expected) in [("name", "March"), ("shortname", "Mar")] {
            let result = futures::executor::block_on(month_builtin(
                datetime.clone(),
                vec![Value::from(mode)],
            ))
            .expect("month name mode");
            let Value::Cell(cell) = result else {
                panic!("expected month name cell");
            };
            assert_eq!(cell.shape, vec![1, 1]);
            assert_eq!(
                cell.data,
                vec![Value::CharArray(CharArray::new_row(expected))]
            );
        }
    }

    #[test]
    fn datetime_builds_from_components() {
        let value = run_datetime(vec![Value::Num(2024.0), Value::Num(3.0), Value::Num(14.0)]);
        let object = as_datetime(value);
        assert_eq!(object.class_name, DATETIME_CLASS);
        assert_eq!(format_for_object(&object), DEFAULT_DATE_FORMAT);
        let serials = serial_tensor_for_object(&object).expect("serials");
        assert_eq!(serials.materialize_f64().len(), 1);
        let year =
            futures::executor::block_on(year_builtin(Value::Object(object.clone()), Vec::new()))
                .expect("year");
        assert_eq!(year, Value::Num(2024.0));
    }

    #[test]
    fn year_typed_legacy_serial_is_separately_gated() {
        let serial = serial_for_date(2024, 1, 1) as i64;
        let input = Value::Int(runmat_value::IntValue::I64(serial));
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = futures::executor::block_on(year_builtin(input.clone(), Vec::new()))
            .expect_err("strict gate");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:YearTypedLegacySerialExtension")
        );
        drop(_strict);

        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = futures::executor::block_on(year_builtin(input, Vec::new()))
            .expect("typed legacy serial");
        assert_eq!(result, Value::Num(2024.0));
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
    fn datetime_typed_integer_components_and_serials_cross_double_boundary_exactly() {
        let years = integer_tensor(
            runmat_value::IntegerStorage::U16(vec![2024, 2025]),
            vec![1, 2],
        );
        let months = integer_tensor(runmat_value::IntegerStorage::U8(vec![1, 6]), vec![1, 2]);
        let days = integer_tensor(runmat_value::IntegerStorage::I16(vec![15, 20]), vec![1, 2]);
        let value = run_datetime(vec![years, months, days]);
        let rendered = datetime_display_text(&value)
            .expect("display")
            .expect("datetime text");
        assert!(rendered.contains("15-Jan-2024"));
        assert!(rendered.contains("20-Jun-2025"));

        let serial = serial_for_date(2024, 3, 14);
        let object = run_datetime(vec![
            integer_tensor(
                runmat_value::IntegerStorage::U32(vec![serial as u32]),
                vec![1, 1],
            ),
            Value::from("ConvertFrom"),
            Value::from("datenum"),
        ]);
        assert_eq!(
            serials_from_datetime_value(&object)
                .unwrap()
                .materialize_f64(),
            vec![serial.floor()]
        );
    }

    #[test]
    fn datetime_date_vectors_cover_all_integer_classes_and_preaccess_gates() {
        let storages = [
            runmat_value::IntegerStorage::I8(vec![24, 1, 2]),
            runmat_value::IntegerStorage::I16(vec![2024, 1, 2]),
            runmat_value::IntegerStorage::I32(vec![2024, 1, 2]),
            runmat_value::IntegerStorage::I64(vec![2024, 1, 2]),
            runmat_value::IntegerStorage::U8(vec![24, 1, 2]),
            runmat_value::IntegerStorage::U16(vec![2024, 1, 2]),
            runmat_value::IntegerStorage::U32(vec![2024, 1, 2]),
            runmat_value::IntegerStorage::U64(vec![2024, 1, 2]),
        ];
        for storage in storages {
            let small_year = matches!(
                storage,
                runmat_value::IntegerStorage::I8(_) | runmat_value::IntegerStorage::U8(_)
            );
            let value = run_datetime(vec![integer_tensor(storage, vec![1, 3])]);
            assert_eq!(
                datetime_string_array(&value).unwrap().unwrap().data,
                vec![if small_year {
                    "02-Jan-0024"
                } else {
                    "02-Jan-2024"
                }
                .to_string()]
            );
        }

        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let logical = futures::executor::block_on(datetime_builtin(vec![Value::Bool(true)]))
            .expect_err("logical input is gated");
        assert_eq!(
            logical.identifier(),
            Some("RunMat:compatibility:DatetimeLogicalInputExtension")
        );
        let implicit_serial =
            futures::executor::block_on(datetime_builtin(vec![Value::Num(739_000.0)]))
                .expect_err("implicit serial input is gated");
        assert_eq!(
            implicit_serial.identifier(),
            Some("RunMat:compatibility:DatetimeImplicitDatenumExtension")
        );
        let legacy_arity = futures::executor::block_on(datetime_builtin(vec![
            Value::Num(2024.0),
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(3.0),
        ]))
        .expect_err("four-component constructor is gated");
        assert_eq!(
            legacy_arity.identifier(),
            Some("RunMat:compatibility:DatetimeLegacyComponentArityExtension")
        );
        let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 3],
            device_id: 0,
            buffer_id: 9_397_001,
            descriptor: Default::default(),
        });
        let resident = futures::executor::block_on(datetime_builtin(vec![resident]))
            .expect_err("resident input is gated before provider access");
        assert_eq!(
            resident.identifier(),
            Some("RunMat:compatibility:DatetimeGpuInputExtension")
        );
    }

    #[test]
    fn wide_integer_serials_are_rejected_before_lossy_conversion() {
        for storage in [
            runmat_value::IntegerStorage::U64(vec![(1_u64 << 53) + 1]),
            runmat_value::IntegerStorage::I64(vec![i64::MIN]),
        ] {
            let explicit = futures::executor::block_on(datetime_builtin(vec![
                integer_tensor(storage.clone(), vec![1, 1]),
                Value::from("ConvertFrom"),
                Value::from("datenum"),
            ]))
            .expect_err("wide explicit serial must be rejected while exact");
            assert!(explicit.message().contains("supported serial-date range"));

            let legacy_day = futures::executor::block_on(day_builtin(
                integer_tensor(storage, vec![1, 1]),
                Vec::new(),
            ))
            .expect_err("wide legacy day serial must be rejected while exact");
            assert!(legacy_day.message().contains("supported serial-date range"));
        }

        let extreme =
            naive_from_datenum(f64::MAX).expect_err("extreme finite serial must not wrap or panic");
        assert!(extreme.message().contains("outside the supported range"));
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
            serials_from_datetime_value(&converted)
                .unwrap()
                .materialize_f64(),
            serials_from_datetime_value(&value)
                .unwrap()
                .materialize_f64()
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
    fn dateshift_supports_sunday_start_of_week_and_public_month_end() {
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
            vec!["2024-03-10".to_string(), "2024-03-17".to_string()]
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
        assert_eq!(rendered.data, vec!["2024-02-29 00:00:00".to_string()]);
    }

    #[test]
    fn dateshift_rules_follow_current_week_and_boundary_contracts() {
        let input = run_datetime(vec![Value::from("2024-03-14")]);
        let monday = futures::executor::block_on(dateshift_builtin(
            input.clone(),
            Value::from("dayofweek"),
            Value::from("monday"),
            vec![Value::from("current")],
        ))
        .expect("current-week Monday");
        assert_eq!(
            datetime_string_array(&monday).unwrap().unwrap().data,
            vec!["11-Mar-2024".to_string()]
        );

        let current_zero = futures::executor::block_on(dateshift_builtin(
            input.clone(),
            Value::from("dayofweek"),
            Value::from("monday"),
            vec![Value::Int(runmat_value::IntValue::I8(0))],
        ))
        .expect("numeric zero current-week Monday");
        assert_eq!(
            datetime_string_array(&current_zero).unwrap().unwrap().data,
            vec!["11-Mar-2024".to_string()]
        );

        let next = futures::executor::block_on(dateshift_builtin(
            run_datetime(vec![Value::from("2024-03-01")]),
            Value::from("start"),
            Value::from("month"),
            vec![Value::from("next")],
        ))
        .expect("next exact boundary");
        assert_eq!(
            datetime_string_array(&next).unwrap().unwrap().data,
            vec!["01-Apr-2024".to_string()]
        );

        for weekday in [
            runmat_value::IntValue::I8(2),
            runmat_value::IntValue::I16(2),
            runmat_value::IntValue::I32(2),
            runmat_value::IntValue::I64(2),
            runmat_value::IntValue::U8(2),
            runmat_value::IntValue::U16(2),
            runmat_value::IntValue::U32(2),
            runmat_value::IntValue::U64(2),
        ] {
            let shifted = futures::executor::block_on(dateshift_builtin(
                input.clone(),
                Value::from("dayofweek"),
                Value::Int(weekday),
                Vec::new(),
            ))
            .expect("typed weekday");
            assert_eq!(
                datetime_string_array(&shifted).unwrap().unwrap().data,
                vec!["18-Mar-2024".to_string()]
            );
        }
    }

    #[test]
    fn dateshift_day_occurrences_use_bounded_calendar_arithmetic() {
        let origin = NaiveDate::from_ymd_opt(2024, 3, 14)
            .unwrap()
            .and_hms_opt(12, 30, 0)
            .unwrap();

        let sixth_weekday =
            shift_day_target(origin, DayTarget::Weekday, DateShiftRule::Occurrence(6))
                .expect("sixth weekday");
        assert_eq!(
            sixth_weekday.date(),
            NaiveDate::from_ymd_opt(2024, 3, 21).unwrap()
        );

        let third_prior_weekend =
            shift_day_target(origin, DayTarget::Weekend, DateShiftRule::Occurrence(-3))
                .expect("third prior weekend day");
        assert_eq!(
            third_prior_weekend.date(),
            NaiveDate::from_ymd_opt(2024, 3, 3).unwrap()
        );

        assert!(shift_day_target(
            origin,
            DayTarget::Exact(Weekday::Mon),
            DateShiftRule::Occurrence(MAX_DATESHIFT_DAY_OCCURRENCE as i64 + 1),
        )
        .is_err());
        assert!(shift_day_target(
            origin,
            DayTarget::Exact(Weekday::Mon),
            DateShiftRule::Occurrence(i64::MIN),
        )
        .is_err());
    }

    #[test]
    fn dateshift_float_integer_conversion_uses_half_open_i64_bounds() {
        const TWO_TO_63: f64 = 9_223_372_036_854_775_808.0;

        assert!(exact_integer_values(&Value::Num(TWO_TO_63), "rule").is_err());
        assert_eq!(
            exact_integer_values(&Value::Num(-TWO_TO_63), "rule")
                .expect("exact i64 minimum")
                .0,
            vec![i64::MIN]
        );

        let values = Value::Tensor(Tensor::new(vec![1.0, TWO_TO_63], vec![1, 2]).unwrap());
        assert!(exact_integer_values(&values, "rule").is_err());
    }

    #[test]
    fn dateshift_boundary_helpers_report_chrono_limits_without_panicking() {
        let maximum = midnight(NaiveDate::MAX);
        assert!(next_unit_start(maximum, DateShiftUnit::Day).is_err());
        assert!(unit_end(maximum, DateShiftUnit::Day).is_err());

        let maximum_year_start = midnight(
            NaiveDate::from_ymd_opt(NaiveDate::MAX.year(), 1, 1)
                .expect("start of Chrono's maximum year"),
        );
        assert!(next_unit_start(maximum_year_start, DateShiftUnit::Year).is_err());
        assert!(unit_end(maximum_year_start, DateShiftUnit::Year).is_err());

        let origin = NaiveDate::from_ymd_opt(2024, 3, 14)
            .unwrap()
            .and_hms_opt(12, 30, 0)
            .unwrap();
        for unit in [
            DateShiftUnit::Week,
            DateShiftUnit::Day,
            DateShiftUnit::Hour,
            DateShiftUnit::Minute,
            DateShiftUnit::Second,
        ] {
            assert!(unit_step(origin, unit, i64::MAX).is_err());
            assert!(unit_step(origin, unit, i64::MIN).is_err());
        }

        let minimum = midnight(NaiveDate::MIN);
        let current = minimum.weekday().num_days_from_monday() as i64;
        let previous_weekday = [
            Weekday::Mon,
            Weekday::Tue,
            Weekday::Wed,
            Weekday::Thu,
            Weekday::Fri,
            Weekday::Sat,
            Weekday::Sun,
        ]
        .into_iter()
        .find(|weekday| (current - weekday.num_days_from_monday() as i64).rem_euclid(7) == 1)
        .expect("a weekday immediately precedes the minimum date");
        assert!(start_of_week(minimum, previous_weekday).is_err());
    }

    #[test]
    fn dateshift_capability_separates_public_form_from_typed_coverage_evidence() {
        let input = &DATESHIFT_INTEGER_INPUTS[0];
        assert_eq!(
            input.availability,
            BuiltinIntegerInputAvailability::Documented
        );
        assert_eq!(input.classes.len(), 8);
        assert!(input.notes.contains("without a per-storage-class table"));
        assert!(input.notes.contains("settled compatibility coverage"));
    }

    #[test]
    fn datetime_date_vectors_normalize_and_day_supports_modern_and_legacy_forms() {
        let datetime = run_datetime(vec![integer_tensor(
            runmat_value::IntegerStorage::I64(vec![2024, 13, 1]),
            vec![1, 3],
        )]);
        assert_eq!(
            datetime_string_array(&datetime).unwrap().unwrap().data,
            vec!["01-Jan-2025".to_string()]
        );
        assert_eq!(
            futures::executor::block_on(day_builtin(
                datetime.clone(),
                vec![Value::from("dayofyear")],
            ))
            .unwrap(),
            Value::Num(1.0)
        );
        let names =
            futures::executor::block_on(day_builtin(datetime, vec![Value::from("shortname")]))
                .unwrap();
        let Value::Cell(names) = names else {
            panic!("expected cell names")
        };
        let Value::CharArray(name) = &names.data[0] else {
            panic!("expected char name")
        };
        assert_eq!(name.data.iter().collect::<String>(), "Wed");
        assert_eq!(
            futures::executor::block_on(day_builtin(
                Value::from("2021/28/09"),
                vec![Value::from("yyyy/dd/mm")],
            ))
            .unwrap(),
            Value::Num(28.0)
        );
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
            Value::Cell(runmat_value::CellArray::new(vec![Value::Num(2.0)], 1, 1).unwrap());
        let indexed =
            futures::executor::block_on(datetime_subsref(value.clone(), "()".to_string(), payload))
                .expect("subsref");
        let year = futures::executor::block_on(year_builtin(indexed, Vec::new())).expect("year");
        assert_eq!(year, Value::Num(2025.0));

        let lhs = run_datetime(vec![Value::Num(2024.0), Value::Num(1.0), Value::Num(1.0)]);
        let rhs = run_datetime(vec![Value::Num(2024.0), Value::Num(1.0), Value::Num(2.0)]);
        let cmp = futures::executor::block_on(datetime_lt(lhs, rhs)).expect("lt");
        assert_eq!(cmp, Value::Num(1.0));
    }

    #[test]
    fn datetime_typed_integer_index_selectors_are_exact() {
        let years = integer_tensor(
            runmat_value::IntegerStorage::U16(vec![2024, 2025]),
            vec![1, 2],
        );
        let months = integer_tensor(runmat_value::IntegerStorage::U8(vec![1, 6]), vec![1, 2]);
        let days = integer_tensor(runmat_value::IntegerStorage::U8(vec![15, 20]), vec![1, 2]);
        let value = run_datetime(vec![years, months, days]);
        let payload = Value::Cell(
            runmat_value::CellArray::new(
                vec![integer_tensor(
                    runmat_value::IntegerStorage::U64(vec![2]),
                    vec![1, 1],
                )],
                1,
                1,
            )
            .unwrap(),
        );
        let indexed =
            futures::executor::block_on(datetime_subsref(value, "()".to_string(), payload))
                .expect("subsref");
        let year = futures::executor::block_on(year_builtin(indexed, Vec::new())).expect("year");
        assert_eq!(year, Value::Num(2025.0));
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
        assert_eq!(&date_vector.materialize_f64()[..3], &[2024.0, 3.0, 14.0]);

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
    fn datenum_typed_integer_date_vector_reads_exact_storage() {
        let serial = serial_for_date(2024, 3, 14);
        let typed_date_vector = Tensor::new_integer(
            runmat_value::IntegerStorage::U16(vec![2024, 3, 14]),
            vec![1, 3],
        )
        .expect("typed date vector");
        let typed_round_trip =
            futures::executor::block_on(datenum_builtin(vec![Value::Tensor(typed_date_vector)]))
                .expect("datenum typed date vector");
        assert_eq!(typed_round_trip, Value::Num(serial));
    }

    #[test]
    fn datenum_typed_integer_serials_read_exact_storage() {
        let serial = serial_for_date(2024, 3, 14).floor() as u32;
        let scalar =
            Tensor::new_integer(runmat_value::IntegerStorage::U32(vec![serial]), vec![1, 1])
                .expect("typed serial");
        let scalar_out = futures::executor::block_on(datenum_builtin(vec![Value::Tensor(scalar)]))
            .expect("datenum typed scalar serial");
        assert_eq!(scalar_out, Value::Num(f64::from(serial)));

        let vector = Tensor::new_integer(
            runmat_value::IntegerStorage::U32(vec![serial, serial + 1]),
            vec![1, 2],
        )
        .expect("typed serial vector");
        let vector_out = futures::executor::block_on(datenum_builtin(vec![Value::Tensor(vector)]))
            .expect("datenum typed vector serial");
        let Value::Tensor(vector_out) = vector_out else {
            panic!("expected datenum vector tensor");
        };
        assert_eq!(vector_out.shape, vec![1, 2]);
        assert_eq!(
            vector_out.materialize_f64(),
            vec![f64::from(serial), f64::from(serial + 1)]
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
        assert_eq!(months.materialize_f64(), vec![14.0]);
        assert_eq!(days.materialize_f64(), vec![3.0]);

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
        assert_eq!(mask.materialize_f64(), vec![0.0, 1.0, 0.0]);

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
        assert_eq!(business_days.materialize_f64(), vec![friday, friday + 3.0]);

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
        let typed_basis =
            Tensor::new_integer(runmat_value::IntegerStorage::U8(vec![1]), vec![1, 1])
                .expect("basis");
        assert_eq!(
            futures::executor::block_on(daysdif_builtin(
                Value::Num(serial_for_date(2024, 1, 30)),
                Value::Num(serial_for_date(2024, 2, 29)),
                vec![Value::Tensor(typed_basis)],
            ))
            .expect("daysdif typed basis"),
            Value::Num(29.0)
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
        assert!(serials
            .materialize_f64()
            .contains(&serial_for_date(2024, 1, 1)));

        let typed_year =
            Tensor::new_integer(runmat_value::IntegerStorage::U16(vec![2024]), vec![1, 1]).unwrap();
        let holidays =
            futures::executor::block_on(holidays_builtin(vec![Value::Tensor(typed_year)]))
                .expect("holidays from typed year");
        let serials = serials_from_datetime_value(&holidays).expect("holiday serials");
        assert!(serials
            .materialize_f64()
            .contains(&serial_for_date(2024, 1, 1)));
    }
}
