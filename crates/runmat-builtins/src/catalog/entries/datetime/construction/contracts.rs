use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "t",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host datetime object array.",
}];
const TEXT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "textOrArray",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Date/time string, string array, character array, or cell array of character vectors.",
}];
const DATE_VECTORS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "dateVectors",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "m-by-3 or m-by-6 numeric date-vector matrix.",
}];
const YEAR: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "year",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Calendar year.",
};
const MONTH: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "month",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Calendar month.",
};
const DAY: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "day",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Calendar day.",
};
const HOUR: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "hour",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Hour component.",
};
const MINUTE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "minute",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Minute component.",
};
const SECOND: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "second",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Second component; may be fractional.",
};
const MILLISECOND: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "millisecond",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Millisecond component.",
};
const COMPONENTS_THREE: [BuiltinParamDescriptor; 3] = [YEAR, MONTH, DAY];
const COMPONENTS_SIX: [BuiltinParamDescriptor; 6] = [YEAR, MONTH, DAY, HOUR, MINUTE, SECOND];
const COMPONENTS_SEVEN: [BuiltinParamDescriptor; 7] =
    [YEAR, MONTH, DAY, HOUR, MINUTE, SECOND, MILLISECOND];
const SERIAL_CONVERSION: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "serialDateNumbers",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Serial date numbers to convert.",
    },
    BuiltinParamDescriptor {
        name: "ConvertFrom",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Conversion option name.",
    },
    BuiltinParamDescriptor {
        name: "dateType",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Currently only 'datenum' is implemented.",
    },
];
const WITH_OPTIONS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Constructor arguments followed by Format, InputFormat, or ConvertFrom pairs.",
}];

pub const DATETIME_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DATETIME.INVALID_ARGUMENT",
    identifier: Some("RunMat:datetime:InvalidArgument"),
    when: "Arguments or option grammar do not match supported datetime forms.",
    message: "datetime: invalid argument",
};
pub const DATETIME_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DATETIME.INVALID_INPUT",
    identifier: Some("RunMat:datetime:InvalidInput"),
    when: "Input values cannot be parsed, converted, or broadcast to a valid datetime result.",
    message: "datetime: invalid input",
};
pub const DATETIME_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DATETIME.INTERNAL",
    identifier: Some("RunMat:datetime:Internal"),
    when: "Internal datetime state is invalid.",
    message: "datetime: internal operation failed",
};
pub const DATETIME_ERRORS: [BuiltinErrorDescriptor; 3] = [
    DATETIME_ERROR_INVALID_ARGUMENT,
    DATETIME_ERROR_INVALID_INPUT,
    DATETIME_ERROR_INTERNAL,
];

pub const DATETIME_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "t = datetime()",
            inputs: &[],
            outputs: &OUTPUT,
        },
        BuiltinSignatureDescriptor {
            label: "t = datetime(textOrArray)",
            inputs: &TEXT,
            outputs: &OUTPUT,
        },
        BuiltinSignatureDescriptor {
            label: "t = datetime(dateVectors)",
            inputs: &DATE_VECTORS,
            outputs: &OUTPUT,
        },
        BuiltinSignatureDescriptor {
            label: "t = datetime(year, month, day)",
            inputs: &COMPONENTS_THREE,
            outputs: &OUTPUT,
        },
        BuiltinSignatureDescriptor {
            label: "t = datetime(year, month, day, hour, minute, second)",
            inputs: &COMPONENTS_SIX,
            outputs: &OUTPUT,
        },
        BuiltinSignatureDescriptor {
            label: "t = datetime(year, month, day, hour, minute, second, millisecond)",
            inputs: &COMPONENTS_SEVEN,
            outputs: &OUTPUT,
        },
        BuiltinSignatureDescriptor {
            label: "t = datetime(serialDateNumbers, \"ConvertFrom\", \"datenum\")",
            inputs: &SERIAL_CONVERSION,
            outputs: &OUTPUT,
        },
        BuiltinSignatureDescriptor {
            label: "t = datetime(___, \"Format\", format)",
            inputs: &WITH_OPTIONS,
            outputs: &OUTPUT,
        },
        BuiltinSignatureDescriptor {
            label: "t = datetime(textOrArray, \"InputFormat\", inputFormat)",
            inputs: &WITH_OPTIONS,
            outputs: &OUTPUT,
        },
        BuiltinSignatureDescriptor {
            label: "t = datetime(___, Name, Value, ...)",
            inputs: &WITH_OPTIONS,
            outputs: &OUTPUT,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DATETIME_ERRORS,
};

const INTEGER_COMPONENT_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "date vectors and calendar components",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight classes are read from authoritative storage and range-checked before the host serial-date boundary.",
    }];
const INTEGER_CONVERT_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X with ConvertFrom",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Integer conversion input is documented; RunMat currently implements datenum only.",
}];
const INTEGER_RESIDENT_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "resident numeric input",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "MATLAB mode rejects resident input before provider access; RunMat mode gathers it.",
    }];

pub const DATETIME_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "datetime(integer_date_vector_or_components)", inputs: &INTEGER_COMPONENT_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Civil components are resolved from exact integer storage before the representable instant crosses into the floating serial-date representation.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "datetime(integer_X, 'ConvertFrom', dateType)", inputs: &INTEGER_CONVERT_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Only the datenum conversion epoch is implemented; other epochs reject explicitly.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "datetime(resident_integer, ...)", inputs: &INTEGER_RESIDENT_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The result is always a host datetime object, never a resident datetime array.",
    },
];
