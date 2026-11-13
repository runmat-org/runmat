use crate::*;

const READMATRIX_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "M",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Imported matrix output (numeric/logical, and may preserve gpuArray residency with Like).",
}];
const READMATRIX_INPUTS_FILENAME: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Delimited text file path.",
}];
const READMATRIX_INPUTS_FILENAME_OPTIONS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Delimited text file path.",
    },
    BuiltinParamDescriptor {
        name: "options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Options struct containing readmatrix name/value fields.",
    },
];
const READMATRIX_INPUTS_FILENAME_NAME_VALUE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Delimited text file path.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Option name.",
    },
    BuiltinParamDescriptor {
        name: "optionValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Value for the preceding option name.",
    },
];
const READMATRIX_INPUTS_FILENAME_OPTIONS_NAME_VALUE: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Delimited text file path.",
    },
    BuiltinParamDescriptor {
        name: "options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Options struct containing readmatrix name/value fields.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Option name.",
    },
    BuiltinParamDescriptor {
        name: "optionValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Value for the preceding option name.",
    },
];
const READMATRIX_INPUTS_FILENAME_NAME_VALUE_PAIRS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Delimited text file path.",
    },
    BuiltinParamDescriptor {
        name: "nameValuePairs...",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value option pairs.",
    },
];
const READMATRIX_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "M = readmatrix(filename)",
        inputs: &READMATRIX_INPUTS_FILENAME,
        outputs: &READMATRIX_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = readmatrix(filename, options)",
        inputs: &READMATRIX_INPUTS_FILENAME_OPTIONS,
        outputs: &READMATRIX_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = readmatrix(filename, name, optionValue)",
        inputs: &READMATRIX_INPUTS_FILENAME_NAME_VALUE,
        outputs: &READMATRIX_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = readmatrix(filename, options, name, optionValue)",
        inputs: &READMATRIX_INPUTS_FILENAME_OPTIONS_NAME_VALUE,
        outputs: &READMATRIX_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = readmatrix(filename, nameValuePairs...)",
        inputs: &READMATRIX_INPUTS_FILENAME_NAME_VALUE_PAIRS,
        outputs: &READMATRIX_OUTPUT,
    },
];
pub const READMATRIX_ERROR_ARG_CONFIG: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READMATRIX.ARG_CONFIG",
    identifier: None,
    when: "Argument list is malformed for readmatrix call forms.",
    message: "readmatrix: invalid argument configuration",
};
pub const READMATRIX_ERROR_OPTION_NAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READMATRIX.OPTION_NAME",
    identifier: None,
    when: "Option name is not a scalar string/char vector.",
    message: "readmatrix: invalid option name",
};
pub const READMATRIX_ERROR_OPTION_VALUE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READMATRIX.OPTION_VALUE",
    identifier: None,
    when: "A provided option value is invalid.",
    message: "readmatrix: invalid option value",
};
pub const READMATRIX_ERROR_RANGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READMATRIX.RANGE",
    identifier: None,
    when: "Range specification is malformed or semantically invalid.",
    message: "readmatrix: invalid range",
};
pub const READMATRIX_ERROR_FILENAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READMATRIX.FILENAME",
    identifier: None,
    when: "Filename argument is not a scalar string/char vector.",
    message: "readmatrix: invalid filename input",
};
pub const READMATRIX_ERROR_FILENAME_EMPTY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READMATRIX.FILENAME_EMPTY",
    identifier: None,
    when: "Filename resolves to an empty path.",
    message: "readmatrix: filename must not be empty",
};
pub const READMATRIX_ERROR_IO_OPEN: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READMATRIX.IO_OPEN",
    identifier: None,
    when: "Input file cannot be opened.",
    message: "readmatrix: unable to read file",
};
pub const READMATRIX_ERROR_IO_READ: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READMATRIX.IO_READ",
    identifier: None,
    when: "Input file cannot be read.",
    message: "readmatrix: failed to read file",
};
pub const READMATRIX_ERROR_NON_NUMERIC_TOKEN: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READMATRIX.NON_NUMERIC_TOKEN",
    identifier: None,
    when: "A token cannot be parsed as numeric in the selected import region.",
    message: "readmatrix: nonnumeric token encountered",
};
pub const READMATRIX_ERROR_OUTPUT_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READMATRIX.OUTPUT_TYPE",
    identifier: None,
    when: "OutputType/Like output coercion configuration is invalid.",
    message: "readmatrix: invalid output type configuration",
};
pub const READMATRIX_ERROR_TENSOR_BUILD: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READMATRIX.TENSOR_BUILD",
    identifier: None,
    when: "Internal tensor/logical output materialization fails.",
    message: "readmatrix: unable to construct output matrix",
};
const READMATRIX_ERRORS: [BuiltinErrorDescriptor; 11] = [
    READMATRIX_ERROR_ARG_CONFIG,
    READMATRIX_ERROR_OPTION_NAME,
    READMATRIX_ERROR_OPTION_VALUE,
    READMATRIX_ERROR_RANGE,
    READMATRIX_ERROR_FILENAME,
    READMATRIX_ERROR_FILENAME_EMPTY,
    READMATRIX_ERROR_IO_OPEN,
    READMATRIX_ERROR_IO_READ,
    READMATRIX_ERROR_NON_NUMERIC_TOKEN,
    READMATRIX_ERROR_OUTPUT_TYPE,
    READMATRIX_ERROR_TENSOR_BUILD,
];
pub const READMATRIX_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &READMATRIX_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &READMATRIX_ERRORS,
};

pub const READMATRIX_LIKE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "readmatrix-like-output",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "readmatrix accepts a Like prototype and optional resident result as a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ReadmatrixLikeOutputExtension"),
};
pub const READMATRIX_TYPED_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "readmatrix-typed-integer-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "readmatrix accepts typed-integer range, header, size, empty-value, and related numeric controls as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ReadmatrixTypedIntegerControlExtension"),
    };
pub const READMATRIX_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    READMATRIX_LIKE_EXTENSION,
    READMATRIX_TYPED_INTEGER_CONTROL_EXTENSION,
];
const READMATRIX_INTEGER_OUTPUT_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "OutputType integer class selector",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented OutputType selector names any of the eight integer classes. Decimal integer tokens are parsed directly into native storage, including exact int64 and uint64 endpoints, while fractional and exceptional values follow saturating cast rules.",
    }];
const READMATRIX_INTEGER_CONTROL_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Range, NumHeaderLines, ExpectedNumVariables, EmptyValue, or related numeric option",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target's public datatype tables constrain implemented numeric controls to single/double. RunMat mode retains the typed-integer extension and reads structural values exactly; compatibility mode rejects before gather or file access.",
    }];
const READMATRIX_INTEGER_LIKE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Like integer prototype",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Like is not a documented readmatrix option. RunMat mode uses the exact prototype class and owner; integer text is parsed directly in that class rather than through binary64.",
    }];
pub const READMATRIX_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "M = readmatrix(filename, 'OutputType', integer_class)",
        inputs: &READMATRIX_INTEGER_OUTPUT_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "File I/O and parsing are host operations. Every integer OutputType produces the requested integer class and preserves exactly representable parsed values.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "M = readmatrix(filename, typed_integer_controls...)",
        inputs: &READMATRIX_INTEGER_CONTROL_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed controls are an independently gated extension; ordinary automatic residency may gather transparently for host file I/O.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "M = readmatrix(filename, 'Like', integer_prototype)",
        inputs: &READMATRIX_INTEGER_LIKE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Compatibility mode rejects the whole Like extension. RunMat mode preserves host class and attempts explicit owner-resident restoration only when the provider can represent the exact class.",
    },
];
