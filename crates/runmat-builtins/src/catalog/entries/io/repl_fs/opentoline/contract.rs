use crate::*;
pub const OPENTOLINE_INTEGER_LINE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "opentoline-integer-line",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "opentoline with a native typed-integer line number is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:OpentolineIntegerLineExtension"),
    };
pub const OPENTOLINE_INTEGER_COLUMN_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "opentoline-integer-column",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "opentoline with a native typed-integer column number is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:OpentolineIntegerColumnExtension"),
    };
pub const OPENTOLINE_RESIDENT_POSITION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "opentoline-resident-position",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "opentoline with an explicit GPU-resident line or column is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:OpentolineResidentPositionExtension"),
    };

pub const OPENTOLINE_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    OPENTOLINE_INTEGER_LINE_EXTENSION,
    OPENTOLINE_INTEGER_COLUMN_EXTENSION,
    OPENTOLINE_RESIDENT_POSITION_EXTENSION,
];

const INTEGER_LINE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "line",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Public evidence establishes a positive line-number control but does not enumerate native typed classes; ordinary integer-valued double remains the compatibility form.",
    }];
const INTEGER_COLUMN_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "column",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Public evidence establishes an optional structural column position but does not enumerate native typed classes; ordinary integer-valued double remains the compatibility form.",
    }];

const fn position_capability(
    form: &'static str,
    inputs: &'static [BuiltinIntegerInputCapability],
) -> BuiltinIntegerCapabilityDescriptor {
    BuiltinIntegerCapabilityDescriptor {
        form,
        inputs,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The exact positive position controls only editor navigation. Typed-class and explicit-residency gates run before gather, file resolution, or side effects; automatic residency remains transparent.",
    }
}

pub const OPENTOLINE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    position_capability(
        "opentoline(filename, integer_line, ...)",
        &INTEGER_LINE_INPUTS,
    ),
    position_capability(
        "opentoline(filename, line, integer_column, ...)",
        &INTEGER_COLUMN_INPUTS,
    ),
];

const FILE_INPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "File to open in the editor.",
};

const LINE_INPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "line",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Positive scalar line number, or scalar text containing a positive integer for command syntax.",
};

const COLUMN_INPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "column",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: Some("1"),
    description: "Positive scalar column number, or scalar text containing a positive integer for command syntax.",
};

const SELECT_INPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "option",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Editor navigation option such as 'select'.",
};

const TWO_INPUTS: [BuiltinParamDescriptor; 2] = [FILE_INPUT, LINE_INPUT];
const THREE_INPUTS: [BuiltinParamDescriptor; 3] = [FILE_INPUT, LINE_INPUT, COLUMN_INPUT];
const FOUR_INPUTS: [BuiltinParamDescriptor; 4] =
    [FILE_INPUT, LINE_INPUT, COLUMN_INPUT, SELECT_INPUT];

const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "opentoline(filename, line)",
        inputs: &TWO_INPUTS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "opentoline(filename, line, column)",
        inputs: &THREE_INPUTS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "opentoline(filename, line, column, option)",
        inputs: &FOUR_INPUTS,
        outputs: &[],
    },
];

pub const OPENTOLINE_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPENTOLINE.ARG_COUNT",
    identifier: Some("RunMat:opentoline:ArgumentCount"),
    when: "The call does not provide two to four input arguments.",
    message: "opentoline: expected filename, line, optional column, and optional option",
};

pub const OPENTOLINE_ERROR_FILENAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPENTOLINE.FILENAME",
    identifier: Some("RunMat:opentoline:InvalidFilename"),
    when: "The filename is not a string scalar or row character vector.",
    message: "opentoline: filename must be a character vector or string scalar",
};

pub const OPENTOLINE_ERROR_POSITION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPENTOLINE.POSITION",
    identifier: Some("RunMat:opentoline:InvalidPosition"),
    when: "The line or column argument is not a positive integer scalar.",
    message: "opentoline: line and column must be positive integer scalars",
};

pub const OPENTOLINE_ERROR_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPENTOLINE.OPTION",
    identifier: Some("RunMat:opentoline:InvalidOption"),
    when: "The optional editor behavior argument is unsupported.",
    message: "opentoline: option must be 'select' when provided",
};

pub const OPENTOLINE_ERROR_NOT_FOUND: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPENTOLINE.NOT_FOUND",
    identifier: Some("RunMat:opentoline:FileNotFound"),
    when: "No file matching the supplied name can be resolved.",
    message: "opentoline: file was not found",
};

pub const OPENTOLINE_ERROR_OUTPUT_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPENTOLINE.OUTPUT_COUNT",
    identifier: Some("RunMat:opentoline:TooManyOutputs"),
    when: "The call requests one or more output arguments.",
    message: "opentoline: expected no output arguments",
};

pub const OPENTOLINE_ERROR_HOST: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPENTOLINE.HOST",
    identifier: Some("RunMat:host:DocumentInteractionUnavailable"),
    when: "The active application host cannot perform editor navigation.",
    message: "The active host does not provide document or editor interaction",
};
const ERRORS: [BuiltinErrorDescriptor; 7] = [
    OPENTOLINE_ERROR_HOST,
    OPENTOLINE_ERROR_ARG_COUNT,
    OPENTOLINE_ERROR_FILENAME,
    OPENTOLINE_ERROR_POSITION,
    OPENTOLINE_ERROR_OPTION,
    OPENTOLINE_ERROR_NOT_FOUND,
    OPENTOLINE_ERROR_OUTPUT_COUNT,
];

pub const OPENTOLINE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
