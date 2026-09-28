use crate::*;

pub const EXTRACT_BETWEEN_FULL_BROADCAST_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "extractbetween-full-broadcast",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "extractBetween with non-scalar boundary expansion is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ExtractBetweenFullBroadcastExtension"),
    };
pub const EXTRACT_BETWEEN_RESIDENT_POSITION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "extractbetween-resident-position",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "extractBetween with resident numeric positions is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ExtractBetweenResidentPositionExtension"),
    };
pub const EXTRACT_BETWEEN_CHAR_MATRIX_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "extractbetween-char-matrix",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "extractBetween row-wise character-matrix input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ExtractBetweenCharMatrixExtension"),
    };
pub const EXTRACT_BETWEEN_STRING_CELL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "extractbetween-string-cell",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "extractBetween cells containing string scalars are a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ExtractBetweenStringCellExtension"),
    };
pub const EXTRACT_BETWEEN_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    EXTRACT_BETWEEN_FULL_BROADCAST_EXTENSION,
    EXTRACT_BETWEEN_RESIDENT_POSITION_EXTENSION,
    EXTRACT_BETWEEN_CHAR_MATRIX_EXTENSION,
    EXTRACT_BETWEEN_STRING_CELL_EXTENSION,
];
const INTEGER_POSITION_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "startPos",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The public numeric position accepts every built-in integer class and is read exactly with one-based indexing.",
    },
    BuiltinIntegerInputCapability {
        name: "endPos",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The public numeric position accepts every built-in integer class and is read exactly with one-based indexing.",
    },
];
const INTEGER_TEXT_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "str",
    classes: &[],
    availability: BuiltinIntegerInputAvailability::Rejected,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "The first argument is text; integer data rejects before provider access.",
}];
pub const EXTRACT_BETWEEN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "newStr = extractBetween(str, integer_startPos, integer_endPos)",
        inputs: &INTEGER_POSITION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
        notes: "Strict compatibility accepts scalar positions or arrays exactly the same size as str; string input returns string and other documented text input returns cellstr.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "newStr = extractBetween(integer_str, start, end)",
        inputs: &INTEGER_TEXT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Integer text input is outside the public domain and rejects without numeric-to-text conversion.",
    },
];

const EXTRACT_BETWEEN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "newText",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Extracted text preserving scalar/array/cell text container semantics.",
}];

const EXTRACT_BETWEEN_INPUTS_BASE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text scalar/array/cell.",
    },
    BuiltinParamDescriptor {
        name: "start",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Start boundary marker text or positive integer position(s).",
    },
    BuiltinParamDescriptor {
        name: "end",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "End boundary marker text or positive integer position(s).",
    },
];

const EXTRACT_BETWEEN_INPUTS_NAME_VALUE: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text scalar/array/cell.",
    },
    BuiltinParamDescriptor {
        name: "start",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Start boundary marker text or positive integer position(s).",
    },
    BuiltinParamDescriptor {
        name: "end",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "End boundary marker text or positive integer position(s).",
    },
    BuiltinParamDescriptor {
        name: "Name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Option name (`Boundaries`).",
    },
    BuiltinParamDescriptor {
        name: "Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option value and additional Name/Value pairs.",
    },
];

const EXTRACT_BETWEEN_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "newText = extractBetween(str, start, end)",
        inputs: &EXTRACT_BETWEEN_INPUTS_BASE,
        outputs: &EXTRACT_BETWEEN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "newText = extractBetween(str, start, end, Name, Value, ...)",
        inputs: &EXTRACT_BETWEEN_INPUTS_NAME_VALUE,
        outputs: &EXTRACT_BETWEEN_OUTPUT,
    },
];

pub const EXTRACT_BETWEEN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EXTRACT_BETWEEN.INVALID_INPUT",
    identifier: Some("RunMat:extractBetween:InvalidInput"),
    when: "First argument is not a string array, character array, or cell array of text scalars.",
    message:
        "extractBetween: first argument must be a string array, character array, or cell array of character vectors",
};

pub const EXTRACT_BETWEEN_ERROR_BOUNDARY_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EXTRACT_BETWEEN.BOUNDARY_TYPE",
    identifier: Some("RunMat:extractBetween:BoundaryType"),
    when: "Start/end boundaries are mixed text/numeric domains or use unsupported boundary types.",
    message:
        "extractBetween: start and end arguments must both be text or both be numeric positions",
};

pub const EXTRACT_BETWEEN_ERROR_POSITION_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EXTRACT_BETWEEN.POSITION_TYPE",
    identifier: Some("RunMat:extractBetween:PositionType"),
    when: "Numeric boundary positions are not positive finite integers.",
    message: "extractBetween: position arguments must be positive integers",
};

pub const EXTRACT_BETWEEN_ERROR_NAME_VALUE_PAIR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EXTRACT_BETWEEN.NAME_VALUE_PAIR",
    identifier: Some("RunMat:extractBetween:NameValuePair"),
    when: "Name/value options are not supplied in complete pairs.",
    message: "extractBetween: name-value arguments must appear in pairs",
};

pub const EXTRACT_BETWEEN_ERROR_OPTION_NAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EXTRACT_BETWEEN.OPTION_NAME",
    identifier: Some("RunMat:extractBetween:OptionName"),
    when: "An option name other than `Boundaries` was supplied.",
    message: "extractBetween: unrecognized parameter name",
};

pub const EXTRACT_BETWEEN_ERROR_OPTION_VALUE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EXTRACT_BETWEEN.OPTION_VALUE",
    identifier: Some("RunMat:extractBetween:OptionValue"),
    when: "`Boundaries` option value is not `inclusive` or `exclusive`.",
    message: "extractBetween: 'Boundaries' must be either 'inclusive' or 'exclusive'",
};

pub const EXTRACT_BETWEEN_ERROR_CELL_ELEMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EXTRACT_BETWEEN.CELL_ELEMENT",
    identifier: Some("RunMat:extractBetween:CellElement"),
    when: "Cell text input/boundary contains non-text values or non-row char arrays.",
    message: "extractBetween: cell array elements must be string scalars or character vectors",
};

pub const EXTRACT_BETWEEN_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EXTRACT_BETWEEN.SIZE_MISMATCH",
    identifier: Some("RunMat:extractBetween:SizeMismatch"),
    when: "Text/boundary inputs are not broadcast-compatible for extraction.",
    message: "extractBetween: boundary sizes must be compatible with the text input",
};

pub const EXTRACT_BETWEEN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EXTRACT_BETWEEN.INTERNAL",
    identifier: Some("RunMat:extractBetween:InternalError"),
    when: "Internal output construction failed.",
    message: "extractBetween: internal error",
};

const EXTRACT_BETWEEN_ERRORS: [BuiltinErrorDescriptor; 9] = [
    EXTRACT_BETWEEN_ERROR_INVALID_INPUT,
    EXTRACT_BETWEEN_ERROR_BOUNDARY_TYPE,
    EXTRACT_BETWEEN_ERROR_POSITION_TYPE,
    EXTRACT_BETWEEN_ERROR_NAME_VALUE_PAIR,
    EXTRACT_BETWEEN_ERROR_OPTION_NAME,
    EXTRACT_BETWEEN_ERROR_OPTION_VALUE,
    EXTRACT_BETWEEN_ERROR_CELL_ELEMENT,
    EXTRACT_BETWEEN_ERROR_SIZE_MISMATCH,
    EXTRACT_BETWEEN_ERROR_INTERNAL,
];

pub const EXTRACT_BETWEEN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &EXTRACT_BETWEEN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &EXTRACT_BETWEEN_ERRORS,
};
