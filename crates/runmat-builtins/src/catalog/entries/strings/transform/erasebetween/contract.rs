use crate::*;

pub const ERASE_BETWEEN_FULL_BROADCAST_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "erasebetween-full-broadcast",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "eraseBetween with non-scalar boundary expansion is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:EraseBetweenFullBroadcastExtension"),
    };
pub const ERASE_BETWEEN_RESIDENT_POSITION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "erasebetween-resident-position",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "eraseBetween with GPU-resident position arrays is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:EraseBetweenResidentPositionExtension"),
    };
pub const ERASE_BETWEEN_CHAR_MATRIX_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "erasebetween-char-matrix",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "eraseBetween row-wise character-matrix input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:EraseBetweenCharMatrixExtension"),
    };
pub const ERASE_BETWEEN_STRING_CELL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "erasebetween-string-cell",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "eraseBetween cells containing string scalars are a RunMat extension",
        error_identifier: Some("RunMat:compatibility:EraseBetweenStringCellExtension"),
    };
pub const ERASE_BETWEEN_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    ERASE_BETWEEN_FULL_BROADCAST_EXTENSION,
    ERASE_BETWEEN_RESIDENT_POSITION_EXTENSION,
    ERASE_BETWEEN_CHAR_MATRIX_EXTENSION,
    ERASE_BETWEEN_STRING_CELL_EXTENSION,
];

const INTEGER_POSITION_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "startPos", classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The public numeric position array accepts positive integer values; typed storage is parsed exactly with one-based indexing.",
    },
    BuiltinIntegerInputCapability {
        name: "endPos", classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The public numeric position array accepts positive integer values; typed storage is parsed exactly with one-based indexing.",
    },
];
const INTEGER_TEXT_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "str",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Rejected,
    scalar_double: BuiltinIntegerScalarDoubleRule::Rejected,
    notes:
        "The public first argument is text; integer data is rejected before any provider access.",
}];
pub const ERASE_BETWEEN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "newStr = eraseBetween(str, integer_startPos, integer_endPos)",
        inputs: &INTEGER_POSITION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
        notes: "Positions are exact structural controls; strict compatibility accepts scalar positions or arrays the same size as str. In RunMat mode a resident numeric control may gather at this host-text boundary after extension admission.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "newStr = eraseBetween(integer_str, start, end)",
        inputs: &INTEGER_TEXT_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Integer text input is outside the public text domain and is rejected without gathering.",
    },
];

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "newText",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Text with between-boundary content erased, preserving text container semantics.",
}];
const INPUTS_BASE: [BuiltinParamDescriptor; 3] = [
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
const INPUTS_NAME_VALUE: [BuiltinParamDescriptor; 5] = [
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
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "newText = eraseBetween(str, start, end)",
        inputs: &INPUTS_BASE,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "newText = eraseBetween(str, start, end, Name, Value, ...)",
        inputs: &INPUTS_NAME_VALUE,
        outputs: &OUTPUTS,
    },
];

pub const ERASE_BETWEEN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERASE_BETWEEN.INVALID_INPUT", identifier: Some("RunMat:eraseBetween:InvalidInput"),
    when: "First argument is not a string array, character array, or cell array of text scalars.",
    message: "eraseBetween: first argument must be a string array, character array, or cell array of character vectors",
};
pub const ERASE_BETWEEN_ERROR_BOUNDARY_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERASE_BETWEEN.BOUNDARY_TYPE",
    identifier: Some("RunMat:eraseBetween:BoundaryType"),
    when: "Start/end boundaries are mixed text/numeric domains or use unsupported boundary types.",
    message: "eraseBetween: start and end arguments must both be text or both be numeric positions",
};
pub const ERASE_BETWEEN_ERROR_POSITION_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERASE_BETWEEN.POSITION_TYPE",
    identifier: Some("RunMat:eraseBetween:PositionType"),
    when: "Numeric boundary positions are not positive finite integers.",
    message: "eraseBetween: position arguments must be positive integers",
};
pub const ERASE_BETWEEN_ERROR_NAME_VALUE_PAIR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERASE_BETWEEN.NAME_VALUE_PAIR",
    identifier: Some("RunMat:eraseBetween:NameValuePair"),
    when: "Name/value options are not supplied in complete pairs.",
    message: "eraseBetween: name-value arguments must appear in pairs",
};
pub const ERASE_BETWEEN_ERROR_OPTION_NAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERASE_BETWEEN.OPTION_NAME",
    identifier: Some("RunMat:eraseBetween:OptionName"),
    when: "An option name other than `Boundaries` was supplied.",
    message: "eraseBetween: unrecognized parameter name",
};
pub const ERASE_BETWEEN_ERROR_OPTION_VALUE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERASE_BETWEEN.OPTION_VALUE",
    identifier: Some("RunMat:eraseBetween:OptionValue"),
    when: "`Boundaries` option value is not `inclusive` or `exclusive`.",
    message: "eraseBetween: 'Boundaries' must be either 'inclusive' or 'exclusive'",
};
pub const ERASE_BETWEEN_ERROR_CELL_ELEMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERASE_BETWEEN.CELL_ELEMENT",
    identifier: Some("RunMat:eraseBetween:CellElement"),
    when: "Cell text input/boundary contains non-text values or non-row char arrays.",
    message: "eraseBetween: cell array elements must be string scalars or character vectors",
};
pub const ERASE_BETWEEN_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERASE_BETWEEN.SIZE_MISMATCH",
    identifier: Some("RunMat:eraseBetween:SizeMismatch"),
    when: "Text/boundary inputs are not broadcast-compatible for erase semantics.",
    message: "eraseBetween: boundary sizes must be compatible with the text input",
};
pub const ERASE_BETWEEN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERASE_BETWEEN.INTERNAL",
    identifier: Some("RunMat:eraseBetween:InternalError"),
    when: "Internal output construction failed.",
    message: "eraseBetween: internal error",
};
pub const ERASE_BETWEEN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        ERASE_BETWEEN_ERROR_INVALID_INPUT,
        ERASE_BETWEEN_ERROR_BOUNDARY_TYPE,
        ERASE_BETWEEN_ERROR_POSITION_TYPE,
        ERASE_BETWEEN_ERROR_NAME_VALUE_PAIR,
        ERASE_BETWEEN_ERROR_OPTION_NAME,
        ERASE_BETWEEN_ERROR_OPTION_VALUE,
        ERASE_BETWEEN_ERROR_CELL_ELEMENT,
        ERASE_BETWEEN_ERROR_SIZE_MISMATCH,
        ERASE_BETWEEN_ERROR_INTERNAL,
    ],
};
