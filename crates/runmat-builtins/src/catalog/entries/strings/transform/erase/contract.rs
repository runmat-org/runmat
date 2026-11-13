use crate::*;

pub const ERASE_CHAR_MATRIX_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "erase-char-matrix",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "erase row-wise character-matrix input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:EraseCharMatrixExtension"),
};
pub const ERASE_STRING_CELL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "erase-string-cell",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "erase cells containing string scalars are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:EraseStringCellExtension"),
};
pub const ERASE_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [ERASE_CHAR_MATRIX_EXTENSION, ERASE_STRING_CELL_EXTENSION];

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "str", classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::Rejected,
        notes: "The public input text accepts string arrays, a character vector, or a cell array of character vectors; numeric values are rejected before provider access.",
    },
    BuiltinIntegerInputCapability {
        name: "match", classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::Rejected,
        notes: "The public match argument accepts text or pattern values; numeric values are rejected before provider access.",
    },
];
pub const ERASE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "newStr = erase(str, match)", inputs: &INTEGER_INPUTS,
    computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
    output_class: BuiltinIntegerOutputClassRule::NotApplicable,
    overflow: BuiltinIntegerOverflowRule::NotApplicable,
    backend: BuiltinIntegerBackendRule::HostOnly,
    overload: BuiltinIntegerOverloadKind::FunctionSpecific,
    notes: "erase is text-only; integer values in either role, including nested resident values, are rejected without gathering.",
}];

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "newStr",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Text with substring occurrences removed, preserving input container kind.",
}];
const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text (string/char/cell).",
    },
    BuiltinParamDescriptor {
        name: "pattern",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Pattern text list (scalar or array/cell).",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "newStr = erase(str, pattern)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

pub const ERASE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERASE.INVALID_INPUT", identifier: Some("RunMat:erase:InvalidInput"),
    when: "First argument is not a string array, char array, or cell array of text scalars.",
    message: "erase: first argument must be a string array, character array, or cell array of character vectors",
};
pub const ERASE_ERROR_PATTERN_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERASE.PATTERN_TYPE", identifier: Some("RunMat:erase:PatternType"),
    when: "Second argument is not a text scalar/array/cell of text scalars.",
    message: "erase: second argument must be a string array, character array, or cell array of character vectors",
};
pub const ERASE_ERROR_CELL_ELEMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERASE.CELL_ELEMENT",
    identifier: Some("RunMat:erase:CellElement"),
    when: "Cell arrays contain non-text elements or non-row char arrays.",
    message: "erase: cell array elements must be string scalars or character vectors",
};
pub const ERASE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERASE.INTERNAL",
    identifier: Some("RunMat:erase:InternalError"),
    when: "Internal output container construction failed.",
    message: "erase: internal error",
};
pub const ERASE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        ERASE_ERROR_INVALID_INPUT,
        ERASE_ERROR_PATTERN_TYPE,
        ERASE_ERROR_CELL_ELEMENT,
        ERASE_ERROR_INTERNAL,
    ],
};
