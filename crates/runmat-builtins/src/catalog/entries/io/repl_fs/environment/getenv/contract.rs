use crate::*;

const VALUE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Environment value or values in the input container form.",
};
const ENVIRONMENT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "environment",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Dictionary containing all visible environment names and values.",
};
const NAME: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "name",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Character row, string scalar or array, or cell array of character rows.",
};
const SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "environment = getenv()",
        inputs: &[],
        outputs: &[ENVIRONMENT],
    },
    BuiltinSignatureDescriptor {
        label: "value = getenv(name)",
        inputs: &[NAME],
        outputs: &[VALUE],
    },
];

pub const GETENV_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETENV.TOO_MANY_INPUTS",
    identifier: Some("RunMat:getenv:TooManyInputs"),
    when: "More than one input is supplied.",
    message: "getenv: too many input arguments",
};
pub const GETENV_ERROR_INVALID_NAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor { code: "RM.GETENV.INVALID_NAME", identifier: Some("RunMat:getenv:InvalidName"), when: "The name is not a supported text scalar or container.", message: "getenv: name must be a character vector, string scalar or array, or cell array of character vectors" };
pub const GETENV_ERROR_CELL_ELEMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETENV.CELL_ELEMENT",
    identifier: Some("RunMat:getenv:InvalidCellElement"),
    when: "A name cell contains a value other than a character row.",
    message: "getenv: cell array elements must be character vectors",
};
pub const GETENV_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETENV.INTERNAL",
    identifier: Some("RunMat:getenv:Internal"),
    when: "The environment result cannot be represented.",
    message: "getenv: internal conversion failure",
};
const ERRORS: &[BuiltinErrorDescriptor] = &[
    GETENV_ERROR_TOO_MANY_INPUTS,
    GETENV_ERROR_INVALID_NAME,
    GETENV_ERROR_CELL_ELEMENT,
    GETENV_ERROR_INTERNAL,
];

pub const GETENV_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: ERRORS,
};

pub const GETENV_CHAR_MATRIX_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "getenv-character-matrix-name",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "getenv with a multirow character-matrix name is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GetenvCharacterMatrixNameExtension"),
};
pub const GETENV_CELL_STRING_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "getenv-cell-string-name",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "getenv with string scalars inside a cell-array name is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GetenvCellStringNameExtension"),
};
pub(super) const EXTENSIONS: &[BuiltinExtensionDescriptor] =
    &[GETENV_CHAR_MATRIX_EXTENSION, GETENV_CELL_STRING_EXTENSION];

pub const GETENV_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor { kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None, notes: "Environment names are text. Integer and provider-resident numeric values reject before provider or environment access." };
