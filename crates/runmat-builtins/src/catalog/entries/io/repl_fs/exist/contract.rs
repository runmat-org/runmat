use crate::*;

const EXIST_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "typeID",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Existence code (0,1,2,3,4,5,6,7,8) following MATLAB semantics.",
}];
const EXIST_INPUTS_NAME: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Variable/function/file/class name to query.",
}];
const EXIST_INPUTS_NAME_TYPE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Variable/function/file/class name to query.",
    },
    BuiltinParamDescriptor {
        name: "type",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Documented query kind: builtin|class|dir|file|var.",
    },
];
const EXIST_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "exist name",
        inputs: &EXIST_INPUTS_NAME,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "exist name searchType",
        inputs: &EXIST_INPUTS_NAME_TYPE,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "typeID = exist(name)",
        inputs: &EXIST_INPUTS_NAME,
        outputs: &EXIST_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "typeID = exist(name, searchType)",
        inputs: &EXIST_INPUTS_NAME_TYPE,
        outputs: &EXIST_OUTPUT,
    },
];
pub const EXIST_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EXIST.TOO_MANY_INPUTS",
    identifier: Some("RunMat:exist:TooManyInputs"),
    when: "More than two total input arguments are provided.",
    message: "exist: too many input arguments",
};
pub const EXIST_ERROR_NAME_ARG: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EXIST.NAME_ARG",
    identifier: Some("RunMat:exist:InvalidName"),
    when: "Name input is not a character vector or string scalar/array scalar.",
    message: "exist: name must be a character vector or string scalar",
};
pub const EXIST_ERROR_TYPE_ARG: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EXIST.TYPE_ARG",
    identifier: Some("RunMat:exist:InvalidTypeArgument"),
    when: "Type input is not a character vector or string scalar/array scalar.",
    message: "exist: type must be a character vector or string scalar",
};
pub const EXIST_ERROR_INVALID_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EXIST.INVALID_TYPE",
    identifier: Some("RunMat:exist:InvalidType"),
    when: "Type input is not one of the supported exist query types.",
    message: "exist: invalid type. Type must be one of 'builtin', 'class', 'dir', 'file', or 'var'",
};
const EXIST_ERRORS: [BuiltinErrorDescriptor; 5] = [
    EXIST_ERROR_TOO_MANY_INPUTS,
    EXIST_ERROR_NAME_ARG,
    EXIST_ERROR_TYPE_ARG,
    EXIST_ERROR_INVALID_TYPE,
    EXIST_ERROR_TOO_MANY_OUTPUTS,
];
pub const EXIST_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &EXIST_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &EXIST_ERRORS,
};

pub const EXIST_SEARCH_TYPE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "exist-search-type-extension",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "exist selectors and aliases beyond builtin, class, dir, file, and var are RunMat extensions",
    error_identifier: Some("RunMat:compatibility:ExistSearchTypeExtension"),
};
pub const EXIST_NONTEXT_HANDLE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "exist-nontext-handle-query",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "exist(handleObject, 'handle') with a nontext first argument is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ExistNontextHandleExtension"),
};
pub const EXIST_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [EXIST_SEARCH_TYPE_EXTENSION, EXIST_NONTEXT_HANDLE_EXTENSION];
pub const EXIST_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Documented name and searchType arguments are text. Integer and logical names reject in ordinary queries; the RunMat-only handle query accepts nontext values and reports zero for nonhandles. Resident numeric inputs reject before provider access. The integer-valued typeID remains a host double scalar, not typed-integer storage.",
};

pub const EXIST_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EXIST.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:exist:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "exist: too many output arguments",
};
