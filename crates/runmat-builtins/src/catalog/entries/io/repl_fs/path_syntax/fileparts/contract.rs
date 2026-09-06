use crate::*;

const INPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Path text as a character row, string array, or cell array of character rows.",
};
const OUTPUTS: &[BuiltinParamDescriptor] = &[
    BuiltinParamDescriptor {
        name: "filepath",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Folder component with the input representation and shape.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Base filename component with the input representation and shape.",
    },
    BuiltinParamDescriptor {
        name: "ext",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description:
            "Extension including its leading dot, with the input representation and shape.",
    },
];
const SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "[filepath, name, ext] = fileparts(filename)",
    inputs: &[INPUT],
    outputs: OUTPUTS,
}];
pub const FILEPARTS_ERROR_ARITY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILEPARTS.ARITY",
    identifier: Some("RunMat:fileparts:InvalidArity"),
    when: "The call does not have exactly one input.",
    message: "fileparts: expected exactly one input argument",
};
pub const FILEPARTS_ERROR_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor { code: "RM.FILEPARTS.TYPE", identifier: Some("RunMat:fileparts:InvalidFilename"), when: "The input is not an admitted text container.", message: "fileparts: filename must be a character vector, string array, or cell array of character vectors" };
pub const FILEPARTS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[FILEPARTS_ERROR_ARITY, FILEPARTS_ERROR_TYPE],
};
pub const FILEPARTS_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "fileparts parses text and has no numeric input role.",
};
