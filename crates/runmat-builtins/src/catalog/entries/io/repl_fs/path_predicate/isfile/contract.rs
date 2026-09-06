use crate::*;

const PATH: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "String array, character row, or cell array of character rows.",
};
const RESULT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical result with the input container's shape.",
};
const SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "tf = isfile(filename)",
    inputs: &[PATH],
    outputs: &[RESULT],
}];

pub const ISFILE_ERROR_ARITY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISFILE.ARITY",
    identifier: Some("RunMat:isfile:Arity"),
    when: "The call does not have exactly one input.",
    message: "isfile: expected exactly one input argument",
};
pub const ISFILE_ERROR_PATH: BuiltinErrorDescriptor = BuiltinErrorDescriptor { code: "RM.ISFILE.PATH", identifier: Some("RunMat:isfile:InvalidPath"), when: "The input is not a string array, character row, or cell array of character rows.", message: "isfile: filename must be a string array, character vector, or cell array of character vectors" };
pub const ISFILE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[ISFILE_ERROR_ARITY, ISFILE_ERROR_PATH],
};
pub const ISFILE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor { kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None, notes: "File names are text. Integer and provider-resident numeric values reject before provider or filesystem access." };
