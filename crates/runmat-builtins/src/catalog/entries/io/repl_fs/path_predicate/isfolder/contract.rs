use crate::*;

const PATH: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "folderName",
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
    label: "tf = isfolder(folderName)",
    inputs: &[PATH],
    outputs: &[RESULT],
}];

pub const ISFOLDER_ERROR_ARITY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISFOLDER.ARITY",
    identifier: Some("RunMat:isfolder:Arity"),
    when: "The call does not have exactly one input.",
    message: "isfolder: expected exactly one input argument",
};
pub const ISFOLDER_ERROR_PATH: BuiltinErrorDescriptor = BuiltinErrorDescriptor { code: "RM.ISFOLDER.PATH", identifier: Some("RunMat:isfolder:InvalidPath"), when: "The input is not a string array, character row, or cell array of character rows.", message: "isfolder: folderName must be a string array, character vector, or cell array of character vectors" };
pub const ISFOLDER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[ISFOLDER_ERROR_ARITY, ISFOLDER_ERROR_PATH],
};
pub const ISFOLDER_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor { kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None, notes: "Folder names are text. Integer and provider-resident numeric values reject before provider or filesystem access." };
