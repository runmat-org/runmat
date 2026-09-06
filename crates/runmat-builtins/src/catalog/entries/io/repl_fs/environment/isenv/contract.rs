use crate::*;

const NAME: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "name",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Character row, string scalar or array, or cell array of character rows.",
};
const EXISTS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical result with the name container's shape.",
};
const SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "tf = isenv(name)",
    inputs: &[NAME],
    outputs: &[EXISTS],
}];
pub const ISENV_ERROR_ARITY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISENV.ARITY",
    identifier: Some("RunMat:isenv:Arity"),
    when: "The call does not have exactly one input.",
    message: "isenv: expected exactly one input",
};
pub const ISENV_ERROR_INVALID_NAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor { code: "RM.ISENV.INVALID_NAME", identifier: Some("RunMat:isenv:InvalidName"), when: "The name is not a supported text scalar or container.", message: "isenv: name must be a character vector, string scalar or array, or cell array of character vectors" };
pub const ISENV_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[ISENV_ERROR_ARITY, ISENV_ERROR_INVALID_NAME],
};
pub const ISENV_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor { kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None, notes: "Environment names are text. Integer and provider-resident numeric values reject before provider or environment access." };
