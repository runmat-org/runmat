use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "handles",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cell row of visible local function handles.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "handles = localfunctions()",
    inputs: &[],
    outputs: &OUTPUT,
}];
pub const LOCALFUNCTIONS_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOCALFUNCTIONS.TOO_MANY_INPUTS",
    identifier: Some("RunMat:TooManyInputs"),
    when: "Any input argument is provided.",
    message: "localfunctions: too many input arguments",
};
pub const LOCALFUNCTIONS_ERRORS: [BuiltinErrorDescriptor; 1] =
    [LOCALFUNCTIONS_ERROR_TOO_MANY_INPUTS];
pub const LOCALFUNCTIONS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LOCALFUNCTIONS_ERRORS,
};
pub const LOCALFUNCTIONS_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "The function takes no inputs and returns function handles, so no integer operand or result is converted.",
};
