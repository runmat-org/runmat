use crate::*;

pub const GPU_INFO_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "gpuInfo accepts no inputs and returns a formatted host string; integer values are not an input or result class, and extra arguments reject before provider metadata access.",
};

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "summary",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Formatted text summary for the active provider/device.",
}];
const INPUTS: [BuiltinParamDescriptor; 0] = [];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "summary = gpuInfo()",
    inputs: &INPUTS,
    outputs: &OUTPUT,
}];
pub const GPU_INFO_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GPU_INFO.TOO_MANY_INPUTS",
    identifier: Some("RunMat:gpuInfo:TooManyInputs"),
    when: "An input argument was supplied to gpuInfo().",
    message: "gpuInfo: too many input arguments",
};
const ERRORS: [BuiltinErrorDescriptor; 1] = [GPU_INFO_ERROR_TOO_MANY_INPUTS];
pub const GPU_INFO_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
