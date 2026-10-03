use crate::*;

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
const ERRORS: [BuiltinErrorDescriptor; 0] = [];
pub const GPU_INFO_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
