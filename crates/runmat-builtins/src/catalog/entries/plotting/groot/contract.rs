use crate::*;
const GROOT_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "root",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Root graphics object handle.",
}];

const GROOT_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const GROOT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "root = groot()",
    inputs: &GROOT_INPUTS_NONE,
    outputs: &GROOT_OUTPUT_HANDLE,
}];

const GROOT_ERRORS: [BuiltinErrorDescriptor; 1] = [GROOT_ERROR_TOO_MANY_OUTPUTS];

pub const GROOT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GROOT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GROOT_ERRORS,
};

pub const GROOT_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GROOT.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:groot:TooManyOutputs"),
    when: "More than 1 output arguments are requested.",
    message: "groot: too many output arguments",
};
