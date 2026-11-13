use crate::*;
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "s",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "String scalar containing one line-feed character.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "s = newline",
    inputs: &[],
    outputs: &OUTPUTS,
}];
pub const NEWLINE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NEWLINE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:newline:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "newline: too many output arguments",
};
pub const NEWLINE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[NEWLINE_ERROR_TOO_MANY_OUTPUTS],
};
