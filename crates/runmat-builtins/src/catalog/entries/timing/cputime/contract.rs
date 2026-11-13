use crate::*;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "t",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Total process CPU seconds, if available on the host.",
}];
const INPUTS: [BuiltinParamDescriptor; 0] = [];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "t = cputime()",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

pub const CPUTIME_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CPUTIME.TOO_MANY_INPUTS",
    identifier: Some("RunMat:cputime:TooManyInputs"),
    when: "Any input arguments are supplied.",
    message: "cputime: too many input arguments",
};
pub const CPUTIME_ERROR_UNAVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CPUTIME.UNAVAILABLE",
    identifier: Some("RunMat:cputime:Unavailable"),
    when: "The host platform does not expose process CPU time.",
    message: "cputime: process CPU time is unavailable",
};
const ERRORS: [BuiltinErrorDescriptor; 2] =
    [CPUTIME_ERROR_TOO_MANY_INPUTS, CPUTIME_ERROR_UNAVAILABLE];
pub const CPUTIME_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
