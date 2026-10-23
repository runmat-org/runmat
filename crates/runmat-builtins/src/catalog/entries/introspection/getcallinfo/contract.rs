use super::super::debugging_contract::*;
use crate::*;
pub const GETCALLINFO_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "info = getcallinfo",
    inputs: &[],
    outputs: &GETCALLINFO_OUTPUT,
}];

pub const GETCALLINFO_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "info",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar struct describing the current call context and stack.",
}];

pub const GETCALLINFO_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GETCALLINFO_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GETCALLINFO_PUBLIC_ERRORS,
};

pub const GETCALLINFO_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETCALLINFO.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:getcallinfo:TooManyOutputs"),
    when: "More outputs are requested than this form supports.",
    message: "getcallinfo: too many output arguments",
};
const GETCALLINFO_PUBLIC_ERRORS: [BuiltinErrorDescriptor; 6] = [
    DEBUG_ERROR_TOO_MANY_INPUTS,
    DEBUG_ERROR_INVALID_INPUT,
    DEBUG_ERROR_NO_CURRENT_FILE,
    DEBUG_ERROR_FILE_NOT_FOUND,
    DEBUG_ERROR_FILE_READ,
    GETCALLINFO_ERROR_TOO_MANY_OUTPUTS,
];
