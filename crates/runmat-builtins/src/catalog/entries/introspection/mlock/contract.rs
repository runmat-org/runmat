use super::super::debugging_contract::*;
use crate::*;
pub const MLOCK_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "mlock",
    inputs: &[],
    outputs: &[],
}];

pub const MLOCK_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MLOCK_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &MLOCK_PUBLIC_ERRORS,
};

pub const MLOCK_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MLOCK.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:mlock:TooManyOutputs"),
    when: "More outputs are requested than this form supports.",
    message: "mlock: too many output arguments",
};
const MLOCK_PUBLIC_ERRORS: [BuiltinErrorDescriptor; 6] = [
    DEBUG_ERROR_TOO_MANY_INPUTS,
    DEBUG_ERROR_INVALID_INPUT,
    DEBUG_ERROR_NO_CURRENT_FILE,
    DEBUG_ERROR_FILE_NOT_FOUND,
    DEBUG_ERROR_FILE_READ,
    MLOCK_ERROR_TOO_MANY_OUTPUTS,
];
