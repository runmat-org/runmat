use super::super::debugging_contract::*;
use crate::*;
pub const MUNLOCK_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "munlock",
        inputs: &[],
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "munlock(fun)",
        inputs: &TEXT_INPUT,
        outputs: &[],
    },
];

pub const MUNLOCK_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MUNLOCK_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &MUNLOCK_PUBLIC_ERRORS,
};

pub const MUNLOCK_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUNLOCK.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:munlock:TooManyOutputs"),
    when: "More outputs are requested than this form supports.",
    message: "munlock: too many output arguments",
};
const MUNLOCK_PUBLIC_ERRORS: [BuiltinErrorDescriptor; 6] = [
    DEBUG_ERROR_TOO_MANY_INPUTS,
    DEBUG_ERROR_INVALID_INPUT,
    DEBUG_ERROR_NO_CURRENT_FILE,
    DEBUG_ERROR_FILE_NOT_FOUND,
    DEBUG_ERROR_FILE_READ,
    MUNLOCK_ERROR_TOO_MANY_OUTPUTS,
];
