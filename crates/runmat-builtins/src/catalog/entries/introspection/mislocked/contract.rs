use super::super::debugging_contract::*;
use crate::*;
pub const MISLOCKED_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical scalar indicating whether the function or script is locked.",
}];

pub const MISLOCKED_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "tf = mislocked",
        inputs: &[],
        outputs: &MISLOCKED_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "tf = mislocked(fun)",
        inputs: &TEXT_INPUT,
        outputs: &MISLOCKED_OUTPUT,
    },
];

pub const MISLOCKED_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MISLOCKED_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &MISLOCKED_PUBLIC_ERRORS,
};

pub const MISLOCKED_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MISLOCKED.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:mislocked:TooManyOutputs"),
    when: "More outputs are requested than this form supports.",
    message: "mislocked: too many output arguments",
};
const MISLOCKED_PUBLIC_ERRORS: [BuiltinErrorDescriptor; 6] = [
    DEBUG_ERROR_TOO_MANY_INPUTS,
    DEBUG_ERROR_INVALID_INPUT,
    DEBUG_ERROR_NO_CURRENT_FILE,
    DEBUG_ERROR_FILE_NOT_FOUND,
    DEBUG_ERROR_FILE_READ,
    MISLOCKED_ERROR_TOO_MANY_OUTPUTS,
];
