use super::super::debugging_contract::*;
use crate::*;
pub const DBCLEAR_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "dbclear(options...)",
    inputs: &DBCLEAR_OPTION_INPUTS,
    outputs: &[],
}];
pub const DBCLEAR_OPTION_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor { name:"options", ty:BuiltinParamType::StringScalar, arity:BuiltinParamArity::Variadic, default:None, description:"Compatibility command text. Execution breakpoints are not implemented; these words do not enable breakpoint execution." }];

pub const DBCLEAR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DBCLEAR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DBCLEAR_PUBLIC_ERRORS,
};

pub const DBCLEAR_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DBCLEAR.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:dbclear:TooManyOutputs"),
    when: "More outputs are requested than this form supports.",
    message: "dbclear: too many output arguments",
};
const DBCLEAR_PUBLIC_ERRORS: [BuiltinErrorDescriptor; 6] = [
    DEBUG_ERROR_TOO_MANY_INPUTS,
    DEBUG_ERROR_INVALID_INPUT,
    DEBUG_ERROR_NO_CURRENT_FILE,
    DEBUG_ERROR_FILE_NOT_FOUND,
    DEBUG_ERROR_FILE_READ,
    DBCLEAR_ERROR_TOO_MANY_OUTPUTS,
];
