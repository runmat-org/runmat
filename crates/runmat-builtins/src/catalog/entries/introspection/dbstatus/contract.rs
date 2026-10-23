use super::super::debugging_contract::*;
use crate::*;
pub const DBSTATUS_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "dbstatus(options...)",
        inputs: &DBSTATUS_OPTION_INPUTS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "b = dbstatus(options...)",
        inputs: &DBSTATUS_OPTION_INPUTS,
        outputs: &DBSTATUS_OUTPUT,
    },
];
pub const DBSTATUS_OPTION_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor { name:"options", ty:BuiltinParamType::StringScalar, arity:BuiltinParamArity::Variadic, default:None, description:"Compatibility command text. Execution breakpoints are not implemented; these words do not enable breakpoint execution." }];

pub const DBSTATUS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "b",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Empty 0-by-0 cell list; execution breakpoints are not implemented.",
}];

pub const DBSTATUS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DBSTATUS_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DBSTATUS_PUBLIC_ERRORS,
};

pub const DBSTATUS_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DBSTATUS.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:dbstatus:TooManyOutputs"),
    when: "More outputs are requested than this form supports.",
    message: "dbstatus: too many output arguments",
};
const DBSTATUS_PUBLIC_ERRORS: [BuiltinErrorDescriptor; 6] = [
    DEBUG_ERROR_TOO_MANY_INPUTS,
    DEBUG_ERROR_INVALID_INPUT,
    DEBUG_ERROR_NO_CURRENT_FILE,
    DEBUG_ERROR_FILE_NOT_FOUND,
    DEBUG_ERROR_FILE_READ,
    DBSTATUS_ERROR_TOO_MANY_OUTPUTS,
];
