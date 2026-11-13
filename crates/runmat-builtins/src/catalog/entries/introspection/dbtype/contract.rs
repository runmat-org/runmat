use super::super::debugging_contract::*;
use crate::*;
pub const DBTYPE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "file",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "MATLAB source file, path, or function name.",
    },
    BuiltinParamDescriptor {
        name: "range",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional line range such as `3:8`.",
    },
];

pub const DBTYPE_FILE_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "file",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "MATLAB source file, path, or function name.",
}];

pub const DBTYPE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "dbtype file",
        inputs: &DBTYPE_FILE_INPUT,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "dbtype file start:end",
        inputs: &DBTYPE_INPUTS,
        outputs: &[],
    },
];

pub const DBTYPE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DBTYPE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DBTYPE_PUBLIC_ERRORS,
};

pub const DBTYPE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DBTYPE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:dbtype:TooManyOutputs"),
    when: "More outputs are requested than this form supports.",
    message: "dbtype: too many output arguments",
};
const DBTYPE_PUBLIC_ERRORS: [BuiltinErrorDescriptor; 6] = [
    DEBUG_ERROR_TOO_MANY_INPUTS,
    DEBUG_ERROR_INVALID_INPUT,
    DEBUG_ERROR_NO_CURRENT_FILE,
    DEBUG_ERROR_FILE_NOT_FOUND,
    DEBUG_ERROR_FILE_READ,
    DBTYPE_ERROR_TOO_MANY_OUTPUTS,
];
