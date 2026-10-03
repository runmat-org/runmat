use super::super::descriptor::{ENVIRONMENT, FOREIGN_ERRORS, NONE};
use crate::*;

const ENV_INPUTS: &[BuiltinParamDescriptor] = &[
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Version or ExecutionMode option.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Configuration value.",
    },
];

const PYENV_SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "environment = pyenv()",
        inputs: NONE,
        outputs: ENVIRONMENT,
    },
    BuiltinSignatureDescriptor {
        label: "environment = pyenv(name, value, ...)",
        inputs: ENV_INPUTS,
        outputs: ENVIRONMENT,
    },
];

pub const PYENV_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: PYENV_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: FOREIGN_ERRORS,
};
