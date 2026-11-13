use super::super::descriptor::{
    ENVIRONMENT, PYTHON_ERROR_ADAPTER_UNAVAILABLE, PYTHON_ERROR_CAPABILITY_UNAVAILABLE,
    PYTHON_ERROR_EXCEPTION, PYTHON_ERROR_HOST_UNAVAILABLE, PYTHON_ERROR_INVALID,
    PYTHON_ERROR_MANIFEST, PYTHON_ERROR_UNAVAILABLE,
};
use crate::*;

const ERROR_STALE_ENVIRONMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PYTHON.STALE_ENVIRONMENT",
    identifier: Some("RunMat:Foreign:StaleHandle"),
    when: "The Python environment ticket was invalidated by a host restart.",
    message: "Python environment belongs to a prior host generation.",
};
const ERROR_ENVIRONMENT_OWNER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PYTHON.ENVIRONMENT_OWNER",
    identifier: Some("RunMat:Foreign:HandleMetadataMismatch"),
    when: "The environment ticket belongs to another Python session registry.",
    message: "Python environment belongs to another session.",
};
const ERRORS: &[BuiltinErrorDescriptor] = &[
    PYTHON_ERROR_UNAVAILABLE,
    PYTHON_ERROR_INVALID,
    PYTHON_ERROR_HOST_UNAVAILABLE,
    PYTHON_ERROR_ADAPTER_UNAVAILABLE,
    PYTHON_ERROR_CAPABILITY_UNAVAILABLE,
    PYTHON_ERROR_EXCEPTION,
    PYTHON_ERROR_MANIFEST,
    ERROR_STALE_ENVIRONMENT,
    ERROR_ENVIRONMENT_OWNER,
];

const ENV_INPUT: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "environment",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Python environment returned by pyenv.",
}];

const TERMINATE_SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "environment = terminate(environment)",
    inputs: ENV_INPUT,
    outputs: ENVIRONMENT,
}];

pub const TERMINATE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: TERMINATE_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: ERRORS,
};
