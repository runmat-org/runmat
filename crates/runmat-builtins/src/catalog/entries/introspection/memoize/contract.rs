use super::super::memoize_contract::*;
use crate::*;
pub const MEMOIZE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fh",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Function handle to memoize.",
}];

pub const MEMOIZE_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "memoizedFcn",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "MemoizedFunction handle object.",
}];

pub const MEMOIZE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "memoizedFcn = memoize(fh)",
    inputs: &MEMOIZE_INPUTS,
    outputs: &MEMOIZE_OUTPUTS,
}];

pub const MEMOIZE_ERROR_INVALID_FUNCTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEMOIZE.INVALID_FUNCTION",
    identifier: Some("RunMat:memoize:InvalidFunction"),
    when: "The input is not a function handle or closure.",
    message: "memoize: input must be a function handle",
};

pub const MEMOIZE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MEMOIZE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &MEMOIZE_PUBLIC_ERRORS,
};

pub const MEMOIZE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "memoize accepts only a function handle and returns a MemoizedFunction object. Integer values used later as arguments to that object belong to callable cache-key semantics, not to the memoize constructor's public integer surface.",
};

pub const MEMOIZE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEMOIZE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:memoize:TooManyOutputs"),
    when: "More outputs are requested than this form supports.",
    message: "memoize: too many output arguments",
};
const MEMOIZE_PUBLIC_ERRORS: [BuiltinErrorDescriptor; 3] = [
    MEMOIZE_ERROR_INVALID_FUNCTION,
    MEMOIZE_ERROR_GC,
    MEMOIZE_ERROR_TOO_MANY_OUTPUTS,
];
