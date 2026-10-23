use super::super::memoize_contract::*;
use crate::*;
pub const CLEAR_CACHE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "clearCache(memoizedFcn)",
    inputs: &MEMOIZED_INPUTS,
    outputs: &[],
}];

pub const CLEAR_CACHE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "clearCache accepts only a MemoizedFunction handle and has no public output; integer values may exist inside cached results but are opaque to cache clearing.",
    };

pub const CLEAR_CACHE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CLEAR_CACHE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:clearCache:TooManyOutputs"),
    when: "An output is requested from clearCache.",
    message: "clearCache does not return output arguments",
};

pub const CLEAR_CACHE_ERRORS: [BuiltinErrorDescriptor; 3] = [
    MEMOIZE_ERROR_INVALID_OBJECT,
    MEMOIZE_ERROR_GC,
    CLEAR_CACHE_ERROR_TOO_MANY_OUTPUTS,
];

pub const CLEAR_CACHE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CLEAR_CACHE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CLEAR_CACHE_ERRORS,
};
