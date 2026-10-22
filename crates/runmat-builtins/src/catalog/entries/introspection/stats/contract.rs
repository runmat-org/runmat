use super::super::memoize_contract::*;
use crate::*;
pub const STATS_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "stats",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Structure with cache entries and hit/miss counters.",
}];

pub const STATS_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "stats accepts only a MemoizedFunction object. Cached integer inputs and outputs are opaque payloads preserved by the cache; the counters produced by stats are metadata, not integer input roles.",
};

pub const STATS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "S = stats(memoizedFcn)",
    inputs: &MEMOIZED_INPUTS,
    outputs: &STATS_OUTPUTS,
}];

pub const MEMOIZE_ERROR_INVALID_CACHE_SIZE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEMOIZE.INVALID_CACHE_SIZE",
    identifier: Some("RunMat:memoize:InvalidCacheSize"),
    when: "CacheSize is not a positive finite integer scalar.",
    message: "memoize: CacheSize must be a positive integer scalar",
};

pub const STATS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &STATS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &STATS_PUBLIC_ERRORS,
};

pub const STATS_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STATS.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:stats:TooManyOutputs"),
    when: "More outputs are requested than this form supports.",
    message: "stats: too many output arguments",
};
const STATS_PUBLIC_ERRORS: [BuiltinErrorDescriptor; 4] = [
    MEMOIZE_ERROR_INVALID_OBJECT,
    MEMOIZE_ERROR_INVALID_CACHE_SIZE,
    MEMOIZE_ERROR_GC,
    STATS_ERROR_TOO_MANY_OUTPUTS,
];
