use super::super::memoize_contract::*;
use crate::*;
pub const STATUS_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Zero when the operation completes.",
}];

pub const CLEAR_ALL_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "clearAllMemoizedCaches",
        inputs: &[],
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "status = clearAllMemoizedCaches",
        inputs: &[],
        outputs: &STATUS_OUTPUTS,
    },
];

pub const CLEAR_ALL_MEMOIZED_CACHES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CLEAR_ALL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CLEAR_ALL_MEMOIZED_CACHES_PUBLIC_ERRORS,
};

pub const CLEAR_ALL_MEMOIZED_CACHES_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.CLEAR_ALL_MEMOIZED_CACHES.TOO_MANY_OUTPUTS",
        identifier: Some("RunMat:clearAllMemoizedCaches:TooManyOutputs"),
        when: "More outputs are requested than this form supports.",
        message: "clearAllMemoizedCaches: too many output arguments",
    };
const CLEAR_ALL_MEMOIZED_CACHES_PUBLIC_ERRORS: [BuiltinErrorDescriptor; 2] = [
    MEMOIZE_ERROR_GC,
    CLEAR_ALL_MEMOIZED_CACHES_ERROR_TOO_MANY_OUTPUTS,
];
