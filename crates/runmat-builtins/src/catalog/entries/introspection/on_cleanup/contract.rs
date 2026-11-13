use crate::*;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "cleanupObj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host handle object whose final owner release invokes the cleanup function.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "cleanupFun",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Zero-input function handle or closure retained for later cleanup.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "cleanupObj = onCleanup(cleanupFun)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

pub const ON_CLEANUP_ERROR_INVALID_CALLBACK: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ONCLEANUP.INVALID_CALLBACK",
    identifier: Some("RunMat:onCleanup:InvalidCallback"),
    when: "The cleanup function is not a function handle or closure.",
    message: "onCleanup: cleanupFun must be a function handle",
};
pub const ON_CLEANUP_ERROR_INVALID_OBJECT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ONCLEANUP.INVALID_OBJECT",
    identifier: Some("RunMat:onCleanup:InvalidObject"),
    when: "A cleanup operation targets a value that is not an onCleanup object.",
    message: "onCleanup: invalid cleanup object",
};
pub const ON_CLEANUP_ERROR_GC: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ONCLEANUP.GC",
    identifier: Some("RunMat:onCleanup:GcFailure"),
    when: "The cleanup target cannot be allocated, rooted, registered, or mutated.",
    message: "onCleanup: internal object storage failed",
};
pub const ON_CLEANUP_ERROR_HOST: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ONCLEANUP.LIFECYCLE_HOST_UNAVAILABLE",
    identifier: Some("RunMat:onCleanup:LifecycleHostUnavailable"),
    when: "No explicit session lifecycle host owns cleanup callback dispatch.",
    message: "onCleanup: lifecycle host is unavailable",
};
pub const ON_CLEANUP_ERRORS: [BuiltinErrorDescriptor; 4] = [
    ON_CLEANUP_ERROR_INVALID_CALLBACK,
    ON_CLEANUP_ERROR_INVALID_OBJECT,
    ON_CLEANUP_ERROR_GC,
    ON_CLEANUP_ERROR_HOST,
];
pub const ON_CLEANUP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ON_CLEANUP_ERRORS,
};
pub const ON_CLEANUP_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "onCleanup accepts only a callable. Numeric and integer callbacks are rejected; captured exact integer or resident values retain ordinary callback ownership and are not inspected or converted.",
};
