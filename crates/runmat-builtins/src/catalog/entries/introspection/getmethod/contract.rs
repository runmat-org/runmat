use crate::*;
const GETMETHOD_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fh",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Bound method closure/handle.",
}];
const GETMETHOD_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "obj_or_class",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Object receiver or class reference.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Method name.",
    },
];
const GETMETHOD_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "fh = getmethod(obj_or_class, name)",
    inputs: &GETMETHOD_INPUTS,
    outputs: &GETMETHOD_OUTPUT,
}];
pub const GETMETHOD_ERROR_NAME_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETMETHOD.NAME_INVALID",
    identifier: Some("RunMat:GetMethodNameInvalid"),
    when: "Method name is empty.",
    message: "getmethod: method name must not be empty",
};
pub const GETMETHOD_ERROR_RECEIVER_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETMETHOD.RECEIVER_UNSUPPORTED",
    identifier: Some("RunMat:GetMethodReceiverUnsupported"),
    when: "Receiver is neither object nor class reference.",
    message: "getmethod: unsupported receiver",
};
pub const GETMETHOD_ERROR_METHOD_PRIVATE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETMETHOD.METHOD_PRIVATE",
    identifier: Some("RunMat:MethodPrivate"),
    when: "Resolved method exists but is inaccessible from the current class scope.",
    message: "getmethod: method is not accessible from current scope",
};
const GETMETHOD_ERRORS: [BuiltinErrorDescriptor; 5] = [
    GETMETHOD_ERROR_NAME_INVALID,
    GETMETHOD_ERROR_RECEIVER_UNSUPPORTED,
    GETMETHOD_ERROR_METHOD_PRIVATE,
    GETMETHOD_ERROR_ARITY,
    GETMETHOD_ERROR_OUTPUTS,
];
pub const GETMETHOD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GETMETHOD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GETMETHOD_ERRORS,
};
pub const GETMETHOD_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "getmethod-bound-method-handle",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "getmethod bound method-handle creation is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GetmethodExtension"),
};
pub const GETMETHOD_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [GETMETHOD_EXTENSION];
pub const GETMETHOD_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "RunMat getmethod accepts an object or class reference and a host text method name only. All eight integer classes and provider-resident numeric values reject without conversion, gather, or provider access; this audit does not imply compatibility with MATLAB methods or ismethod.",
};
pub const GETMETHOD_ERROR_ARITY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETMETHOD.ARITY",
    identifier: Some("RunMat:GetMethod:Arity"),
    when: "Input count differs from two.",
    message: "getmethod: expected receiver and method name",
};
pub const GETMETHOD_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETMETHOD.OUTPUTS",
    identifier: Some("RunMat:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "getmethod: at most one output is supported",
};
