use crate::*;

const CALL_BOUND_METHOD_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Method return value(s).",
}];

const CALL_BOUND_METHOD_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "base",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Object receiver.",
    },
    BuiltinParamDescriptor {
        name: "method",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Method name.",
    },
    BuiltinParamDescriptor {
        name: "scope",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Creator class scope for access checks.",
    },
    BuiltinParamDescriptor {
        name: "varargin",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Method arguments.",
    },
];

const CALL_BOUND_METHOD_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "[out] = __runmat_call_bound_method__(base, method, scope, varargin)",
        inputs: &CALL_BOUND_METHOD_INPUTS,
        outputs: &CALL_BOUND_METHOD_OUTPUT,
    }];

pub const CALL_BOUND_METHOD_ERRORS: [BuiltinErrorDescriptor; 1] = [BuiltinErrorDescriptor {
    code: "RM.CALL_BOUND_METHOD.SCOPE_INVALID",
    identifier: Some("RunMat:CallMethodScopeInvalid"),
    when: "Scope argument is not a string scalar.",
    message: "__runmat_call_bound_method__: invalid scope argument",
}];

pub const CALL_BOUND_METHOD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CALL_BOUND_METHOD_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &CALL_BOUND_METHOD_ERRORS,
};
