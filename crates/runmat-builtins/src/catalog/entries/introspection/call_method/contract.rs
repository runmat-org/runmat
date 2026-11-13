use crate::*;

const CALL_METHOD_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Method return value(s).",
}];

const CALL_METHOD_INPUTS: [BuiltinParamDescriptor; 3] = [
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
        name: "varargin",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Method arguments.",
    },
];

const CALL_METHOD_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "[out] = call_method(base, method, varargin)",
    inputs: &CALL_METHOD_INPUTS,
    outputs: &CALL_METHOD_OUTPUT,
}];

pub const CALL_METHOD_ERROR_NAME_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CALL_METHOD.NAME_INVALID",
    identifier: Some("RunMat:CallMethodNameInvalid"),
    when: "The method name is not scalar text, is empty, or is missing.",
    message: "call_method: method name must be nonempty scalar text",
};

pub const CALL_METHOD_ERROR_RECEIVER_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CALL_METHOD.RECEIVER_INVALID",
    identifier: Some("RunMat:InvalidObjectDispatch"),
    when: "Receiver is not an object or handle object.",
    message: "call_method: requires object receiver",
};

pub const CALL_METHOD_ERRORS: [BuiltinErrorDescriptor; 2] = [
    CALL_METHOD_ERROR_NAME_INVALID,
    CALL_METHOD_ERROR_RECEIVER_INVALID,
];

pub const CALL_METHOD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CALL_METHOD_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &CALL_METHOD_ERRORS,
};
