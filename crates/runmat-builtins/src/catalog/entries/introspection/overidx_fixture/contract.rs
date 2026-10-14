use crate::*;

const SUBSREF_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Indexed value.",
}];
const SUBSREF_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "OverIdx object.",
    },
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Standard substruct-compatible indexing path.",
    },
];
const SUBSREF_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.subsref(obj, S)",
    inputs: &SUBSREF_INPUTS,
    outputs: &SUBSREF_OUTPUT,
}];

pub const OVERIDX_ERROR_SUBSREF_PAYLOAD_UNSUPPORTED: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.OVERIDX.SUBSREF_PAYLOAD_UNSUPPORTED",
        identifier: Some("RunMat:OverIdxSubsrefPayloadUnsupported"),
        when: "Indexing kind/payload combination is unsupported.",
        message: "OverIdx.subsref: unsupported payload",
    };
pub const OVERIDX_ERROR_SUBSASGN_PAYLOAD_UNSUPPORTED: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.OVERIDX.SUBSASGN_PAYLOAD_UNSUPPORTED",
        identifier: Some("RunMat:OverIdxSubsasgnPayloadUnsupported"),
        when: "Indexing assignment kind/payload combination is unsupported.",
        message: "OverIdx.subsasgn: unsupported payload",
    };
pub const OVERIDX_ERROR_RECEIVER_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OVERIDX.RECEIVER_INVALID",
    identifier: Some("RunMat:OverIdxReceiverInvalid"),
    when: "Receiver is not an object value.",
    message: "OverIdx: receiver must be object",
};

const SUBSREF_ERRORS: [BuiltinErrorDescriptor; 2] = [
    OVERIDX_ERROR_SUBSREF_PAYLOAD_UNSUPPORTED,
    OVERIDX_ERROR_RECEIVER_INVALID,
];
pub const OVERIDX_SUBSREF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SUBSREF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &SUBSREF_ERRORS,
};

const SUBSASGN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "obj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Updated object.",
}];
const SUBSASGN_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "OverIdx object.",
    },
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Standard substruct-compatible indexing path.",
    },
    BuiltinParamDescriptor {
        name: "rhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Assigned value.",
    },
];
const SUBSASGN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "obj = OverIdx.subsasgn(obj, S, rhs)",
    inputs: &SUBSASGN_INPUTS,
    outputs: &SUBSASGN_OUTPUT,
}];
const SUBSASGN_ERRORS: [BuiltinErrorDescriptor; 2] = [
    OVERIDX_ERROR_SUBSASGN_PAYLOAD_UNSUPPORTED,
    OVERIDX_ERROR_RECEIVER_INVALID,
];
pub const OVERIDX_SUBSASGN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SUBSASGN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &SUBSASGN_ERRORS,
};
