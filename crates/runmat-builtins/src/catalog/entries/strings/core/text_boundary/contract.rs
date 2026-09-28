use crate::*;
const TYPE_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "type",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Boundary type: either, start, or end.",
}];
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "pat",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host pattern value.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "pat = textBoundary",
        inputs: &[],
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "pat = textBoundary(type)",
        inputs: &TYPE_INPUT,
        outputs: &OUTPUTS,
    },
];
pub const TEXT_BOUNDARY_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEXT_BOUNDARY.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:textBoundary:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "textBoundary: too many output arguments",
};
pub const TEXT_BOUNDARY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[TEXT_BOUNDARY_ERROR_TOO_MANY_OUTPUTS],
};
