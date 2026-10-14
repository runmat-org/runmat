use crate::*;

const SQUARE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "dlarray input differentiated by the fixture.",
}];
const SQUARE_OUTPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "loss",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Summed square loss.",
    },
    BuiltinParamDescriptor {
        name: "gradient",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Gradient of the loss with respect to x.",
    },
];
const SQUARE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "[loss, gradient] = __deep_learning_square_grad(x)",
    inputs: &SQUARE_INPUTS,
    outputs: &SQUARE_OUTPUTS,
}];

pub const DEEP_LEARNING_SQUARE_GRAD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SQUARE_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &[],
};

const NETWORK_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "network",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Network whose parameters are differentiated.",
    },
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input dlarray forwarded through the network.",
    },
];
const NETWORK_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "gradient",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Gradients for the network learnables.",
}];
const NETWORK_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "gradient = __deep_learning_network_grad(network, x)",
    inputs: &NETWORK_INPUTS,
    outputs: &NETWORK_OUTPUTS,
}];

pub const DEEP_LEARNING_NETWORK_GRAD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NETWORK_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &[],
};
