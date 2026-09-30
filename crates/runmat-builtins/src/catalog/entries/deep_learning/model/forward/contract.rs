use crate::*;

const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "net",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "A dlnetwork or supported trained network object.",
    },
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Observations in rows and features in columns, or a dlarray with such Data.",
    },
];
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric scores, or a dlarray preserving input format and residency.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = forward(net, X)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

pub const FORWARD_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEEP_LEARNING.INVALID_INPUT",
    identifier: Some("RunMat:deepLearning:InvalidInput"),
    when: "The network or input data is invalid, layer dimensions or parameters disagree, options are supplied, numerical normalization fails, or more than one output is requested.",
    message: "forward received invalid network or input data",
};
pub const FORWARD_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEEP_LEARNING.UNSUPPORTED",
    identifier: Some("RunMat:deepLearning:Unsupported"),
    when: "The requested network, resident storage, tape operation, or provider capability lies outside the supported forward subset.",
    message: "forward cannot execute this network or placement",
};
pub const FORWARD_PROVIDER_PAYLOAD_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GPU.PROVIDER_PAYLOAD_MISMATCH",
    identifier: Some("RunMat:gpu:ProviderPayloadMismatch"),
    when: "An acceleration provider returns a result with the wrong owner, shape, physical element class, storage, or alias relationship.",
    message: "provider returned an invalid forward result descriptor",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    FORWARD_INVALID_INPUT,
    FORWARD_UNSUPPORTED,
    FORWARD_PROVIDER_PAYLOAD_MISMATCH,
];
pub const FORWARD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const FORWARD_INTEGER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "forward-typed-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "typed integer predictor data or consumed fully connected learnables enter RunMat's floating-point forward calculation",
    error_identifier: Some("RunMat:compatibility:ForwardIntegerDataExtension"),
};
pub const EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [FORWARD_INTEGER_EXTENSION];
const INTEGER_DATA: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X or fullyConnectedLayer Weights/Bias",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Typed integer data are converted at the floating-point forward boundary; integer values outside binary64 exact range can round.",
}];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "forward(net, typed_integer_X_or_learnable)",
        inputs: &INTEGER_DATA,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Typed integer observations and consumed fully connected parameters use the host double forward calculation in RunMat mode; this is not a resident integer GPU kernel.",
    },
];
