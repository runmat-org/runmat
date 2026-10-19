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
        description: "Raw features, sequence or spatial data in the declared input layer's axis convention, or a formatted dlarray.",
    },
];
const OUTPUT_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric output, or a dlarray whose semantic axes follow the executed layers; placement depends on execution route.",
};
const OUTPUTS: [BuiltinParamDescriptor; 1] = [OUTPUT_Y];
const OUTPUTS_WITH_STATE: [BuiltinParamDescriptor; 2] = [
    OUTPUT_Y,
    BuiltinParamDescriptor {
        name: "state",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Canonical Layer/Parameter/Value table of updated recurrent state; the input network is unchanged.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = forward(net, X)",
        inputs: &INPUTS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[Y, state] = forward(net, X)",
        inputs: &INPUTS,
        outputs: &OUTPUTS_WITH_STATE,
    },
];

pub const FORWARD_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEEP_LEARNING.INVALID_INPUT",
    identifier: Some("RunMat:deepLearning:InvalidInput"),
    when: "The network or input data is invalid, layer dimensions or parameters disagree, options are supplied, numerical normalization fails, or more than two outputs are requested.",
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
const INTEGER_PREDICTORS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Typed integer observations are converted to double at the floating-point forward boundary; values outside binary64 exact range can round.",
}];
const INTEGER_LEARNABLES: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "fullyConnectedLayer Weights/Bias",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Consumed integer parameters are converted at the floating-point forward boundary; values outside binary64 exact range can round.",
}];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "forward(net, typed_integer_X)",
        inputs: &INTEGER_PREDICTORS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Typed integer observations enter the host double forward calculation in RunMat mode; this is not a resident integer GPU kernel.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "forward(net_with_typed_integer_learnables, floating_X)",
        inputs: &INTEGER_LEARNABLES,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::FunctionSpecific,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer learnables are converted to the floating predictor's single or double precision. Host inputs use host execution; supported floating GPU-backed dlarray inputs selectively prepare the learnables and execute resident provider operations. This is not a resident integer GPU kernel.",
    },
];
