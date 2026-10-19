use crate::*;

const LAYERS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "layers",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One supported layer object, a cell collection, or a layer-graph/network object.",
}];
const INITIALIZED_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "layers",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description:
            "One supported layer object, a cell collection, or a layer-graph/network object.",
    },
    BuiltinParamDescriptor {
        name: super::DeepLearningNetworkOption::Initialize.name(),
        ty: BuiltinParamType::PropertyName,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "The Initialize name-value key.",
    },
    BuiltinParamDescriptor {
        name: "tf",
        ty: BuiltinParamType::PropertyValue,
        arity: BuiltinParamArity::Optional,
        default: Some("true"),
        description: "Whether to fill missing supported fully connected weights and bias.",
    },
];
const OUTPUT_NAME_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "layers",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "A supported sequential layer collection.",
    },
    BuiltinParamDescriptor {
        name: super::DeepLearningNetworkOption::OutputNames.name(),
        ty: BuiltinParamType::PropertyName,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "The OutputNames name-value key.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "The final layer name in the supported single-output sequential network.",
    },
];
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "net",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Value-backed dlnetwork with canonical Layers and a derived Learnables table.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "net = dlnetwork(layers)",
        inputs: &LAYERS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "net = dlnetwork(graph)",
        inputs: &LAYERS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "net = dlnetwork(layers, 'Initialize', tf)",
        inputs: &INITIALIZED_INPUTS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "net = dlnetwork(layers, 'OutputNames', name)",
        inputs: &OUTPUT_NAME_INPUTS,
        outputs: &OUTPUTS,
    },
];

pub const DLNETWORK_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEEP_LEARNING.INVALID_INPUT",
    identifier: Some("RunMat:deepLearning:InvalidInput"),
    when: "The layer collection, supported sequential topology, name-value options, initialization dimensions, or requested output count is invalid.",
    message: "dlnetwork received invalid layer or initialization metadata",
};
const ERRORS: [BuiltinErrorDescriptor; 1] = [DLNETWORK_INVALID_INPUT];
pub const DLNETWORK_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
