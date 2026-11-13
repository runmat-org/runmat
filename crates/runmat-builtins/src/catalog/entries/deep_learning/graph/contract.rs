use crate::*;

const LAYERS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "layers",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "A layer object or cell collection of layer objects.",
}];
const NETWORK: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "networkOrLayers",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "A layer graph, one layer, or a cell collection of layers.",
}];
const GRAPH: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "graph",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host layer-graph metadata object.",
}];
const ANALYSIS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "analysis",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host network-analysis metadata object; no analyzer UI is opened.",
}];
const LAYER_GRAPH_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "graph = layerGraph()",
        inputs: &[],
        outputs: &GRAPH,
    },
    BuiltinSignatureDescriptor {
        label: "graph = layerGraph(layers)",
        inputs: &LAYERS,
        outputs: &GRAPH,
    },
];
const ANALYZE_NETWORK_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "analysis = analyzeNetwork(networkOrLayers)",
    inputs: &NETWORK,
    outputs: &ANALYSIS,
}];

pub const GRAPH_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEEP_LEARNING.INVALID_INPUT",
    identifier: Some("RunMat:deepLearning:InvalidInput"),
    when: "The input count, layer collection, graph metadata, or a contained layer is invalid, or more than one output is requested.",
    message: "deep learning builtin received invalid input",
};
const GRAPH_ERRORS: [BuiltinErrorDescriptor; 1] = [GRAPH_INVALID_INPUT];

pub const LAYER_GRAPH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LAYER_GRAPH_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GRAPH_ERRORS,
};
pub const ANALYZE_NETWORK_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ANALYZE_NETWORK_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GRAPH_ERRORS,
};
