use crate::*;

const INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "net", ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required, default: None,
        description: "dlnetwork, SeriesNetwork, or DAGNetwork compatibility object with a supported sequential feed-forward Layers collection.",
    },
    BuiltinParamDescriptor {
        name: "filename", ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required, default: None,
        description: "Host scalar path to the ONNX file written through RunMat's filesystem service.",
    },
    BuiltinParamDescriptor {
        name: "Name, Value", ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic, default: None,
        description: "Paired OpsetVersion (13–20), BatchSize, InputNames, OutputNames, or Verbose options.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "exportONNXNetwork(net, filename, Name, Value, ...)",
    inputs: &INPUTS,
    outputs: &[],
}];

pub const EXPORT_ONNX_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEEP_LEARNING.INVALID_INPUT",
    identifier: Some("RunMat:deepLearning:InvalidInput"),
    when: "The network, filename, name-value list, supported layer metadata, learned parameter storage, requested output count, or filesystem write is invalid or fails.",
    message: "exportONNXNetwork: invalid model, option, filename, or write operation",
};
pub const EXPORT_ONNX_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEEP_LEARNING.UNSUPPORTED",
    identifier: Some("RunMat:deepLearning:Unsupported"),
    when: "A recognized layer is outside the supported sequential ONNX lowering subset.",
    message: "exportONNXNetwork: unsupported layer or graph structure",
};
const ERRORS: [BuiltinErrorDescriptor; 2] = [EXPORT_ONNX_INVALID_INPUT, EXPORT_ONNX_UNSUPPORTED];
pub const EXPORT_ONNX_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const EXPORT_ONNX_INTEGER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "export-onnx-typed-integer-controls",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "Native fixed-width integer OpsetVersion, BatchSize, and Verbose controls are a RunMat extension.",
    error_identifier: Some("RunMat:compatibility:ExportOnnxIntegerControlExtension"),
};
pub const EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [EXPORT_ONNX_INTEGER_EXTENSION];

const STRUCTURAL_CONTROLS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "OpsetVersion, BatchSize, Verbose",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Typed integer scalar controls are decoded from native storage without a double detour; OpsetVersion and BatchSize are positive and range checked, Verbose is zero or one.",
}];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "exportONNXNetwork(net, filename, integer_control, ...)",
    inputs: &STRUCTURAL_CONTROLS,
    computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
    output_class: BuiltinIntegerOutputClassRule::NotApplicable,
    overflow: BuiltinIntegerOverflowRule::NotApplicable,
    backend: BuiltinIntegerBackendRule::HostOnly,
    overload: BuiltinIntegerOverloadKind::Multiple,
    notes: "The serializer emits floating ONNX initializers for supported numeric learnables; this control capability does not promise exact integer model-data export.",
}];
