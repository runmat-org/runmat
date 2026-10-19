use crate::*;

pub const SS2TF_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ss2tf-explicit-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ss2tf with explicit gpuArray matrices is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Ss2tfExplicitGpuExtension"),
};

const MATRICES: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "State matrix.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix.",
    },
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Output matrix.",
    },
    BuiltinParamDescriptor {
        name: "D",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Feedthrough matrix.",
    },
];
const WITH_INPUT: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "State matrix.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix.",
    },
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Output matrix.",
    },
    BuiltinParamDescriptor {
        name: "D",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Feedthrough matrix.",
    },
    BuiltinParamDescriptor {
        name: "ni",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-based input channel index.",
    },
];
const OUTPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One numerator row per output channel.",
    },
    BuiltinParamDescriptor {
        name: "a",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Common denominator row.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "[b,a] = ss2tf(A,B,C,D)",
        inputs: &MATRICES,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[b,a] = ss2tf(A,B,C,D,ni)",
        inputs: &WITH_INPUT,
        outputs: &OUTPUTS,
    },
];
const ERRORS: [BuiltinErrorDescriptor; 4] = [
    BuiltinErrorDescriptor {
        code: "RM.SS2TF.INVALID_ARGUMENT",
        identifier: Some("RunMat:ss2tf:InvalidArgument"),
        when: "Input index, class, or output form is invalid.",
        message: "ss2tf: invalid argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.SS2TF.INVALID_MODEL",
        identifier: Some("RunMat:ss2tf:InvalidModel"),
        when: "State-space dimensions are incompatible.",
        message: "ss2tf: invalid model",
    },
    BuiltinErrorDescriptor {
        code: "RM.SS2TF.UNSUPPORTED_MODEL",
        identifier: Some("RunMat:ss2tf:UnsupportedModel"),
        when: "A requested model form is unsupported.",
        message: "ss2tf: unsupported model",
    },
    BuiltinErrorDescriptor {
        code: "RM.SS2TF.INTERNAL",
        identifier: Some("RunMat:ss2tf:Internal"),
        when: "Polynomial result construction fails.",
        message: "ss2tf: internal error",
    },
];
pub const SS2TF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
