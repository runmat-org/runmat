use crate::*;

const OUTPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Square symmetric matrix or condensed distance vector, preserving the input class.",
};
const INPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Z",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Condensed distance vector or square distance matrix.",
};
const FORCE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "force",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Conversion direction: 'tomatrix' or 'tovector'.",
};
const OUTPUTS: [BuiltinParamDescriptor; 1] = [OUTPUT];
const INPUTS: [BuiltinParamDescriptor; 1] = [INPUT];
const INPUTS_FORCE: [BuiltinParamDescriptor; 2] = [INPUT, FORCE];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = squareform(Z)",
        inputs: &INPUTS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "Y = squareform(Z, Force)",
        inputs: &INPUTS_FORCE,
        outputs: &OUTPUTS,
    },
];
pub const SQUAREFORM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &super::super::distance_errors::DISTANCE_ERRORS,
};

pub const SQUAREFORM_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "squareform-integer-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "squareform with typed-integer distance data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:SquareformIntegerInputExtension"),
    };
pub const SQUAREFORM_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "squareform-explicit-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "squareform with an explicit GPU input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:SquareformExplicitGpuInputExtension"),
    };
pub const SQUAREFORM_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    SQUAREFORM_INTEGER_INPUT_EXTENSION,
    SQUAREFORM_EXPLICIT_GPU_EXTENSION,
];
