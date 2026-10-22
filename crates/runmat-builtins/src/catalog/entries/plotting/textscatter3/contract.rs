use super::super::textscatter::TEXTSCATTER_INTEGER_CAPABILITIES;
use crate::*;

pub const TEXTSCATTER3_INTEGER_AXES_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "textscatter3-integer-axes-handle",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "textscatter3 with a typed-integer numeric axes alias is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:TextScatter3IntegerAxesHandleExtension"),
    };

pub const TEXTSCATTER3_EXPLICIT_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "textscatter3-explicit-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "textscatter3 with explicit gpuArray input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:TextScatter3ExplicitGpuInputExtension"),
    };

pub const TEXTSCATTER3_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    TEXTSCATTER3_INTEGER_AXES_EXTENSION,
    TEXTSCATTER3_EXPLICIT_GPU_INPUT_EXTENSION,
];

pub const TEXTSCATTER3_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] =
    TEXTSCATTER_INTEGER_CAPABILITIES;

const OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ts",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the TextScatter chart object.",
}];

const INPUTS_TEXTSCATTER3_XYZ: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X coordinates.",
    },
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y coordinates.",
    },
    BuiltinParamDescriptor {
        name: "z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Z coordinates.",
    },
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "String vector or cell array of character vectors.",
    },
];

const INPUTS_TEXTSCATTER3_MATRIX: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "xyz",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "N-by-3 coordinate matrix.",
    },
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "String vector or cell array of character vectors.",
    },
];

const INPUTS_PROPS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "props",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Name/value TextScatter properties.",
}];

const TEXTSCATTER3_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "ts = textscatter3(x, y, z, str)",
        inputs: &INPUTS_TEXTSCATTER3_XYZ,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "ts = textscatter3(xyz, str)",
        inputs: &INPUTS_TEXTSCATTER3_MATRIX,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "ts = textscatter3(ax, ...)",
        inputs: &INPUTS_PROPS,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "ts = textscatter3(..., Name, Value)",
        inputs: &INPUTS_PROPS,
        outputs: &OUTPUT_HANDLE,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEXTSCATTER.INVALID_ARGUMENT",
    identifier: Some("RunMat:textscatter:InvalidArgument"),
    when: "Coordinates, text labels, axes target, or TextScatter property values are invalid.",
    message: "textscatter: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEXTSCATTER.INTERNAL",
    identifier: Some("RunMat:textscatter:Internal"),
    when: "Internal plotting state update fails while creating or mutating TextScatter.",
    message: "textscatter: internal operation failed",
};

const ERRORS: [BuiltinErrorDescriptor; 3] = [
    ERROR_INVALID_ARGUMENT,
    ERROR_INTERNAL,
    TEXTSCATTER3_ERROR_TOO_MANY_OUTPUTS,
];

pub const TEXTSCATTER3_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TEXTSCATTER3_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const TEXTSCATTER3_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEXTSCATTER3.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:textscatter3:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "textscatter3: too many output arguments",
};
