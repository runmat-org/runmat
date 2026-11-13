use crate::*;

const STRING_OUTPUT_S: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "String scalar/array result.",
}];

const STRING_INPUTS_VALUE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input value to convert to string array.",
}];

const STRING_INPUTS_VALUE_ENCODING: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input value to convert to string array.",
    },
    BuiltinParamDescriptor {
        name: "encoding",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"UTF-8\""),
        description: "Character encoding (UTF-8 aliases supported).",
    },
];

const STRING_INPUTS_FORMAT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "formatSpec",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Format specification text/cell/string array.",
    },
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Formatting data arguments.",
    },
];

const STRING_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "S = string(X)",
        inputs: &STRING_INPUTS_VALUE,
        outputs: &STRING_OUTPUT_S,
    },
    BuiltinSignatureDescriptor {
        label: "S = string(X, encoding)",
        inputs: &STRING_INPUTS_VALUE_ENCODING,
        outputs: &STRING_OUTPUT_S,
    },
    BuiltinSignatureDescriptor {
        label: "S = string(formatSpec, A...)",
        inputs: &STRING_INPUTS_FORMAT,
        outputs: &STRING_OUTPUT_S,
    },
];

pub const STRING_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRING.INVALID_INPUT",
    identifier: Some("RunMat:string:InvalidInput"),
    when: "Input conversion/formatting/encoding constraints are violated.",
    message: "string: invalid input",
};

pub const STRING_ERRORS: [BuiltinErrorDescriptor; 1] = [STRING_ERROR_INVALID_INPUT];

pub const STRING_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &STRING_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &STRING_ERRORS,
};

pub const STRING_FORMAT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "string-format-spec",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "string(formatSpec, A...) printf-style formatting is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:StringFormatSpecExtension"),
};

pub const STRING_ENCODING_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "string-encoding",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "string(X, encoding) character-encoding selection is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:StringEncodingExtension"),
};

pub const STRING_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "string-explicit-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "string with explicit gpuArray numeric input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:StringExplicitGpuInputExtension"),
};

pub const STRING_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    STRING_FORMAT_EXTENSION,
    STRING_ENCODING_EXTENSION,
    STRING_EXPLICIT_GPU_EXTENSION,
];

const STRING_VALUE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Every built-in integer class converts element by element to decimal text from authoritative storage, preserving input shape and exact wide values.",
    }];

const STRING_FORMAT_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat printf-style format arguments accept every integer class and retain exact typed scalars until the selected formatter consumes them.",
    }];

pub const STRING_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "str = string(A)",
        inputs: &STRING_VALUE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Host and automatically resident integer arrays convert exactly to host strings, including wide integer values. Explicit gpuArray input is independently mode-gated because the compatibility target does not document interactive GPU-array input for string.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "str = string(formatSpec, A...)",
        inputs: &STRING_FORMAT_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "This printf-style formatting form is a mode-gated RunMat extension and does not redefine the documented string(A) conversion surface.",
    },
];
