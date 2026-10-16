use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when the input matrix satisfies the requested structure predicate.",
}];

const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input numeric, logical, complex, sparse, or gpuArray matrix.",
}];

pub const ISTRIU_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = istriu(A)",
    inputs: &INPUTS,
    outputs: &OUTPUT,
}];

pub const ISTRIU_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISTRIU.INVALID_INPUT",
    identifier: Some("RunMat:istriu:InvalidInput"),
    when: "Input cannot be interpreted as a supported numeric, logical, sparse, complex, or gpuArray matrix.",
    message: "istriu: invalid input",
};

pub const ISTRIU_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISTRIU.INTERNAL",
    identifier: Some("RunMat:istriu:Internal"),
    when: "Runtime fails while gathering GPU input or constructing an internal matrix view.",
    message: "istriu: internal runtime failure",
};

pub const ISTRIU_ERRORS: [BuiltinErrorDescriptor; 2] =
    [ISTRIU_ERROR_INVALID_INPUT, ISTRIU_ERROR_INTERNAL];
pub const ISTRIU_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISTRIU_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ISTRIU_ERRORS,
};

pub const ISTRIU_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "istriu-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "istriu with typed-integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:IstriuIntegerInputExtension"),
};
pub const ISTRIU_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [ISTRIU_INTEGER_INPUT_EXTENSION];
pub const ISTRIU_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "MATLAB documents single, double, and logical input; exact integer host or resident input is a separately declared and gated RunMat extension.",
};
