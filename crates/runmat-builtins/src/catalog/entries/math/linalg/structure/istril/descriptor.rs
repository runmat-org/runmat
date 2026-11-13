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

pub const ISTRIL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = istril(A)",
    inputs: &INPUTS,
    outputs: &OUTPUT,
}];

pub const ISTRIL_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISTRIL.INVALID_INPUT",
    identifier: Some("RunMat:istril:InvalidInput"),
    when: "Input cannot be interpreted as a supported numeric, logical, sparse, complex, or gpuArray matrix.",
    message: "istril: invalid input",
};

pub const ISTRIL_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISTRIL.INTERNAL",
    identifier: Some("RunMat:istril:Internal"),
    when: "Runtime fails while gathering GPU input or constructing an internal matrix view.",
    message: "istril: internal runtime failure",
};

pub const ISTRIL_ERRORS: [BuiltinErrorDescriptor; 2] =
    [ISTRIL_ERROR_INVALID_INPUT, ISTRIL_ERROR_INTERNAL];
pub const ISTRIL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISTRIL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ISTRIL_ERRORS,
};

pub const ISTRIL_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "istril-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "istril with typed-integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:IstrilIntegerInputExtension"),
};
pub const ISTRIL_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [ISTRIL_INTEGER_INPUT_EXTENSION];
pub const ISTRIL_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "MATLAB documents single, double, and logical input; exact integer host or resident input is a separately declared and gated RunMat extension.",
};
