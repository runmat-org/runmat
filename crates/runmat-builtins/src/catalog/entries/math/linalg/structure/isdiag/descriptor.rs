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

pub const ISDIAG_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = isdiag(A)",
    inputs: &INPUTS,
    outputs: &OUTPUT,
}];

pub const ISDIAG_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISDIAG.INVALID_INPUT",
    identifier: Some("RunMat:isdiag:InvalidInput"),
    when: "Input cannot be interpreted as a supported numeric, logical, sparse, complex, or gpuArray matrix.",
    message: "isdiag: invalid input",
};

pub const ISDIAG_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISDIAG.INTERNAL",
    identifier: Some("RunMat:isdiag:Internal"),
    when: "Runtime fails while gathering GPU input or constructing an internal matrix view.",
    message: "isdiag: internal runtime failure",
};

pub const ISDIAG_ERRORS: [BuiltinErrorDescriptor; 2] =
    [ISDIAG_ERROR_INVALID_INPUT, ISDIAG_ERROR_INTERNAL];
pub const ISDIAG_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISDIAG_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ISDIAG_ERRORS,
};

pub const ISDIAG_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "isdiag-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "isdiag with typed-integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:IsdiagIntegerInputExtension"),
};
pub const ISDIAG_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [ISDIAG_INTEGER_INPUT_EXTENSION];
pub const ISDIAG_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "MATLAB documents single, double, and logical input; exact integer host or resident input is a separately declared and gated RunMat extension.",
};
