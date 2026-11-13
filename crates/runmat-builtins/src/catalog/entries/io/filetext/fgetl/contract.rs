use crate::*;

pub const FGETL_INTEGER_ID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fgetl-integer-fileid",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "integer-class fgetl file identifiers are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FgetlIntegerIdExtension"),
};
pub const FGETL_SINGLE_ID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fgetl-single-fileid",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "single-precision fgetl file identifiers are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FgetlSingleIdExtension"),
};
pub const FGETL_RESIDENT_ID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fgetl-resident-fileid",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "provider-resident fgetl file identifiers are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FgetlResidentIdExtension"),
};
pub const FGETL_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    FGETL_INTEGER_ID_EXTENSION,
    FGETL_SINGLE_ID_EXTENSION,
    FGETL_RESIDENT_ID_EXTENSION,
];

pub const FGETL_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "fileID",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "MATLAB-compatible file identifiers are integer-valued doubles; all eight typed integer classes are one gated RunMat extension and are checked exactly against the registry identifier domain.",
    }];
pub const FGETL_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "tline = fgetl(integer_fileID)",
        inputs: &FGETL_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The identifier is validated exactly before registry access; a successful read returns a host character row and end-of-file returns the double sentinel -1.",
    }];

pub const FGETL_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tline",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Next line without terminator, or -1 at end-of-file.",
}];
pub const FGETL_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fid",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "File identifier opened by fopen.",
}];
pub const FGETL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tline = fgetl(fid)",
    inputs: &FGETL_INPUTS,
    outputs: &FGETL_OUTPUT,
}];

pub const FGETL_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FGETL.INVALID_INPUT",
    identifier: Some("RunMat:fgetl:InvalidInput"),
    when: "Input argument count or identifier form is invalid.",
    message: "fgetl: invalid input arguments",
};
pub const FGETL_ERROR_INVALID_IDENTIFIER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FGETL.INVALID_IDENTIFIER",
    identifier: Some("RunMat:fgetl:InvalidIdentifier"),
    when: "Identifier does not refer to a readable open file.",
    message: "fgetl: invalid file identifier. Use fopen to generate a valid file ID.",
};
pub const FGETL_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FGETL.IO",
    identifier: Some("RunMat:fgetl:IoFailure"),
    when: "Line read or decode fails.",
    message: "fgetl: file I/O failed",
};
pub const FGETL_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FGETL.INTERNAL",
    identifier: None,
    when: "Internal control-flow conversion failed.",
    message: "fgetl: internal error",
};
pub const FGETL_ERRORS: [BuiltinErrorDescriptor; 4] = [
    FGETL_ERROR_INVALID_INPUT,
    FGETL_ERROR_INVALID_IDENTIFIER,
    FGETL_ERROR_IO,
    FGETL_ERROR_INTERNAL,
];
pub const FGETL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FGETL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FGETL_ERRORS,
};
