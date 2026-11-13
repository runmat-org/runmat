use crate::*;

pub const FREWIND_INTEGER_ID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "frewind-integer-fileid",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "integer-class frewind file identifiers are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FrewindIntegerIdExtension"),
};
pub const FREWIND_SINGLE_ID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "frewind-single-fileid",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "single-precision frewind file identifiers are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FrewindSingleIdExtension"),
};
pub const FREWIND_LOGICAL_ID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "frewind-logical-fileid",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "logical frewind file identifiers are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FrewindLogicalIdExtension"),
};
pub const FREWIND_RESIDENT_ID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "frewind-resident-fileid",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "provider-resident frewind file identifiers are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FrewindResidentIdExtension"),
};
pub const FREWIND_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    FREWIND_INTEGER_ID_EXTENSION,
    FREWIND_SINGLE_ID_EXTENSION,
    FREWIND_LOGICAL_ID_EXTENSION,
    FREWIND_RESIDENT_ID_EXTENSION,
];
pub const FREWIND_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "fileID",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target documents double identifiers; all eight typed integer classes are one exact, independently gated extension.",
    }];
pub const FREWIND_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "frewind(integer_fileID)",
        inputs: &FREWIND_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The identifier is decoded exactly before the host registry seek and the builtin returns no output.",
    }];

pub const FREWIND_OUTPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
pub const FREWIND_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fid",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "File identifier opened by fopen.",
}];
pub const FREWIND_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "frewind(fid)",
    inputs: &FREWIND_INPUTS,
    outputs: &FREWIND_OUTPUTS_NONE,
}];

pub const FREWIND_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FREWIND.INVALID_INPUT",
    identifier: Some("RunMat:frewind:InvalidInput"),
    when: "Input identifier is malformed or out of range.",
    message: "frewind: invalid input arguments",
};
pub const FREWIND_ERROR_INVALID_IDENTIFIER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FREWIND.INVALID_IDENTIFIER",
    identifier: Some("RunMat:frewind:InvalidIdentifier"),
    when: "Identifier does not refer to an open file handle.",
    message: "frewind: invalid file identifier. Use fopen to generate a valid file ID.",
};
pub const FREWIND_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FREWIND.IO",
    identifier: Some("RunMat:frewind:IoFailure"),
    when: "Underlying seek/rewind operation fails.",
    message: "frewind: file I/O failed",
};
pub const FREWIND_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FREWIND.INTERNAL",
    identifier: None,
    when: "Internal runtime control-flow or conversion failed.",
    message: "frewind: internal error",
};
pub const FREWIND_ERRORS: [BuiltinErrorDescriptor; 4] = [
    FREWIND_ERROR_INVALID_INPUT,
    FREWIND_ERROR_INVALID_IDENTIFIER,
    FREWIND_ERROR_IO,
    FREWIND_ERROR_INTERNAL,
];
pub const FREWIND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FREWIND_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FREWIND_ERRORS,
};
