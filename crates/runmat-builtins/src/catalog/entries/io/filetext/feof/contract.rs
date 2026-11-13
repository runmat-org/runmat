use crate::*;

pub const FEOF_INTEGER_ID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "feof-integer-fileid",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "integer-class feof file identifiers are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FeofIntegerIdExtension"),
};
pub const FEOF_LOGICAL_ID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "feof-logical-fileid",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "logical feof file identifiers are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FeofLogicalIdExtension"),
};
pub const FEOF_RESIDENT_ID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "feof-resident-fileid",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "provider-resident feof file identifiers are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FeofResidentIdExtension"),
};
pub const FEOF_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    FEOF_INTEGER_ID_EXTENSION,
    FEOF_LOGICAL_ID_EXTENSION,
    FEOF_RESIDENT_ID_EXTENSION,
];

pub const FEOF_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "fileID",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "MATLAB-compatible file IDs are integer-valued doubles; all eight typed integer classes are one gated RunMat extension and are checked exactly against i32 before registry access.",
    }];
pub const FEOF_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "tf = feof(integer_fileID)",
        inputs: &FEOF_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The identifier is validated exactly before the registry query; resident and logical identifiers have independent RunMat-only gates, and the result is a host logical scalar.",
    }];

pub const FEOF_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when positioned at end-of-file.",
}];
pub const FEOF_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fid",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "File identifier opened by fopen.",
}];
pub const FEOF_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = feof(fid)",
    inputs: &FEOF_INPUTS,
    outputs: &FEOF_OUTPUT,
}];

pub const FEOF_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEOF.INVALID_INPUT",
    identifier: Some("RunMat:feof:InvalidInput"),
    when: "Input identifier is malformed or out of range.",
    message: "feof: invalid file identifier input",
};
pub const FEOF_ERROR_INVALID_IDENTIFIER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEOF.INVALID_IDENTIFIER",
    identifier: Some("RunMat:feof:InvalidIdentifier"),
    when: "Identifier does not refer to an open file handle.",
    message: "feof: invalid file identifier. Use fopen to generate a valid file ID.",
};
pub const FEOF_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FEOF.INTERNAL",
    identifier: None,
    when: "Internal control-flow conversion failed.",
    message: "feof: internal error",
};
pub const FEOF_ERRORS: [BuiltinErrorDescriptor; 3] = [
    FEOF_ERROR_INVALID_INPUT,
    FEOF_ERROR_INVALID_IDENTIFIER,
    FEOF_ERROR_INTERNAL,
];
pub const FEOF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FEOF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FEOF_ERRORS,
};
