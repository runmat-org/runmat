use crate::*;

macro_rules! fgets_extension {
    ($name:ident, $id:literal, $description:literal, $error:literal) => {
        pub const $name: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
            id: $id,
            mode: BuiltinExtensionMode::RunMatOnly,
            description: $description,
            error_identifier: Some($error),
        };
    };
}
fgets_extension!(
    FGETS_INTEGER_ID_EXTENSION,
    "fgets-integer-fileid",
    "integer-class fgets file identifiers are a RunMat extension",
    "RunMat:compatibility:FgetsIntegerIdExtension"
);
fgets_extension!(
    FGETS_INTEGER_NCHAR_EXTENSION,
    "fgets-integer-nchar",
    "integer-class fgets nchar values are a RunMat extension",
    "RunMat:compatibility:FgetsIntegerNcharExtension"
);
fgets_extension!(
    FGETS_SINGLE_ID_EXTENSION,
    "fgets-single-fileid",
    "single-precision fgets file identifiers are a RunMat extension",
    "RunMat:compatibility:FgetsSingleIdExtension"
);
fgets_extension!(
    FGETS_SINGLE_NCHAR_EXTENSION,
    "fgets-single-nchar",
    "single-precision fgets nchar values are a RunMat extension",
    "RunMat:compatibility:FgetsSingleNcharExtension"
);
fgets_extension!(
    FGETS_RESIDENT_ID_EXTENSION,
    "fgets-resident-fileid",
    "provider-resident fgets file identifiers are a RunMat extension",
    "RunMat:compatibility:FgetsResidentIdExtension"
);
fgets_extension!(
    FGETS_RESIDENT_NCHAR_EXTENSION,
    "fgets-resident-nchar",
    "provider-resident fgets nchar values are a RunMat extension",
    "RunMat:compatibility:FgetsResidentNcharExtension"
);
pub const FGETS_EXTENSIONS: [BuiltinExtensionDescriptor; 6] = [
    FGETS_INTEGER_ID_EXTENSION,
    FGETS_INTEGER_NCHAR_EXTENSION,
    FGETS_SINGLE_ID_EXTENSION,
    FGETS_SINGLE_NCHAR_EXTENSION,
    FGETS_RESIDENT_ID_EXTENSION,
    FGETS_RESIDENT_NCHAR_EXTENSION,
];

const FGETS_INTEGER_ID_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "fileID",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "MATLAB-compatible file identifiers are integer-valued doubles; all eight typed integer classes are one independently gated RunMat extension and are checked exactly against the registry identifier domain.",
    }];
const FGETS_INTEGER_NCHAR_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "nchar",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "MATLAB-compatible nchar values are non-negative integer-valued doubles; all eight typed integer classes are one independently gated RunMat extension and are converted exactly to the runtime length domain.",
    }];
pub const FGETS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "tline = fgets(integer_fileID, ...)",
        inputs: &FGETS_INTEGER_ID_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The identifier is validated exactly before registry access; a successful read returns a host character row and end-of-file returns the double sentinel -1.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "tline = fgets(fileID, integer_nchar)",
        inputs: &FGETS_INTEGER_NCHAR_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The character limit is validated exactly before reading; line and terminator outputs remain host values independent of the nchar class.",
    },
];

const FGETS_OUTPUT_LINE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tline",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Next line including terminators, or -1 at end-of-file.",
}];
const FGETS_OUTPUT_LINE_TERMINATORS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "tline",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Next line including terminators, or -1 at end-of-file.",
    },
    BuiltinParamDescriptor {
        name: "terminators",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Numeric row vector of terminator byte values.",
    },
];
const FGETS_INPUTS_FID: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fid",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "File identifier opened by fopen.",
}];
const FGETS_INPUTS_FID_NCHAR: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "nchar",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Maximum number of characters to read.",
    },
];
const FGETS_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "tline = fgets(fid)",
        inputs: &FGETS_INPUTS_FID,
        outputs: &FGETS_OUTPUT_LINE,
    },
    BuiltinSignatureDescriptor {
        label: "tline = fgets(fid, nchar)",
        inputs: &FGETS_INPUTS_FID_NCHAR,
        outputs: &FGETS_OUTPUT_LINE,
    },
    BuiltinSignatureDescriptor {
        label: "[tline, terminators] = fgets(fid, ...)",
        inputs: &FGETS_INPUTS_FID_NCHAR,
        outputs: &FGETS_OUTPUT_LINE_TERMINATORS,
    },
];

pub const FGETS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FGETS.INVALID_INPUT",
    identifier: Some("RunMat:fgets:InvalidInput"),
    when: "Input argument count or scalar constraints are invalid.",
    message: "fgets: invalid input arguments",
};
pub const FGETS_ERROR_INVALID_IDENTIFIER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FGETS.INVALID_IDENTIFIER",
    identifier: Some("RunMat:fgets:InvalidIdentifier"),
    when: "Identifier does not refer to an open readable file.",
    message: "fgets: invalid file identifier. Use fopen to generate a valid file ID.",
};
pub const FGETS_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FGETS.IO",
    identifier: Some("RunMat:fgets:IoFailure"),
    when: "File read or decode operation failed.",
    message: "fgets: file I/O failed",
};
pub const FGETS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FGETS.INTERNAL",
    identifier: None,
    when: "Internal control-flow conversion failed.",
    message: "fgets: internal error",
};
const FGETS_ERRORS: [BuiltinErrorDescriptor; 4] = [
    FGETS_ERROR_INVALID_INPUT,
    FGETS_ERROR_INVALID_IDENTIFIER,
    FGETS_ERROR_IO,
    FGETS_ERROR_INTERNAL,
];
pub const FGETS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FGETS_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FGETS_ERRORS,
};
