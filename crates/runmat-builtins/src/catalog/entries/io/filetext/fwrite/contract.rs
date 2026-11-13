use crate::*;
pub const FWRITE_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fwrite-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "direct fwrite of gpuArray input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FwriteGpuInputExtension"),
};
pub const FWRITE_INTEGER_ID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fwrite-integer-fileid",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "integer-class fwrite file identifiers are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FwriteIntegerIdExtension"),
};
pub const FWRITE_INTEGER_SKIP_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fwrite-integer-skip",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "typed integer fwrite skip controls are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FwriteIntegerSkipExtension"),
};
pub const FWRITE_LOGICAL_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fwrite-logical-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "logical fwrite file identifiers and skips are a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FwriteLogicalControlExtension"),
    };
pub const FWRITE_SINGLE_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fwrite-single-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "single-precision fwrite file identifiers and skips are a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FwriteSingleControlExtension"),
    };
pub const FWRITE_RESIDENT_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fwrite-resident-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "provider-resident fwrite control arguments are a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FwriteResidentControlExtension"),
    };
pub const FWRITE_ARROW_PRECISION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fwrite-arrow-precision",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "fread-style source=>output fwrite precision syntax is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FwriteArrowPrecisionExtension"),
    };

pub const FWRITE_EXTENSIONS: [BuiltinExtensionDescriptor; 7] = [
    FWRITE_GPU_INPUT_EXTENSION,
    FWRITE_INTEGER_ID_EXTENSION,
    FWRITE_INTEGER_SKIP_EXTENSION,
    FWRITE_LOGICAL_CONTROL_EXTENSION,
    FWRITE_SINGLE_CONTROL_EXTENSION,
    FWRITE_RESIDENT_CONTROL_EXTENSION,
    FWRITE_ARROW_PRECISION_EXTENSION,
];

const FWRITE_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target documents all eight integer data classes; exact source values convert directly to the selected binary precision.",
    }];
const FWRITE_INTEGER_ID_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "fileID",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target documents double identifiers; typed integer identifiers are independently gated.",
    }];
const FWRITE_INTEGER_SKIP_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "skip",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target documents a double skip scalar; typed integer skips are independently gated and range-checked exactly.",
    }];
pub const FWRITE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "count = fwrite(fileID, integer_A, precision, ...)",
        inputs: &FWRITE_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Integer data remains authoritative through binary encoding; narrowing saturates and count is double.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "count = fwrite(integer_fileID, A, ...)",
        inputs: &FWRITE_INTEGER_ID_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The identifier is validated exactly before registry access.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "count = fwrite(fileID, A, precision, integer_skip, ...)",
        inputs: &FWRITE_INTEGER_SKIP_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The skip is validated exactly in the host seek domain.",
    },
];

const FWRITE_OUTPUT_COUNT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "count",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of elements successfully written.",
}];
const FWRITE_INPUTS_FID_DATA: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "data",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric/logical/text payload to write.",
    },
];
const FWRITE_INPUTS_FID_DATA_PRECISION: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "data",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric/logical/text payload to write.",
    },
    BuiltinParamDescriptor {
        name: "precision",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"uint8\""),
        description: "Write precision label (for example \"uint8\", \"double\").",
    },
];
const FWRITE_INPUTS_FID_DATA_PRECISION_SKIP: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "data",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric/logical/text payload to write.",
    },
    BuiltinParamDescriptor {
        name: "precision",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"uint8\""),
        description: "Write precision label (for example \"uint8\", \"double\").",
    },
    BuiltinParamDescriptor {
        name: "skip",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("0"),
        description: "Bytes skipped after each element written.",
    },
];
const FWRITE_INPUTS_FID_DATA_PRECISION_MACHINEFMT: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "data",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric/logical/text payload to write.",
    },
    BuiltinParamDescriptor {
        name: "precision",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"uint8\""),
        description: "Write precision label (for example \"uint8\", \"double\").",
    },
    BuiltinParamDescriptor {
        name: "machinefmt",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"native\""),
        description: "Machine format label (native/little-endian/big-endian aliases).",
    },
];
const FWRITE_INPUTS_FID_DATA_PRECISION_SKIP_MACHINEFMT: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier opened by fopen.",
    },
    BuiltinParamDescriptor {
        name: "data",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric/logical/text payload to write.",
    },
    BuiltinParamDescriptor {
        name: "precision",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"uint8\""),
        description: "Write precision label (for example \"uint8\", \"double\").",
    },
    BuiltinParamDescriptor {
        name: "skip",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("0"),
        description: "Bytes skipped after each element written.",
    },
    BuiltinParamDescriptor {
        name: "machinefmt",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"native\""),
        description: "Machine format label (native/little-endian/big-endian aliases).",
    },
];
const FWRITE_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "count = fwrite(fid, data)",
        inputs: &FWRITE_INPUTS_FID_DATA,
        outputs: &FWRITE_OUTPUT_COUNT,
    },
    BuiltinSignatureDescriptor {
        label: "count = fwrite(fid, data, precision)",
        inputs: &FWRITE_INPUTS_FID_DATA_PRECISION,
        outputs: &FWRITE_OUTPUT_COUNT,
    },
    BuiltinSignatureDescriptor {
        label: "count = fwrite(fid, data, precision, skip)",
        inputs: &FWRITE_INPUTS_FID_DATA_PRECISION_SKIP,
        outputs: &FWRITE_OUTPUT_COUNT,
    },
    BuiltinSignatureDescriptor {
        label: "count = fwrite(fid, data, precision, machinefmt)",
        inputs: &FWRITE_INPUTS_FID_DATA_PRECISION_MACHINEFMT,
        outputs: &FWRITE_OUTPUT_COUNT,
    },
    BuiltinSignatureDescriptor {
        label: "count = fwrite(fid, data, precision, skip, machinefmt)",
        inputs: &FWRITE_INPUTS_FID_DATA_PRECISION_SKIP_MACHINEFMT,
        outputs: &FWRITE_OUTPUT_COUNT,
    },
];

pub const FWRITE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FWRITE.INVALID_INPUT",
    identifier: Some("RunMat:fwrite:InvalidInput"),
    when: "Identifier, payload, or argument cardinality/type constraints are violated.",
    message: "fwrite: invalid input arguments",
};
pub const FWRITE_ERROR_INVALID_IDENTIFIER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FWRITE.INVALID_IDENTIFIER",
    identifier: Some("RunMat:fwrite:InvalidIdentifier"),
    when: "Identifier does not refer to a writable open file.",
    message: "fwrite: invalid file identifier. Use fopen to generate a valid file ID.",
};
pub const FWRITE_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FWRITE.INVALID_OPTION",
    identifier: Some("RunMat:fwrite:InvalidOption"),
    when: "Precision, skip, or machine format options are invalid.",
    message: "fwrite: invalid option configuration",
};
pub const FWRITE_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FWRITE.IO",
    identifier: Some("RunMat:fwrite:IoFailure"),
    when: "Write/seek operation fails.",
    message: "fwrite: file write failed",
};
pub const FWRITE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FWRITE.INTERNAL",
    identifier: None,
    when: "Internal runtime control-flow conversion fails.",
    message: "fwrite: internal error",
};
const FWRITE_ERRORS: [BuiltinErrorDescriptor; 5] = [
    FWRITE_ERROR_INVALID_INPUT,
    FWRITE_ERROR_INVALID_IDENTIFIER,
    FWRITE_ERROR_INVALID_OPTION,
    FWRITE_ERROR_IO,
    FWRITE_ERROR_INTERNAL,
];
pub const FWRITE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FWRITE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FWRITE_ERRORS,
};
