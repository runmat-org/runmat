use crate::*;

pub const FPRINTF_INTEGER_ID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fprintf-integer-fileid",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "integer-class fprintf file identifiers are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FprintfIntegerIdExtension"),
};
pub const FPRINTF_SINGLE_ID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fprintf-single-fileid",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "single-precision fprintf file identifiers are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FprintfSingleIdExtension"),
};
pub const FPRINTF_RESIDENT_ID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fprintf-resident-fileid",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "provider-resident fprintf file identifiers are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FprintfResidentIdExtension"),
};
pub const FPRINTF_INTEGER_FORMAT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fprintf-integer-format",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "numeric integer-code fprintf format specifications are a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FprintfIntegerFormatExtension"),
    };
pub const FPRINTF_NUMERIC_FORMAT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fprintf-numeric-format",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "numeric-code fprintf format tensors are a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FprintfNumericFormatExtension"),
    };
pub const FPRINTF_RESIDENT_FORMAT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fprintf-resident-format",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "provider-resident fprintf format specifications are a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FprintfResidentFormatExtension"),
    };
pub const FPRINTF_STREAM_LABEL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fprintf-stream-label",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "textual stdout and stderr fprintf targets are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FprintfStreamLabelExtension"),
};
pub const FPRINTF_EXTENSIONS: [BuiltinExtensionDescriptor; 7] = [
    FPRINTF_INTEGER_ID_EXTENSION,
    FPRINTF_SINGLE_ID_EXTENSION,
    FPRINTF_RESIDENT_ID_EXTENSION,
    FPRINTF_INTEGER_FORMAT_EXTENSION,
    FPRINTF_NUMERIC_FORMAT_EXTENSION,
    FPRINTF_RESIDENT_FORMAT_EXTENSION,
    FPRINTF_STREAM_LABEL_EXTENSION,
];

const FPRINTF_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target documents all eight integer classes; integer conversions format authoritative values exactly and arrays traverse in column-major order.",
    }];
const FPRINTF_INTEGER_ID_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "fileID",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target documents double identifiers; typed integer identifiers are an independently gated extension.",
    }];
const FPRINTF_INTEGER_FORMAT_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "formatSpec",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Rejected,
        notes: "The compatibility target documents character or string format specifications; numeric code vectors are a gated RunMat extension.",
    }];
pub const FPRINTF_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "count = fprintf(formatSpec, integer_data...)",
        inputs: &FPRINTF_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "The host sink returns a double byte count; documented resident data gathers without changing the caller-owned value.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "count = fprintf(integer_fileID, formatSpec, A...)",
        inputs: &FPRINTF_INTEGER_ID_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The identifier is range-checked exactly before registry access.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "count = fprintf(integer_formatSpec, A...)",
        inputs: &FPRINTF_INTEGER_FORMAT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Integer code points are validated exactly before conversion to a host format string.",
    },
];

const FPRINTF_OUTPUT_COUNT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "count",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of bytes written.",
}];
const FPRINTF_INPUTS_FORMAT_VARIADIC: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "formatSpec",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Format string or character row vector.",
    },
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Values consumed by conversion specifiers.",
    },
];
const FPRINTF_INPUTS_FID_FORMAT_VARIADIC: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "fid_or_stream",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: Some("1"),
        description: "Numeric file identifier, or stream label ('stdout'|'stderr').",
    },
    BuiltinParamDescriptor {
        name: "formatSpec",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Format string or character row vector.",
    },
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Values consumed by conversion specifiers.",
    },
];
const FPRINTF_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "count = fprintf(formatSpec, A...)",
        inputs: &FPRINTF_INPUTS_FORMAT_VARIADIC,
        outputs: &FPRINTF_OUTPUT_COUNT,
    },
    BuiltinSignatureDescriptor {
        label: "count = fprintf(fid_or_stream, formatSpec, A...)",
        inputs: &FPRINTF_INPUTS_FID_FORMAT_VARIADIC,
        outputs: &FPRINTF_OUTPUT_COUNT,
    },
];

pub const FPRINTF_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FPRINTF.INVALID_INPUT",
    identifier: Some("RunMat:fprintf:InvalidInput"),
    when: "Argument count/type does not satisfy fprintf requirements.",
    message: "fprintf: invalid input arguments",
};
pub const FPRINTF_ERROR_INVALID_IDENTIFIER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FPRINTF.INVALID_IDENTIFIER",
    identifier: Some("RunMat:fprintf:InvalidIdentifier"),
    when: "File identifier is invalid or not writable.",
    message: "fprintf: invalid file identifier. Use fopen to generate a valid file ID.",
};
pub const FPRINTF_ERROR_FORMAT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FPRINTF.FORMAT",
    identifier: Some("RunMat:fprintf:InvalidFormat"),
    when: "Format string parsing or placeholder consumption fails.",
    message: "fprintf: invalid format specification",
};
pub const FPRINTF_ERROR_ENCODE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FPRINTF.ENCODE",
    identifier: Some("RunMat:fprintf:EncodeFailed"),
    when: "Rendered text cannot be encoded for destination stream/file encoding.",
    message: "fprintf: failed to encode output",
};
pub const FPRINTF_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FPRINTF.IO",
    identifier: Some("RunMat:fprintf:IoFailure"),
    when: "Write to target stream/file fails.",
    message: "fprintf: write failed",
};
pub const FPRINTF_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FPRINTF.INTERNAL",
    identifier: None,
    when: "Internal runtime control-flow or conversion fails.",
    message: "fprintf: internal error",
};
const FPRINTF_ERRORS: [BuiltinErrorDescriptor; 6] = [
    FPRINTF_ERROR_INVALID_INPUT,
    FPRINTF_ERROR_INVALID_IDENTIFIER,
    FPRINTF_ERROR_FORMAT,
    FPRINTF_ERROR_ENCODE,
    FPRINTF_ERROR_IO,
    FPRINTF_ERROR_INTERNAL,
];
pub const FPRINTF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FPRINTF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FPRINTF_ERRORS,
};
