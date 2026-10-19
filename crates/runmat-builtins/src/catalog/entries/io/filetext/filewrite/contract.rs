use crate::*;

pub const FILEWRITE_NATIVE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "filewrite-runmat-native",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "filewrite is a RunMat-native file-writing API with no MATLAB builtin counterpart",
    error_identifier: Some("RunMat:compatibility:FilewriteNativeExtension"),
};

pub const FILEWRITE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [FILEWRITE_NATIVE_EXTENSION];

const FILEWRITE_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "data",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes are read from authoritative storage and each value must lie in the inclusive byte range 0..255.",
    }];

pub const FILEWRITE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "count = filewrite(filename, integer_data, ...)",
        inputs: &FILEWRITE_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The complete filewrite surface is a gated RunMat-native extension; integer data is validated exactly before encoding and host I/O, resident data gathers to the host, and the byte count is a host double scalar.",
    }];

const FILEWRITE_OUTPUT_COUNT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "count",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of bytes written.",
}];
const FILEWRITE_INPUTS_FILENAME_DATA: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Path to target file.",
    },
    BuiltinParamDescriptor {
        name: "data",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Text or uint8-compatible data to write.",
    },
];
const FILEWRITE_INPUTS_FILENAME_DATA_ENCODING: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Path to target file.",
    },
    BuiltinParamDescriptor {
        name: "data",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Text or uint8-compatible data to write.",
    },
    BuiltinParamDescriptor {
        name: "encoding",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"auto\""),
        description: "Positional encoding label.",
    },
];
const FILEWRITE_INPUTS_FILENAME_DATA_OPTIONS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Path to target file.",
    },
    BuiltinParamDescriptor {
        name: "data",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Text or uint8-compatible data to write.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::PropertyName,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option name: 'Encoding' or 'WriteMode'.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::PropertyValue,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option value associated with option name.",
    },
];
const FILEWRITE_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "count = filewrite(filename, data)",
        inputs: &FILEWRITE_INPUTS_FILENAME_DATA,
        outputs: &FILEWRITE_OUTPUT_COUNT,
    },
    BuiltinSignatureDescriptor {
        label: "count = filewrite(filename, data, encoding)",
        inputs: &FILEWRITE_INPUTS_FILENAME_DATA_ENCODING,
        outputs: &FILEWRITE_OUTPUT_COUNT,
    },
    BuiltinSignatureDescriptor {
        label: "count = filewrite(filename, data, \"Encoding\", encoding)",
        inputs: &FILEWRITE_INPUTS_FILENAME_DATA_OPTIONS,
        outputs: &FILEWRITE_OUTPUT_COUNT,
    },
    BuiltinSignatureDescriptor {
        label: "count = filewrite(filename, data, ..., \"WriteMode\", mode)",
        inputs: &FILEWRITE_INPUTS_FILENAME_DATA_OPTIONS,
        outputs: &FILEWRITE_OUTPUT_COUNT,
    },
];

pub const FILEWRITE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILEWRITE.INVALID_INPUT",
    identifier: Some("RunMat:filewrite:InvalidInput"),
    when: "Filename/data argument constraints are violated.",
    message: "filewrite: invalid input arguments",
};
pub const FILEWRITE_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILEWRITE.INVALID_OPTION",
    identifier: Some("RunMat:filewrite:InvalidOption"),
    when: "Name/value options are malformed or unsupported.",
    message: "filewrite: invalid option configuration",
};
pub const FILEWRITE_ERROR_INVALID_DATA: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILEWRITE.INVALID_DATA",
    identifier: Some("RunMat:filewrite:InvalidData"),
    when: "Input data cannot be converted to writable bytes.",
    message: "filewrite: unsupported data payload",
};
pub const FILEWRITE_ERROR_ENCODE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILEWRITE.ENCODE",
    identifier: Some("RunMat:filewrite:EncodeFailed"),
    when: "Encoding selected output bytes/chars fails.",
    message: "filewrite: failed to encode payload",
};
pub const FILEWRITE_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILEWRITE.IO",
    identifier: Some("RunMat:filewrite:IoFailure"),
    when: "Filesystem open/write/flush operation fails.",
    message: "filewrite: file I/O failed",
};
pub const FILEWRITE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILEWRITE.INTERNAL",
    identifier: None,
    when: "Internal runtime control-flow or conversion fails.",
    message: "filewrite: internal error",
};
const FILEWRITE_ERRORS: [BuiltinErrorDescriptor; 6] = [
    FILEWRITE_ERROR_INVALID_INPUT,
    FILEWRITE_ERROR_INVALID_OPTION,
    FILEWRITE_ERROR_INVALID_DATA,
    FILEWRITE_ERROR_ENCODE,
    FILEWRITE_ERROR_IO,
    FILEWRITE_ERROR_INTERNAL,
];
pub const FILEWRITE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FILEWRITE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FILEWRITE_ERRORS,
};
