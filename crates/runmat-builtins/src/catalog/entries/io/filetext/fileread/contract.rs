use crate::*;

const FILEREAD_OUTPUT_TEXT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "text",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "File contents as a 1-by-N character vector.",
}];
const FILEREAD_INPUTS_FILENAME: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Path to a readable file.",
}];
const FILEREAD_INPUTS_FILENAME_ENCODING: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Path to a readable file.",
    },
    BuiltinParamDescriptor {
        name: "encoding",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"auto\""),
        description: "Encoding label (for example 'utf-8', 'latin1', 'ascii', 'raw').",
    },
];
const FILEREAD_INPUTS_FILENAME_ENCODING_PAIR: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Path to a readable file.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::PropertyName,
        arity: BuiltinParamArity::Optional,
        default: Some("\"Encoding\""),
        description: "Name of supported option; currently only 'Encoding'.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::PropertyValue,
        arity: BuiltinParamArity::Optional,
        default: Some("\"auto\""),
        description: "Option value for the provided option name.",
    },
];
const FILEREAD_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "text = fileread(filename)",
        inputs: &FILEREAD_INPUTS_FILENAME,
        outputs: &FILEREAD_OUTPUT_TEXT,
    },
    BuiltinSignatureDescriptor {
        label: "text = fileread(filename, encoding)",
        inputs: &FILEREAD_INPUTS_FILENAME_ENCODING,
        outputs: &FILEREAD_OUTPUT_TEXT,
    },
    BuiltinSignatureDescriptor {
        label: "text = fileread(filename, \"Encoding\", encoding)",
        inputs: &FILEREAD_INPUTS_FILENAME_ENCODING_PAIR,
        outputs: &FILEREAD_OUTPUT_TEXT,
    },
];

pub const FILEREAD_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILEREAD.INVALID_INPUT",
    identifier: Some("RunMat:fileread:InvalidInput"),
    when: "Filename or argument cardinality/type constraints are violated.",
    message: "fileread: invalid input arguments",
};
pub const FILEREAD_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILEREAD.INVALID_OPTION",
    identifier: Some("RunMat:fileread:InvalidOption"),
    when: "Encoding option syntax or value is invalid.",
    message: "fileread: invalid option configuration",
};
pub const FILEREAD_ERROR_DECODE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILEREAD.DECODE",
    identifier: Some("RunMat:fileread:DecodeFailed"),
    when: "Requested decoding of bytes fails (for example UTF-8 or ASCII mismatch).",
    message: "fileread: unable to decode file contents",
};
pub const FILEREAD_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILEREAD.IO",
    identifier: Some("RunMat:fileread:IoFailure"),
    when: "Filesystem read operation fails.",
    message: "fileread: file read failed",
};
pub const FILEREAD_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILEREAD.INTERNAL",
    identifier: None,
    when: "Internal runtime control-flow or conversion fails.",
    message: "fileread: internal error",
};
const FILEREAD_ERRORS: [BuiltinErrorDescriptor; 5] = [
    FILEREAD_ERROR_INVALID_INPUT,
    FILEREAD_ERROR_INVALID_OPTION,
    FILEREAD_ERROR_DECODE,
    FILEREAD_ERROR_IO,
    FILEREAD_ERROR_INTERNAL,
];
pub const FILEREAD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FILEREAD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FILEREAD_ERRORS,
};

pub const FILEREAD_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "fileread accepts host-text filename and Encoding inputs and returns a character row; numeric and provider-resident values are rejected before any gather, provider access, or filesystem read.",
    };
