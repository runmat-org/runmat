use crate::*;
const UIPUTFILE_OUTPUT_FILE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "file",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Selected filename, or 0 when cancelled.",
}];

const UIPUTFILE_OUTPUT_FILE_PATH: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "file",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Selected filename, or 0 when cancelled.",
    },
    BuiltinParamDescriptor {
        name: "path",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Directory containing the selected file, including a trailing separator, or 0 when cancelled.",
    },
];

const UIPUTFILE_OUTPUT_FILE_PATH_INDEX: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "file",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Selected filename, or 0 when cancelled.",
    },
    BuiltinParamDescriptor {
        name: "path",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Directory containing the selected file, including a trailing separator, or 0 when cancelled.",
    },
    BuiltinParamDescriptor {
        name: "index",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-based selected filter index, or 0 when cancelled.",
    },
];

const UIPUTFILE_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const UIPUTFILE_INPUTS_FILTER: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filter",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: Some("\"*.*\""),
    description: "File extension pattern, semicolon-delimited patterns, or an N-by-1/N-by-2 cell array of patterns and descriptions.",
}];

const UIPUTFILE_INPUTS_FILTER_TITLE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filter",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("\"*.*\""),
        description: "File extension pattern, semicolon-delimited patterns, or an N-by-1/N-by-2 cell array of patterns and descriptions.",
    },
    BuiltinParamDescriptor {
        name: "title",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Dialog title.",
    },
];

const UIPUTFILE_INPUTS_FILTER_TITLE_DEFAULT: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "filter",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("\"*.*\""),
        description: "File extension pattern, semicolon-delimited patterns, or an N-by-1/N-by-2 cell array of patterns and descriptions.",
    },
    BuiltinParamDescriptor {
        name: "title",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Dialog title.",
    },
    BuiltinParamDescriptor {
        name: "defaultName",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Initial file or directory path for the dialog.",
    },
];

const UIPUTFILE_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "file = uiputfile()",
        inputs: &UIPUTFILE_INPUTS_NONE,
        outputs: &UIPUTFILE_OUTPUT_FILE,
    },
    BuiltinSignatureDescriptor {
        label: "file = uiputfile(filter)",
        inputs: &UIPUTFILE_INPUTS_FILTER,
        outputs: &UIPUTFILE_OUTPUT_FILE,
    },
    BuiltinSignatureDescriptor {
        label: "[file, path] = uiputfile(filter, title)",
        inputs: &UIPUTFILE_INPUTS_FILTER_TITLE,
        outputs: &UIPUTFILE_OUTPUT_FILE_PATH,
    },
    BuiltinSignatureDescriptor {
        label: "[file, path, index] = uiputfile(filter, title, defaultName)",
        inputs: &UIPUTFILE_INPUTS_FILTER_TITLE_DEFAULT,
        outputs: &UIPUTFILE_OUTPUT_FILE_PATH_INDEX,
    },
];

pub const UIPUTFILE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UIPUTFILE.INVALID_ARGUMENT",
    identifier: Some("RunMat:uiputfile:InvalidArgument"),
    when: "A filter, title, or default path has an unsupported type or shape.",
    message: "uiputfile: invalid argument",
};

pub const UIPUTFILE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UIPUTFILE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:uiputfile:TooManyOutputs"),
    when: "More than three output arguments are requested.",
    message: "uiputfile: too many output arguments",
};

pub const UIPUTFILE_ERROR_HOST: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UIPUTFILE.HOST",
    identifier: Some("RunMat:uiputfile:HostError"),
    when: "The active filesystem provider fails while opening the host save-file UI.",
    message: "uiputfile: file selection failed",
};

pub const UIPUTFILE_ERROR_INVALID_SELECTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UIPUTFILE.INVALID_SELECTION",
    identifier: Some("RunMat:uiputfile:InvalidSelection"),
    when: "The active filesystem provider returns a malformed save-file selection.",
    message: "uiputfile: invalid file selection",
};

const UIPUTFILE_ERRORS: [BuiltinErrorDescriptor; 4] = [
    UIPUTFILE_ERROR_INVALID_ARGUMENT,
    UIPUTFILE_ERROR_TOO_MANY_OUTPUTS,
    UIPUTFILE_ERROR_HOST,
    UIPUTFILE_ERROR_INVALID_SELECTION,
];

pub const UIPUTFILE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &UIPUTFILE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &UIPUTFILE_ERRORS,
};

pub const UIPUTFILE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "uiputfile accepts textual filter, title, and default-name inputs. Its filter-index output is an integer-valued double scalar, not a native integer-class input boundary.",
    };
