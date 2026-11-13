use crate::*;
const UIGETFILE_OUTPUT_FILE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "file",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Selected filename, cell array of filenames for multiple selections, or 0 when cancelled.",
}];

const UIGETFILE_OUTPUT_FILE_PATH: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "file",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Selected filename, cell array of filenames for multiple selections, or 0 when cancelled.",
    },
    BuiltinParamDescriptor {
        name: "path",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Directory containing the selected file, including a trailing separator, or 0 when cancelled.",
    },
];

const UIGETFILE_OUTPUT_FILE_PATH_INDEX: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "file",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Selected filename, cell array of filenames for multiple selections, or 0 when cancelled.",
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

const UIGETFILE_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const UIGETFILE_INPUTS_FILTER: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filter",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: Some("\"*.*\""),
    description: "File extension pattern, semicolon-delimited patterns, or an N-by-1/N-by-2 cell array of patterns and descriptions.",
}];

const UIGETFILE_INPUTS_FILTER_TITLE: [BuiltinParamDescriptor; 2] = [
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

const UIGETFILE_INPUTS_FILTER_TITLE_DEFAULT: [BuiltinParamDescriptor; 3] = [
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

const UIGETFILE_INPUTS_MULTISELECT: [BuiltinParamDescriptor; 5] = [
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
    BuiltinParamDescriptor {
        name: "optionName",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"MultiSelect\""),
        description: "Name-value option name. `MultiSelect` is supported.",
    },
    BuiltinParamDescriptor {
        name: "optionValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("\"off\""),
        description: "`\"on\"`, `\"off\"`, or a scalar logical/numeric value.",
    },
];

const UIGETFILE_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "file = uigetfile()",
        inputs: &UIGETFILE_INPUTS_NONE,
        outputs: &UIGETFILE_OUTPUT_FILE,
    },
    BuiltinSignatureDescriptor {
        label: "file = uigetfile(filter)",
        inputs: &UIGETFILE_INPUTS_FILTER,
        outputs: &UIGETFILE_OUTPUT_FILE,
    },
    BuiltinSignatureDescriptor {
        label: "[file, path] = uigetfile(filter, title)",
        inputs: &UIGETFILE_INPUTS_FILTER_TITLE,
        outputs: &UIGETFILE_OUTPUT_FILE_PATH,
    },
    BuiltinSignatureDescriptor {
        label: "[file, path, index] = uigetfile(filter, title, defaultName)",
        inputs: &UIGETFILE_INPUTS_FILTER_TITLE_DEFAULT,
        outputs: &UIGETFILE_OUTPUT_FILE_PATH_INDEX,
    },
    BuiltinSignatureDescriptor {
        label:
            "[file, path, index] = uigetfile(filter, title, defaultName, \"MultiSelect\", value)",
        inputs: &UIGETFILE_INPUTS_MULTISELECT,
        outputs: &UIGETFILE_OUTPUT_FILE_PATH_INDEX,
    },
];

pub const UIGETFILE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UIGETFILE.INVALID_ARGUMENT",
    identifier: Some("RunMat:uigetfile:InvalidArgument"),
    when: "A filter, title, default path, or name-value option has an unsupported type or shape.",
    message: "uigetfile: invalid argument",
};

pub const UIGETFILE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UIGETFILE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:uigetfile:TooManyOutputs"),
    when: "More than three output arguments are requested.",
    message: "uigetfile: too many output arguments",
};

pub const UIGETFILE_ERROR_HOST: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UIGETFILE.HOST",
    identifier: Some("RunMat:uigetfile:HostError"),
    when: "The active filesystem provider fails while opening the host file-selection UI.",
    message: "uigetfile: file selection failed",
};

pub const UIGETFILE_ERROR_INVALID_SELECTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UIGETFILE.INVALID_SELECTION",
    identifier: Some("RunMat:uigetfile:InvalidSelection"),
    when: "The active filesystem provider returns a malformed or internally inconsistent file selection.",
    message: "uigetfile: invalid file selection",
};

const UIGETFILE_ERRORS: [BuiltinErrorDescriptor; 4] = [
    UIGETFILE_ERROR_INVALID_ARGUMENT,
    UIGETFILE_ERROR_TOO_MANY_OUTPUTS,
    UIGETFILE_ERROR_HOST,
    UIGETFILE_ERROR_INVALID_SELECTION,
];

pub const UIGETFILE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &UIGETFILE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &UIGETFILE_ERRORS,
};

pub const UIGETFILE_MULTISELECT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
 id: "uigetfile-multiselect-controls",
 mode: BuiltinExtensionMode::RunMatOnly,
 description: "Logical and numeric MultiSelect controls and true/false/yes/no text aliases are RunMat extensions.",
 error_identifier: Some("RunMat:compatibility:UigetfileMultiSelectExtension"),
};
pub(super) const INTEGER_CAPABILITIES: &[BuiltinIntegerCapabilityDescriptor] = &[BuiltinIntegerCapabilityDescriptor {
 form: "uigetfile(..., 'MultiSelect', integer_scalar)",
 inputs: &[BuiltinIntegerInputCapability {
 name: "MultiSelect", classes: &ALL_INTEGER_CLASSES,
 availability: BuiltinIntegerInputAvailability::RunMatOnly,
 scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
 notes: "A native integer scalar is tested for exact zero without a floating-point conversion.",
 }],
 computation_domain: BuiltinIntegerComputationDomain::Structural,
 output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
 overflow: BuiltinIntegerOverflowRule::NotApplicable,
 backend: BuiltinIntegerBackendRule::HostOnly,
 overload: BuiltinIntegerOverloadKind::StructuralParameter,
 notes: "Integer controls choose single/multiple selection; filename/path results remain text or cancellation zeros and filter index remains double.",
}];
