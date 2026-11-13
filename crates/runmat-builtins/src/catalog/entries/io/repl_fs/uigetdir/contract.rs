use crate::*;
const UIGETDIR_OUTPUT_DIR: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "folder",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Selected folder path as a character vector, or 0 when cancelled.",
}];

const UIGETDIR_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const UIGETDIR_INPUTS_START_PATH: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "startPath",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Initial folder shown by the dialog.",
}];

const UIGETDIR_INPUTS_START_PATH_TITLE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "startPath",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Initial folder shown by the dialog.",
    },
    BuiltinParamDescriptor {
        name: "title",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Dialog title.",
    },
];

const UIGETDIR_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "folder = uigetdir()",
        inputs: &UIGETDIR_INPUTS_NONE,
        outputs: &UIGETDIR_OUTPUT_DIR,
    },
    BuiltinSignatureDescriptor {
        label: "folder = uigetdir(startPath)",
        inputs: &UIGETDIR_INPUTS_START_PATH,
        outputs: &UIGETDIR_OUTPUT_DIR,
    },
    BuiltinSignatureDescriptor {
        label: "folder = uigetdir(startPath, title)",
        inputs: &UIGETDIR_INPUTS_START_PATH_TITLE,
        outputs: &UIGETDIR_OUTPUT_DIR,
    },
];

pub const UIGETDIR_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UIGETDIR.INVALID_ARGUMENT",
    identifier: Some("RunMat:uigetdir:InvalidArgument"),
    when: "The start path or title has an unsupported type or shape.",
    message: "uigetdir: invalid argument",
};

pub const UIGETDIR_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UIGETDIR.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:uigetdir:TooManyOutputs"),
    when: "More than one output argument is requested.",
    message: "uigetdir: too many output arguments",
};

pub const UIGETDIR_ERROR_HOST: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UIGETDIR.HOST",
    identifier: Some("RunMat:uigetdir:HostError"),
    when: "The active filesystem provider fails while opening the host folder-selection UI.",
    message: "uigetdir: folder selection failed",
};

pub const UIGETDIR_ERROR_INVALID_SELECTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UIGETDIR.INVALID_SELECTION",
    identifier: Some("RunMat:uigetdir:InvalidSelection"),
    when: "The active filesystem provider returns a malformed folder selection.",
    message: "uigetdir: invalid folder selection",
};

const UIGETDIR_ERRORS: [BuiltinErrorDescriptor; 4] = [
    UIGETDIR_ERROR_INVALID_ARGUMENT,
    UIGETDIR_ERROR_TOO_MANY_OUTPUTS,
    UIGETDIR_ERROR_HOST,
    UIGETDIR_ERROR_INVALID_SELECTION,
];

pub const UIGETDIR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &UIGETDIR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &UIGETDIR_ERRORS,
};
