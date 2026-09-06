use crate::*;

const OLD_PATH: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "oldpath",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Previous search path as a character row vector.",
};
const OUTPUTS: [BuiltinParamDescriptor; 1] = [OLD_PATH];
const FOLDER: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "folder",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "One or more folders or text containers, followed by optional position flags.",
};
const INPUTS: [BuiltinParamDescriptor; 1] = [FOLDER];
const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "oldpath = addpath(folder1, ..., folderN)",
        inputs: &INPUTS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "oldpath = addpath(folder1, ..., folderN, position)",
        inputs: &INPUTS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "oldpath = addpath(folder1, ..., folderN, options)",
        inputs: &INPUTS,
        outputs: &OUTPUTS,
    },
];

macro_rules! error {
    ($name:ident, $code:literal, $identifier:literal, $when:literal, $message:literal) => {
        pub const $name: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: $code,
            identifier: Some($identifier),
            when: $when,
            message: $message,
        };
    };
}

error!(ADDPATH_ERROR_ARGUMENT_TYPE, "RM.ADDPATH.ARG_TYPE", "RunMat:addpath:InvalidFolder", "A folder argument is not admitted text, a text container, or a mode-admitted numeric character-code row.", "addpath: folder names must be character vectors, strings, string arrays, or cell arrays of text");
error!(
    ADDPATH_ERROR_TOO_FEW_ARGUMENTS,
    "RM.ADDPATH.TOO_FEW_ARGS",
    "RunMat:addpath:NotEnoughInputs",
    "No nonempty folder token is provided.",
    "addpath: at least one folder must be specified"
);
error!(
    ADDPATH_ERROR_POSITION,
    "RM.ADDPATH.POSITION_REPEATED",
    "RunMat:addpath:InvalidPosition",
    "More than one position option is provided.",
    "addpath: position option must be '-begin' or '-end' and may only appear once"
);
error!(
    ADDPATH_ERROR_PATHDEF,
    "RM.ADDPATH.PATHDEF_UNSUPPORTED",
    "RunMat:addpath:PathdefUnsupported",
    "The pathdef or pathdef.m token is provided.",
    "addpath: loading pathdef.m is not implemented yet"
);
error!(
    ADDPATH_ERROR_CURRENT_FOLDER,
    "RM.ADDPATH.CWD_RESOLVE",
    "RunMat:addpath:CurrentFolderUnavailable",
    "The current folder cannot be resolved while normalizing a relative folder.",
    "addpath: unable to resolve current directory"
);
error!(
    ADDPATH_ERROR_FOLDER_NOT_FOUND,
    "RM.ADDPATH.FOLDER_NOT_FOUND",
    "RunMat:addpath:FolderNotFound",
    "A requested folder does not exist.",
    "addpath: folder not found"
);
error!(
    ADDPATH_ERROR_NOT_FOLDER,
    "RM.ADDPATH.NOT_FOLDER",
    "RunMat:addpath:NotFolder",
    "A requested path exists but is not a folder.",
    "addpath: path is not a folder"
);
error!(
    ADDPATH_ERROR_PROVIDER,
    "RM.ADDPATH.PROVIDER_FAILED",
    "RunMat:addpath:ProviderFailed",
    "An admitted resident numeric character-code row cannot be gathered from its owner.",
    "addpath: unable to read resident character codes"
);

const ERRORS: [BuiltinErrorDescriptor; 8] = [
    ADDPATH_ERROR_ARGUMENT_TYPE,
    ADDPATH_ERROR_TOO_FEW_ARGUMENTS,
    ADDPATH_ERROR_POSITION,
    ADDPATH_ERROR_PATHDEF,
    ADDPATH_ERROR_CURRENT_FOLDER,
    ADDPATH_ERROR_FOLDER_NOT_FOUND,
    ADDPATH_ERROR_NOT_FOLDER,
    ADDPATH_ERROR_PROVIDER,
];

pub const ADDPATH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const ADDPATH_NUMERIC_CHARACTER_CODES_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "addpath-numeric-character-codes",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "addpath with numeric character-code rows is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:AddpathNumericCharacterCodesExtension"),
    };
pub(super) const EXTENSIONS: &[BuiltinExtensionDescriptor] =
    &[ADDPATH_NUMERIC_CHARACTER_CODES_EXTENSION];
const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "folder",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "A dense real integer row may encode one folder name in RunMat mode.",
}];
pub(super) const INTEGER_CAPABILITIES: &[BuiltinIntegerCapabilityDescriptor] =
    &[BuiltinIntegerCapabilityDescriptor {
        form: "oldpath = addpath(integer_character_codes, options)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes:
            "Code points decode exactly, invalid Unicode rejects, and mutation remains host-owned.",
    }];
