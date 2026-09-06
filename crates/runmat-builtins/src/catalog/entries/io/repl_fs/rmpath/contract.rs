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
    description: "One or more folders or text containers to remove.",
};
const INPUTS: [BuiltinParamDescriptor; 1] = [FOLDER];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "oldpath = rmpath(folder1, ..., folderN)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

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

error!(RMPATH_ERROR_ARGUMENT_TYPE, "RM.RMPATH.ARG_TYPE", "RunMat:rmpath:InvalidFolder", "A folder argument is not character, string, or cell text.", "rmpath: folder names must be character vectors, strings, string arrays, or cell arrays of text");
error!(
    RMPATH_ERROR_TOO_FEW_ARGUMENTS,
    "RM.RMPATH.TOO_FEW_ARGS",
    "RunMat:rmpath:NotEnoughInputs",
    "No nonempty folder token is provided.",
    "rmpath: at least one folder must be specified"
);
error!(
    RMPATH_ERROR_CURRENT_FOLDER,
    "RM.RMPATH.CWD_RESOLVE",
    "RunMat:rmpath:CurrentFolderUnavailable",
    "The current folder cannot be resolved while normalizing a relative folder.",
    "rmpath: unable to resolve current directory"
);
error!(
    RMPATH_ERROR_NOT_FOLDER,
    "RM.RMPATH.NOT_FOLDER",
    "RunMat:rmpath:NotFolder",
    "A requested path exists but is not a folder.",
    "rmpath: path is not a folder"
);
error!(
    RMPATH_ERROR_NOT_ON_PATH,
    "RM.RMPATH.NOT_ON_PATH",
    "RunMat:rmpath:NotOnPath",
    "A requested folder exists but is not on the active path.",
    "rmpath: folder not on search path"
);
error!(
    RMPATH_ERROR_FOLDER_NOT_FOUND,
    "RM.RMPATH.FOLDER_NOT_FOUND",
    "RunMat:rmpath:FolderNotFound",
    "A requested folder does not exist.",
    "rmpath: folder not found"
);

const ERRORS: [BuiltinErrorDescriptor; 6] = [
    RMPATH_ERROR_ARGUMENT_TYPE,
    RMPATH_ERROR_TOO_FEW_ARGUMENTS,
    RMPATH_ERROR_CURRENT_FOLDER,
    RMPATH_ERROR_NOT_FOLDER,
    RMPATH_ERROR_NOT_ON_PATH,
    RMPATH_ERROR_FOLDER_NOT_FOUND,
];
pub const RMPATH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub(super) const INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "rmpath accepts text containers only; numeric host and resident values reject before provider lookup.",
};
