use crate::*;

const FOLDER: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "folderName",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Folder path to create, including missing intermediate folders.",
};
const PARENT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "parentFolder",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Parent path to create when necessary.",
};
const CHILD: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "folderName",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Relative child path beneath parentFolder.",
};
const STATUS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical true when the folder exists after the operation.",
};
const MESSAGE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "msg",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Character-row diagnostic, empty after a newly created folder.",
};
const MESSAGE_ID: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "msgID",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Character-row diagnostic identifier.",
};
const OUTPUTS: &[BuiltinParamDescriptor] = &[STATUS, MESSAGE, MESSAGE_ID];
const SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "[status, msg, msgID] = mkdir(folderName)",
        inputs: &[FOLDER],
        outputs: OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[status, msg, msgID] = mkdir(parentFolder, folderName)",
        inputs: &[PARENT, CHILD],
        outputs: OUTPUTS,
    },
];

macro_rules! error {
    ($name:ident, $code:literal, $id:literal, $when:literal, $message:literal) => {
        pub const $name: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: $code,
            identifier: Some($id),
            when: $when,
            message: $message,
        };
    };
}

error!(
    MKDIR_ERROR_ARITY,
    "RM.MKDIR.ARITY",
    "RunMat:mkdir:InvalidArgumentCount",
    "The call has zero or more than two inputs.",
    "mkdir: expected one folder or a parent and relative child"
);
error!(
    MKDIR_ERROR_FOLDER_TYPE,
    "RM.MKDIR.FOLDER_TYPE",
    "RunMat:mkdir:InvalidFolder",
    "A folder input is not a character row or string scalar.",
    "mkdir: folder names must be character vectors or string scalars"
);
error!(
    MKDIR_ERROR_EMPTY_NAME,
    "RM.MKDIR.EMPTY_NAME",
    "RunMat:mkdir:InvalidFolderName",
    "A required folder name is empty.",
    "mkdir: folder name must not be empty"
);
error!(
    MKDIR_ERROR_ABSOLUTE_CHILD,
    "RM.MKDIR.ABSOLUTE_CHILD",
    "RunMat:mkdir:FolderMustBeRelative",
    "The child in the two-input form is rooted.",
    "mkdir: folder name must be relative when a parent folder is supplied"
);
error!(
    MKDIR_ERROR_NOT_DIRECTORY,
    "RM.MKDIR.NOT_DIRECTORY",
    "RunMat:mkdir:TargetNotDirectory",
    "The target or an intermediate component is not a directory.",
    "mkdir: a path component is not a directory"
);
error!(
    MKDIR_ERROR_FILESYSTEM,
    "RM.MKDIR.FILESYSTEM",
    "RunMat:mkdir:OSError",
    "The filesystem cannot create the folder.",
    "mkdir: unable to create folder"
);
pub const MKDIR_NOTICE_EXISTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MKDIR.DIRECTORY_EXISTS",
    identifier: Some("RunMat:mkdir:DirectoryExists"),
    when: "The target already exists as a directory.",
    message: "Directory already exists.",
};

pub const MKDIR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        MKDIR_ERROR_ARITY,
        MKDIR_ERROR_FOLDER_TYPE,
        MKDIR_ERROR_EMPTY_NAME,
        MKDIR_ERROR_ABSOLUTE_CHILD,
        MKDIR_ERROR_NOT_DIRECTORY,
        MKDIR_ERROR_FILESYSTEM,
        MKDIR_NOTICE_EXISTS,
    ],
};

pub const MKDIR_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes:
        "Folder arguments are text; numeric and resident values reject before filesystem access.",
};
