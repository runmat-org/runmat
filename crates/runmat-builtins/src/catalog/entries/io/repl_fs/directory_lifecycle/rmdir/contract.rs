use crate::*;

const FOLDER: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "folderName",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Folder path to remove.",
};
const RECURSIVE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "s",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: Some("'s'"),
    description: "Recursive-removal flag.",
};
const OPTION_NAME: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "ResolveSymbolicLinks",
    ty: BuiltinParamType::PropertyName,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Symbolic-link resolution option name.",
};
const OPTION_VALUE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: Some("false"),
    description: "Logical or exact numeric scalar zero or one.",
};
const STATUS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical true when removal succeeds.",
};
const MESSAGE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "msg",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Character-row diagnostic, empty after success.",
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
        label: "[status, msg, msgID] = rmdir(folderName)",
        inputs: &[FOLDER],
        outputs: OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[status, msg, msgID] = rmdir(folderName, 's')",
        inputs: &[FOLDER, RECURSIVE],
        outputs: OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[status, msg, msgID] = rmdir(___, ResolveSymbolicLinks=tf)",
        inputs: &[FOLDER, OPTION_NAME, OPTION_VALUE],
        outputs: OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[status, msg, msgID] = rmdir(folderName, 's', ResolveSymbolicLinks=tf)",
        inputs: &[FOLDER, RECURSIVE, OPTION_NAME, OPTION_VALUE],
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
    RMDIR_ERROR_ARITY,
    "RM.RMDIR.ARITY",
    "RunMat:rmdir:InvalidArgumentCount",
    "The call has no folder or has unsupported trailing arguments.",
    "rmdir: invalid input arguments"
);
error!(
    RMDIR_ERROR_FOLDER_TYPE,
    "RM.RMDIR.FOLDER_TYPE",
    "RunMat:rmdir:InvalidFolder",
    "The folder is not a character row or string scalar.",
    "rmdir: folder name must be a character vector or string scalar"
);
error!(
    RMDIR_ERROR_OPTION,
    "RM.RMDIR.OPTION",
    "RunMat:rmdir:InvalidOption",
    "The recursive flag or symbolic-link option is malformed.",
    "rmdir: invalid recursive or symbolic-link option"
);
error!(
    RMDIR_ERROR_EMPTY_NAME,
    "RM.RMDIR.EMPTY_NAME",
    "RunMat:rmdir:InvalidFolderName",
    "The folder name is empty.",
    "rmdir: folder name must not be empty"
);
error!(
    RMDIR_ERROR_NOT_FOUND,
    "RM.RMDIR.NOT_FOUND",
    "RunMat:rmdir:DirectoryNotFound",
    "The target does not exist.",
    "rmdir: directory not found"
);
error!(
    RMDIR_ERROR_NOT_DIRECTORY,
    "RM.RMDIR.NOT_DIRECTORY",
    "RunMat:rmdir:NotADirectory",
    "The target exists but is not a directory or directory link.",
    "rmdir: target is not a directory"
);
error!(
    RMDIR_ERROR_NOT_EMPTY,
    "RM.RMDIR.NOT_EMPTY",
    "RunMat:rmdir:DirectoryNotEmpty",
    "A nonrecursive removal targets a nonempty directory.",
    "rmdir: directory is not empty"
);
error!(
    RMDIR_ERROR_FILESYSTEM,
    "RM.RMDIR.FILESYSTEM",
    "RunMat:rmdir:OSError",
    "The filesystem cannot remove the target.",
    "rmdir: unable to remove folder"
);

pub const RMDIR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        RMDIR_ERROR_ARITY,
        RMDIR_ERROR_FOLDER_TYPE,
        RMDIR_ERROR_OPTION,
        RMDIR_ERROR_EMPTY_NAME,
        RMDIR_ERROR_NOT_FOUND,
        RMDIR_ERROR_NOT_DIRECTORY,
        RMDIR_ERROR_NOT_EMPTY,
        RMDIR_ERROR_FILESYSTEM,
    ],
};

const INTEGER_CONTROL: &[BuiltinIntegerInputCapability] = &[BuiltinIntegerInputCapability {
    name: "ResolveSymbolicLinks",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "All integer classes are decoded exactly and must contain scalar zero or one.",
}];
pub const RMDIR_INTEGER_CAPABILITIES: &[BuiltinIntegerCapabilityDescriptor] =
    &[BuiltinIntegerCapabilityDescriptor {
        form: "rmdir(___, ResolveSymbolicLinks=integer_tf)",
        inputs: INTEGER_CONTROL,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The option is consumed as an exact control; status remains logical.",
    }];
