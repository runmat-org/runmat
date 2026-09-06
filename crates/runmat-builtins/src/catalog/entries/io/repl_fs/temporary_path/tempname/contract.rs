use crate::*;

const FOLDER: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "folder",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Base folder as a character row or string scalar.",
};
const OUTPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Temporary path that did not exist when selected.",
};
const SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "filename = tempname()",
        inputs: &[],
        outputs: &[OUTPUT],
    },
    BuiltinSignatureDescriptor {
        label: "filename = tempname(folder)",
        inputs: &[FOLDER],
        outputs: &[OUTPUT],
    },
];

pub const TEMPNAME_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEMPNAME.TOO_MANY_INPUTS",
    identifier: Some("RunMat:tempname:TooManyInputs"),
    when: "More than one input is supplied.",
    message: "tempname: too many input arguments",
};
pub const TEMPNAME_ERROR_FOLDER_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEMPNAME.FOLDER_TYPE",
    identifier: Some("RunMat:tempname:InvalidFolder"),
    when: "The folder is not a character row or string scalar.",
    message: "tempname: folder name must be a character vector or string scalar",
};
pub const TEMPNAME_ERROR_FOLDER_EMPTY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEMPNAME.FOLDER_EMPTY",
    identifier: Some("RunMat:tempname:EmptyFolder"),
    when: "The folder is empty.",
    message: "tempname: folder name must not be empty",
};
pub const TEMPNAME_ERROR_FOLDER_RESOLVE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEMPNAME.FOLDER_RESOLVE",
    identifier: Some("RunMat:tempname:FolderResolve"),
    when: "Home-directory expansion fails.",
    message: "tempname: unable to resolve folder path",
};
pub const TEMPNAME_ERROR_TEMP_DIR_UNAVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEMPNAME.TEMP_DIR_UNAVAILABLE",
    identifier: Some("RunMat:tempname:TempDirectoryUnavailable"),
    when: "The session cannot determine a temporary directory.",
    message: "tempname: unable to determine temporary directory",
};
pub const TEMPNAME_ERROR_UNABLE_TO_GENERATE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEMPNAME.UNABLE_TO_GENERATE",
    identifier: Some("RunMat:tempname:UnableToGenerate"),
    when: "No unused candidate is found within the bounded retry budget.",
    message: "tempname: unable to generate a unique name",
};

pub const TEMPNAME_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        TEMPNAME_ERROR_TOO_MANY_INPUTS,
        TEMPNAME_ERROR_FOLDER_TYPE,
        TEMPNAME_ERROR_FOLDER_EMPTY,
        TEMPNAME_ERROR_FOLDER_RESOLVE,
        TEMPNAME_ERROR_TEMP_DIR_UNAVAILABLE,
        TEMPNAME_ERROR_UNABLE_TO_GENERATE,
    ],
};

pub const TEMPNAME_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes:
        "The optional folder is text; numeric and resident values reject before filesystem access.",
};
