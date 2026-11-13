use crate::*;

const LISTING: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "listing",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Column struct array with name, folder, date, bytes, isdir, and datenum fields.",
};
const NAME: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "File, folder, or wildcard path as a character vector or string scalar.",
};
const PATTERN: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "pattern",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "RunMat-only wildcard pattern evaluated within the first folder input.",
};
const SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "listing = dir()",
        inputs: &[],
        outputs: &[LISTING],
    },
    BuiltinSignatureDescriptor {
        label: "listing = dir(name)",
        inputs: &[NAME],
        outputs: &[LISTING],
    },
    BuiltinSignatureDescriptor {
        label: "listing = dir(folder, pattern)",
        inputs: &[NAME, PATTERN],
        outputs: &[LISTING],
    },
];

pub const DIR_ERROR_ARITY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DIR.ARITY",
    identifier: Some("RunMat:dir:InvalidArity"),
    when: "More than two inputs are supplied.",
    message: "dir: too many input arguments",
};
pub const DIR_ERROR_NAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DIR.NAME",
    identifier: Some("RunMat:dir:InvalidName"),
    when: "The name input is not a character vector or string scalar.",
    message: "dir: name must be a character vector or string scalar",
};
pub const DIR_ERROR_FOLDER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DIR.FOLDER",
    identifier: Some("RunMat:dir:InvalidFolder"),
    when: "The first input in the two-input form is not a character vector or string scalar.",
    message: "dir: folder must be a character vector or string scalar",
};
pub const DIR_ERROR_PATTERN: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DIR.PATTERN",
    identifier: Some("RunMat:dir:InvalidPattern"),
    when: "The second input is not a character vector or string scalar.",
    message: "dir: pattern must be a character vector or string scalar",
};
pub const DIR_ERROR_FOLDER_WILDCARD: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DIR.FOLDER_WILDCARD",
    identifier: Some("RunMat:dir:FolderWildcard"),
    when: "The two-input form uses wildcard characters in its folder input.",
    message: "dir: folder input must not contain wildcard characters",
};
pub const DIR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        DIR_ERROR_ARITY,
        DIR_ERROR_NAME,
        DIR_ERROR_FOLDER,
        DIR_ERROR_PATTERN,
        DIR_ERROR_FOLDER_WILDCARD,
    ],
};
pub const DIR_FOLDER_PATTERN_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "dir-folder-pattern",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "dir(folder, pattern) is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DirFolderPatternExtension"),
};
pub const DIR_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor { kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None, notes: "Directory names and patterns are host text. Numeric and provider-resident values reject before filesystem or provider access." };
