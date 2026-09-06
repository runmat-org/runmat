use crate::*;

const FOLDER: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "folder",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Folder whose source and data artifacts are summarized.",
};
const INFO: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "info",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar structure grouping source, data, extension, class, and package names.",
};
const SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "info = what()",
        inputs: &[],
        outputs: &[INFO],
    },
    BuiltinSignatureDescriptor {
        label: "info = what(folder)",
        inputs: &[FOLDER],
        outputs: &[INFO],
    },
];

pub const WHAT_ERROR_ARITY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WHAT.ARITY",
    identifier: Some("RunMat:what:InvalidArity"),
    when: "More than one folder input is supplied.",
    message: "what: too many input arguments",
};
pub const WHAT_ERROR_FOLDER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WHAT.FOLDER",
    identifier: Some("RunMat:what:InvalidFolder"),
    when: "The folder input is not a character row or string scalar.",
    message: "what: folder must be a character vector or string scalar",
};
pub const WHAT_ERROR_FILESYSTEM: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WHAT.FILESYSTEM",
    identifier: Some("RunMat:what:FilesystemError"),
    when: "The selected folder cannot be enumerated.",
    message: "what: unable to inspect folder",
};

pub const WHAT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[WHAT_ERROR_ARITY, WHAT_ERROR_FOLDER, WHAT_ERROR_FILESYSTEM],
};
pub const WHAT_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "The optional folder is host text. Numeric and provider-resident values reject before filesystem or accelerator-provider access.",
};
