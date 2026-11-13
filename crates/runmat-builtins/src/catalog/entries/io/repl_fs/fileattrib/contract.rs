use crate::*;

const PATH: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "path",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "File or folder path as a host text scalar.",
};
const FLAG: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "flag",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "+w or -w; read-permission mutation needs provider support.",
};
const STATUS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "1 when metadata was read; 0 when the path cannot be read.",
};
const ATTRIBUTES: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "attributes",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Attribute fields; unavailable permission facts are NaN.",
};
const SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "[status, attributes] = fileattrib(path)",
        inputs: &[PATH],
        outputs: &[STATUS, ATTRIBUTES],
    },
    BuiltinSignatureDescriptor {
        label: "[status, attributes] = fileattrib(path, flag)",
        inputs: &[PATH, FLAG],
        outputs: &[STATUS, ATTRIBUTES],
    },
    BuiltinSignatureDescriptor {
        label: "[status, attributes] = fileattrib(path, flag1, flag2)",
        inputs: &[PATH, FLAG, FLAG],
        outputs: &[STATUS, ATTRIBUTES],
    },
];
pub const FILEATTRIB_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        FILEATTRIB_ERROR_TOO_MANY_OUTPUTS,
        FILEATTRIB_ERROR_INVALID_CONTROL,
        FILEATTRIB_ERROR_UNSUPPORTED_PERMISSION,
        FILEATTRIB_ERROR_PROVIDER,
    ],
};
pub const FILEATTRIB_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILEATTRIB.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:fileattrib:TooManyOutputs"),
    when: "More than two outputs are requested.",
    message: "fileattrib: too many output arguments",
};
pub const FILEATTRIB_ERROR_INVALID_CONTROL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILEATTRIB.INVALID_CONTROL",
    identifier: None,
    when: "The path or an attribute flag has an unsupported host type, geometry, or spelling.",
    message: "fileattrib: invalid path or attribute flag",
};
pub const FILEATTRIB_ERROR_UNSUPPORTED_PERMISSION: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.FILEATTRIB.UNSUPPORTED_PERMISSION",
        identifier: Some("RunMat:fileattrib:UnsupportedPermission"),
        when: "A read-permission mutation is requested without provider permission support.",
        message: "fileattrib: read permission changes require provider permission support",
    };
pub const FILEATTRIB_ERROR_PROVIDER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILEATTRIB.PROVIDER",
    identifier: None,
    when: "The selected filesystem owner cannot resolve, mutate, or authenticate the path.",
    message: "fileattrib: filesystem operation failed",
};
pub const FILEATTRIB_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "Paths and flags are host text. Numeric and resident numeric controls reject before filesystem access; numeric status is output only.",
    };
