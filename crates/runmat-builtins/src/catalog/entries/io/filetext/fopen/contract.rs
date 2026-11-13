use crate::*;

pub const FOPEN_INTEGER_ID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fopen-integer-fileid",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "integer-class fopen query identifiers are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FopenIntegerIdExtension"),
};
pub const FOPEN_SINGLE_ID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fopen-single-fileid",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "single-precision fopen query identifiers are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FopenSingleIdExtension"),
};
pub const FOPEN_RESIDENT_ID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fopen-resident-fileid",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "provider-resident fopen query identifiers are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FopenResidentIdExtension"),
};
pub const FOPEN_LEGACY_ALL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fopen-legacy-all",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "legacy fopen no-argument and 'all' listing forms are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FopenLegacyAllExtension"),
};
pub const FOPEN_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    FOPEN_INTEGER_ID_EXTENSION,
    FOPEN_SINGLE_ID_EXTENSION,
    FOPEN_RESIDENT_ID_EXTENSION,
    FOPEN_LEGACY_ALL_EXTENSION,
];

pub const FOPEN_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "fileID",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target documents double query identifiers; typed integer identifiers are checked exactly as one gated RunMat extension.",
    }];
pub const FOPEN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "filename = fopen(integer_fileID)",
        inputs: &FOPEN_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "A successful query returns host character metadata; integer identifiers are validated exactly.",
    }];

pub const FOPEN_OUTPUT_FID: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fid",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "File identifier on success, or -1 on failure.",
}];
pub const FOPEN_OUTPUT_FID_MSG: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier on success, or -1 on failure.",
    },
    BuiltinParamDescriptor {
        name: "msg",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"\""),
        description: "Open failure message string (empty on success).",
    },
];
pub const FOPEN_OUTPUT_FID_MSG_MACHINEFMT_ENCODING: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "fid",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "File identifier on success, or -1 on failure.",
    },
    BuiltinParamDescriptor {
        name: "msg",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"\""),
        description: "Open failure message string (empty on success).",
    },
    BuiltinParamDescriptor {
        name: "machinefmt",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"native\""),
        description: "Resolved machine-format label for the opened stream.",
    },
    BuiltinParamDescriptor {
        name: "encoding",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"UTF-8\""),
        description: "Resolved encoding label for the opened stream.",
    },
];
pub const FOPEN_OUTPUT_FILENAME: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Filename associated with queried file identifier.",
}];
pub const FOPEN_OUTPUT_FILENAME_PERMISSION_MACHINEFMT_ENCODING: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Filename associated with queried file identifier.",
    },
    BuiltinParamDescriptor {
        name: "permission",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"\""),
        description: "Canonical fopen permission string.",
    },
    BuiltinParamDescriptor {
        name: "machinefmt",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"\""),
        description: "Machine-format label of queried stream.",
    },
    BuiltinParamDescriptor {
        name: "encoding",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"\""),
        description: "Encoding label of queried stream.",
    },
];
pub const FOPEN_OUTPUT_HANDLES: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fids",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Column vector of open user file identifiers.",
}];
pub const FOPEN_OUTPUT_HANDLES_NAMES_MACHINEFMTS_ENCODINGS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "fids",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Column vector of open user file identifiers.",
    },
    BuiltinParamDescriptor {
        name: "names",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("{}"),
        description: "Cell column of display names.",
    },
    BuiltinParamDescriptor {
        name: "machinefmts",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("{}"),
        description: "Cell column of machine-format labels.",
    },
    BuiltinParamDescriptor {
        name: "encodings",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("{}"),
        description: "Cell column of encoding labels.",
    },
];

pub const FOPEN_INPUTS_OPEN_FILE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Path to file to open.",
}];
pub const FOPEN_INPUTS_OPEN_FILE_PERMISSION: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Path to file to open.",
    },
    BuiltinParamDescriptor {
        name: "permission",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"r\""),
        description: "Permission string such as 'r', 'w', 'a', optionally with '+'/'b'/'t'.",
    },
];
pub const FOPEN_INPUTS_OPEN_FILE_PERMISSION_MACHINEFMT: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Path to file to open.",
    },
    BuiltinParamDescriptor {
        name: "permission",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"r\""),
        description: "Permission string such as 'r', 'w', 'a', optionally with '+'/'b'/'t'.",
    },
    BuiltinParamDescriptor {
        name: "machinefmt",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"native\""),
        description: "Machine-format label.",
    },
];
pub const FOPEN_INPUTS_OPEN_FILE_PERMISSION_MACHINEFMT_ENCODING: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Path to file to open.",
    },
    BuiltinParamDescriptor {
        name: "permission",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"r\""),
        description: "Permission string such as 'r', 'w', 'a', optionally with '+'/'b'/'t'.",
    },
    BuiltinParamDescriptor {
        name: "machinefmt",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"native\""),
        description: "Machine-format label.",
    },
    BuiltinParamDescriptor {
        name: "encoding",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"UTF-8\""),
        description: "Encoding label.",
    },
];
pub const FOPEN_INPUTS_QUERY_FID: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fid",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric file identifier.",
}];
pub const FOPEN_INPUTS_LIST_ALL: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "selector",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: Some("\"all\""),
    description: "Literal selector 'all'.",
}];
pub const FOPEN_INPUTS_LIST_ALL_MACHINEFMT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "selector",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"all\""),
        description: "Literal selector 'all'.",
    },
    BuiltinParamDescriptor {
        name: "machinefmt",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Machine-format filter for listed handles.",
    },
];

pub const FOPEN_SIGNATURES: [BuiltinSignatureDescriptor; 12] = [
    BuiltinSignatureDescriptor {
        label: "fid = fopen(filename)",
        inputs: &FOPEN_INPUTS_OPEN_FILE,
        outputs: &FOPEN_OUTPUT_FID,
    },
    BuiltinSignatureDescriptor {
        label: "fid = fopen(filename, permission)",
        inputs: &FOPEN_INPUTS_OPEN_FILE_PERMISSION,
        outputs: &FOPEN_OUTPUT_FID,
    },
    BuiltinSignatureDescriptor {
        label: "fid = fopen(filename, permission, machinefmt)",
        inputs: &FOPEN_INPUTS_OPEN_FILE_PERMISSION_MACHINEFMT,
        outputs: &FOPEN_OUTPUT_FID,
    },
    BuiltinSignatureDescriptor {
        label: "fid = fopen(filename, permission, machinefmt, encoding)",
        inputs: &FOPEN_INPUTS_OPEN_FILE_PERMISSION_MACHINEFMT_ENCODING,
        outputs: &FOPEN_OUTPUT_FID,
    },
    BuiltinSignatureDescriptor {
        label: "[fid, msg] = fopen(filename, ...)",
        inputs: &FOPEN_INPUTS_OPEN_FILE_PERMISSION_MACHINEFMT_ENCODING,
        outputs: &FOPEN_OUTPUT_FID_MSG,
    },
    BuiltinSignatureDescriptor {
        label: "[fid, msg, machinefmt, encoding] = fopen(filename, ...)",
        inputs: &FOPEN_INPUTS_OPEN_FILE_PERMISSION_MACHINEFMT_ENCODING,
        outputs: &FOPEN_OUTPUT_FID_MSG_MACHINEFMT_ENCODING,
    },
    BuiltinSignatureDescriptor {
        label: "filename = fopen(fid)",
        inputs: &FOPEN_INPUTS_QUERY_FID,
        outputs: &FOPEN_OUTPUT_FILENAME,
    },
    BuiltinSignatureDescriptor {
        label: "[filename, permission, machinefmt, encoding] = fopen(fid)",
        inputs: &FOPEN_INPUTS_QUERY_FID,
        outputs: &FOPEN_OUTPUT_FILENAME_PERMISSION_MACHINEFMT_ENCODING,
    },
    BuiltinSignatureDescriptor {
        label: "fids = fopen(\"all\")",
        inputs: &FOPEN_INPUTS_LIST_ALL,
        outputs: &FOPEN_OUTPUT_HANDLES,
    },
    BuiltinSignatureDescriptor {
        label: "fids = fopen(\"all\", machinefmt)",
        inputs: &FOPEN_INPUTS_LIST_ALL_MACHINEFMT,
        outputs: &FOPEN_OUTPUT_HANDLES,
    },
    BuiltinSignatureDescriptor {
        label: "[fids, names, machinefmts, encodings] = fopen(\"all\")",
        inputs: &FOPEN_INPUTS_LIST_ALL,
        outputs: &FOPEN_OUTPUT_HANDLES_NAMES_MACHINEFMTS_ENCODINGS,
    },
    BuiltinSignatureDescriptor {
        label: "[fids, names, machinefmts, encodings] = fopen(\"all\", machinefmt)",
        inputs: &FOPEN_INPUTS_LIST_ALL_MACHINEFMT,
        outputs: &FOPEN_OUTPUT_HANDLES_NAMES_MACHINEFMTS_ENCODINGS,
    },
];

pub const FOPEN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FOPEN.INVALID_INPUT",
    identifier: Some("RunMat:fopen:InvalidInput"),
    when: "Argument count or argument shape/type is invalid.",
    message: "fopen: invalid input arguments",
};
pub const FOPEN_ERROR_INVALID_PERMISSION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FOPEN.INVALID_PERMISSION",
    identifier: Some("RunMat:fopen:InvalidPermission"),
    when: "Permission string is invalid or unsupported.",
    message: "fopen: invalid permission string",
};
pub const FOPEN_ERROR_INVALID_MACHINEFMT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FOPEN.INVALID_MACHINEFMT",
    identifier: Some("RunMat:fopen:InvalidMachineFormat"),
    when: "Machine-format argument is invalid or unsupported.",
    message: "fopen: invalid machine format",
};
pub const FOPEN_ERROR_INVALID_ENCODING: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FOPEN.INVALID_ENCODING",
    identifier: Some("RunMat:fopen:InvalidEncoding"),
    when: "Encoding argument is invalid.",
    message: "fopen: invalid encoding",
};
pub const FOPEN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FOPEN.INTERNAL",
    identifier: None,
    when: "Internal runtime conversion/control-flow operation fails.",
    message: "fopen: internal error",
};
pub const FOPEN_ERRORS: [BuiltinErrorDescriptor; 5] = [
    FOPEN_ERROR_INVALID_INPUT,
    FOPEN_ERROR_INVALID_PERMISSION,
    FOPEN_ERROR_INVALID_MACHINEFMT,
    FOPEN_ERROR_INVALID_ENCODING,
    FOPEN_ERROR_INTERNAL,
];
pub const FOPEN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FOPEN_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FOPEN_ERRORS,
};
