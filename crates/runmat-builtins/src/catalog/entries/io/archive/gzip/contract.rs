use crate::*;
pub const GZIP_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None,
    notes: "Text inputs and controls reject numeric host/resident values before filesystem, network, or accelerator access.",
};
pub const GZIP_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GZIP.OUTPUTS",
    identifier: Some("RunMat:gzip:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "gzip: too many output arguments",
};
pub const GZIP_ERROR_RUNTIME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GZIP.RUNTIME",
    identifier: Some("RunMat:gzip:RuntimeError"),
    when: "Input admission, filesystem, codec, or transport execution fails.",
    message: "gzip: operation failed",
};

pub const GZIP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "value = gzip(filename, outputfolder)",
        inputs: &[
            BuiltinParamDescriptor {
                name: "filename",
                ty: BuiltinParamType::Any,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "File paths, folder, or wildcard pattern.",
            },
            BuiltinParamDescriptor {
                name: "outputfolder",
                ty: BuiltinParamType::StringScalar,
                arity: BuiltinParamArity::Optional,
                default: None,
                description: "Destination folder.",
            },
        ],
        outputs: &[BuiltinParamDescriptor {
            name: "value",
            ty: BuiltinParamType::Any,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "Column cell array of character paths.",
        }],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[GZIP_ERROR_OUTPUTS, GZIP_ERROR_RUNTIME],
};
