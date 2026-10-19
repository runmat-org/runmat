use crate::*;
pub const GUNZIP_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None,
    notes: "Text inputs and controls reject numeric host/resident values before filesystem, network, or accelerator access.",
};
pub const GUNZIP_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GUNZIP.OUTPUTS",
    identifier: Some("RunMat:gunzip:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "gunzip: too many output arguments",
};
pub const GUNZIP_ERROR_RUNTIME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GUNZIP.RUNTIME",
    identifier: Some("RunMat:gunzip:RuntimeError"),
    when: "Input admission, filesystem, codec, or transport execution fails.",
    message: "gunzip: operation failed",
};

pub const GUNZIP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "value = gunzip(filename, outputfolder)",
        inputs: &[
            BuiltinParamDescriptor {
                name: "filename",
                ty: BuiltinParamType::Any,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Gzip paths or HTTP/HTTPS URLs.",
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
    errors: &[GUNZIP_ERROR_OUTPUTS, GUNZIP_ERROR_RUNTIME],
};
