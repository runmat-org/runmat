use crate::*;
pub const WRITELINES_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None,
    notes: "Text inputs and controls reject numeric host/resident values before filesystem, network, or accelerator access.",
};
pub const WRITELINES_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WRITELINES.OUTPUTS",
    identifier: Some("RunMat:writelines:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "writelines: too many output arguments",
};
pub const WRITELINES_ERROR_RUNTIME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WRITELINES.RUNTIME",
    identifier: Some("RunMat:writelines:RuntimeError"),
    when: "Input admission, filesystem, codec, or transport execution fails.",
    message: "writelines: operation failed",
};
pub const WRITELINES_BYTE_COUNT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "writelines-byte-count",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "Requesting writelines' byte count is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:WritelinesByteCountExtension"),
    };
pub const WRITELINES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "writelines(lines, filename, options)",
        inputs: &[
            BuiltinParamDescriptor {
                name: "lines",
                ty: BuiltinParamType::Any,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Text values to write.",
            },
            BuiltinParamDescriptor {
                name: "filename",
                ty: BuiltinParamType::StringScalar,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Destination text file.",
            },
            BuiltinParamDescriptor {
                name: "options",
                ty: BuiltinParamType::Any,
                arity: BuiltinParamArity::Variadic,
                default: None,
                description: "LineEnding name-value pairs.",
            },
        ],
        outputs: &[BuiltinParamDescriptor {
            name: "bytes",
            ty: BuiltinParamType::Any,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "Written UTF-8 byte count, only when the RunMat extension is enabled.",
        }],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[WRITELINES_ERROR_OUTPUTS, WRITELINES_ERROR_RUNTIME],
};
