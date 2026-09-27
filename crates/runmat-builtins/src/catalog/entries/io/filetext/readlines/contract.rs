use crate::*;
pub const READLINES_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None,
    notes: "Text inputs and controls reject numeric host/resident values before filesystem, network, or accelerator access.",
};
pub const READLINES_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READLINES.OUTPUTS",
    identifier: Some("RunMat:readlines:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "readlines: too many output arguments",
};
pub const READLINES_ERROR_RUNTIME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READLINES.RUNTIME",
    identifier: Some("RunMat:readlines:RuntimeError"),
    when: "Input admission, filesystem, codec, or transport execution fails.",
    message: "readlines: operation failed",
};

pub const READLINES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "value = readlines(filename, options)",
        inputs: &[
            BuiltinParamDescriptor {
                name: "filename",
                ty: BuiltinParamType::StringScalar,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "UTF-8 text file.",
            },
            BuiltinParamDescriptor {
                name: "options",
                ty: BuiltinParamType::Any,
                arity: BuiltinParamArity::Variadic,
                default: None,
                description: "EmptyLineRule name-value pair.",
            },
        ],
        outputs: &[BuiltinParamDescriptor {
            name: "value",
            ty: BuiltinParamType::Any,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "Column string array of file lines.",
        }],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[READLINES_ERROR_OUTPUTS, READLINES_ERROR_RUNTIME],
};
