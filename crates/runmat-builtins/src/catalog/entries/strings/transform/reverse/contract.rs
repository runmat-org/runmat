use crate::*;
pub const REVERSE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "reverse accepts string arrays, character vectors, or cell arrays of character vectors. Numeric and integer inputs reject without implicit text conversion or provider access.",
};
pub const REVERSE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "out = reverse(text)",
        inputs: &[BuiltinParamDescriptor {
            name: "text",
            ty: BuiltinParamType::Any,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "Input host text.",
        }],
        outputs: &[BuiltinParamDescriptor {
            name: "out",
            ty: BuiltinParamType::Any,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "Host result.",
        }],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        REVERSE_ERROR_TOO_MANY_OUTPUTS,
        REVERSE_ERROR_INVALID_INPUT,
        REVERSE_ERROR_PROCESSING,
    ],
};

pub const REVERSE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REVERSE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:reverse:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "reverse: too many output arguments",
};

pub const REVERSE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REVERSE.INVALID_INPUT",
    identifier: Some("RunMat:reverse:InvalidInput"),
    when: "A required text role contains unsupported input data.",
    message: "reverse: expected supported host text input",
};
pub const REVERSE_ERROR_PROCESSING: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
 code: "RM.REVERSE.PROCESSING", identifier: None,
 when: "Shared text/pattern processing rejects unsupported nested input, pattern syntax, geometry, or output construction.",
 message: "reverse: text processing failed",
};
