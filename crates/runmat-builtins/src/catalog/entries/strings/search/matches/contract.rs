use crate::*;
pub const MATCHES_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "matches accepts exactly text and one scalar text or pattern input. IgnoreCase is not implemented. Numeric and integer data has no documented role and rejects without implicit text conversion or provider access.",
};
pub const MATCHES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "tf = matches(text, pattern)",
        inputs: &[
            BuiltinParamDescriptor {
                name: "text",
                ty: BuiltinParamType::Any,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Input host text.",
            },
            BuiltinParamDescriptor {
                name: "pattern",
                ty: BuiltinParamType::Any,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Whole-text pattern.",
            },
        ],
        outputs: &[BuiltinParamDescriptor {
            name: "out",
            ty: BuiltinParamType::LogicalArray,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "Host result.",
        }],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        MATCHES_ERROR_TOO_MANY_OUTPUTS,
        MATCHES_ERROR_INVALID_INPUT,
        MATCHES_ERROR_PROCESSING,
    ],
};

pub const MATCHES_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MATCHES.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:matches:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "matches: too many output arguments",
};

pub const MATCHES_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MATCHES.INVALID_INPUT",
    identifier: Some("RunMat:matches:InvalidInput"),
    when: "A required text role contains unsupported input data.",
    message: "matches: expected supported host text or pattern input",
};
pub const MATCHES_ERROR_PROCESSING: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
 code: "RM.MATCHES.PROCESSING", identifier: None,
 when: "Shared text/pattern processing rejects unsupported nested input, pattern syntax, geometry, or output construction.",
 message: "matches: text processing failed",
};
