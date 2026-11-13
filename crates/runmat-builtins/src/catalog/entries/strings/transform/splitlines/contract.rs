use crate::*;
pub const SPLITLINES_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "splitlines accepts host string, character, or cell text and returns a string or cell text container. Numeric, logical, symbolic, and provider-resident values reject before gather or provider access.",
    };
pub const SPLITLINES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "out = splitlines(text)",
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
        SPLITLINES_ERROR_TOO_MANY_OUTPUTS,
        SPLITLINES_ERROR_INVALID_INPUT,
        SPLITLINES_ERROR_PROCESSING,
    ],
};

pub const SPLITLINES_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPLITLINES.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:splitlines:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "splitlines: too many output arguments",
};

pub const SPLITLINES_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPLITLINES.INVALID_INPUT",
    identifier: Some("RunMat:splitlines:InvalidInput"),
    when: "A required text role contains unsupported input data.",
    message: "splitlines: expected supported host text input",
};
pub const SPLITLINES_ERROR_PROCESSING: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
 code: "RM.SPLITLINES.PROCESSING", identifier: None,
 when: "Shared text/pattern processing rejects unsupported nested input, pattern syntax, geometry, or output construction.",
 message: "splitlines: text processing failed",
};
