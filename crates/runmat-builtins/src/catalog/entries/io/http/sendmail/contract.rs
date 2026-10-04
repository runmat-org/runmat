use crate::*;

const INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "to",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Text address, string array, or cell array of text addresses.",
    },
    BuiltinParamDescriptor {
        name: "subject",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Message subject text.",
    },
    BuiltinParamDescriptor {
        name: "message",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some(""),
        description: "Optional plain-text body; omitted means empty.",
    },
    BuiltinParamDescriptor {
        name: "attachments",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Accepted argument position; attachment transport is not implemented.",
    },
];

pub const SENDMAIL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "status = sendmail(to, subject, message, attachments)",
        inputs: &INPUTS,
        outputs: &[BuiltinParamDescriptor {
            name: "status",
            ty: BuiltinParamType::NumericScalar,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "Zero on successful local delivery or outbox capture.",
        }],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[BuiltinErrorDescriptor {
        code: "RM.SENDMAIL.INPUT",
        identifier: Some("RunMat:sendmail:InvalidMailInput"),
        when: "A direct numeric, integer, or resident value is supplied as mail text.",
        message: "sendmail: mail arguments must be text, not numeric or resident values",
    }],
};

pub const SENDMAIL_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Direct integer scalar, array, nested, and resident numeric arguments are not mail text and reject before provider access or transport effects. Numeric character codes already in a character vector remain text.",
};
