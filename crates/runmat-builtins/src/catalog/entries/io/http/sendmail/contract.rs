use crate::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SendmailFailure {
    Outputs,
    Arity,
    InvalidMailInput,
    InvalidText,
    EmptyRecipient,
    Attachments,
    Outbox,
    Command,
    Preferences,
    Smtp,
    NoTransport,
}

impl SendmailFailure {
    pub const fn identifier(self) -> &'static str {
        match self {
            Self::Outputs => "RunMat:sendmail:TooManyOutputs",
            Self::Arity => "RunMat:sendmail:ArgumentCount",
            Self::InvalidMailInput => "RunMat:sendmail:InvalidMailInput",
            Self::InvalidText => "RunMat:sendmail:InvalidText",
            Self::EmptyRecipient => "RunMat:sendmail:EmptyRecipient",
            Self::Attachments => "RunMat:sendmail:UnsupportedAttachments",
            Self::Outbox => "RunMat:sendmail:OutboxFailure",
            Self::Command => "RunMat:sendmail:CommandFailure",
            Self::Preferences => "RunMat:sendmail:PreferenceFailure",
            Self::Smtp => "RunMat:sendmail:SmtpUnavailable",
            Self::NoTransport => "RunMat:sendmail:NoTransport",
        }
    }
}

pub const SENDMAIL_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SENDMAIL.OUTPUTS",
    identifier: Some(SendmailFailure::Outputs.identifier()),
    when: "More than one output is requested.",
    message: "sendmail: too many output arguments",
};

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

const ERRORS: &[BuiltinErrorDescriptor] = &[
    SENDMAIL_ERROR_OUTPUTS,
    BuiltinErrorDescriptor {
        code: "RM.SENDMAIL.ARITY",
        identifier: Some(SendmailFailure::Arity.identifier()),
        when: "Argument count is outside the two-to-four-argument form.",
        message:
            "sendmail: recipient, subject, optional message, and optional attachments are expected",
    },
    BuiltinErrorDescriptor {
        code: "RM.SENDMAIL.INPUT",
        identifier: Some(SendmailFailure::InvalidMailInput.identifier()),
        when: "A direct numeric, integer, or resident value is supplied as mail text.",
        message: "sendmail: mail arguments must be text, not numeric or resident values",
    },
    BuiltinErrorDescriptor {
        code: "RM.SENDMAIL.TEXT",
        identifier: Some(SendmailFailure::InvalidText.identifier()),
        when: "Subject, message, or recipient element is not scalar text.",
        message: "sendmail: the argument must be a string scalar or character vector",
    },
    BuiltinErrorDescriptor {
        code: "RM.SENDMAIL.RECIPIENT",
        identifier: Some(SendmailFailure::EmptyRecipient.identifier()),
        when: "Every recipient address is empty.",
        message: "sendmail: recipient must not be empty",
    },
    BuiltinErrorDescriptor {
        code: "RM.SENDMAIL.ATTACHMENTS",
        identifier: Some(SendmailFailure::Attachments.identifier()),
        when: "A fourth attachment argument is supplied.",
        message: "sendmail: attachments are not supported by the current mail transport",
    },
    BuiltinErrorDescriptor {
        code: "RM.SENDMAIL.OUTBOX",
        identifier: Some(SendmailFailure::Outbox.identifier()),
        when: "Creating or writing the configured outbox fails.",
        message: "sendmail: unable to write the configured outbox",
    },
    BuiltinErrorDescriptor {
        code: "RM.SENDMAIL.COMMAND",
        identifier: Some(SendmailFailure::Command.identifier()),
        when:
            "The local mail command cannot start, accept the message, or completes unsuccessfully.",
        message: "sendmail: local command failed",
    },
    BuiltinErrorDescriptor {
        code: "RM.SENDMAIL.PREF",
        identifier: Some(SendmailFailure::Preferences.identifier()),
        when: "The installed Internet preference service reports an operational failure.",
        message: "sendmail: unable to read Internet preferences",
    },
    BuiltinErrorDescriptor {
        code: "RM.SENDMAIL.SMTP",
        identifier: Some(SendmailFailure::Smtp.identifier()),
        when: "Only a direct SMTP server is configured.",
        message: "sendmail: direct SMTP transport is not available",
    },
    BuiltinErrorDescriptor {
        code: "RM.SENDMAIL.TRANSPORT",
        identifier: Some(SendmailFailure::NoTransport.identifier()),
        when: "No local command, outbox, or usable transport exists.",
        message: "sendmail: no mail transport configured",
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
    errors: ERRORS,
};

pub const SENDMAIL_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Direct integer scalar, array, nested, and resident numeric arguments are not mail text and reject before provider access or transport effects. Numeric character codes already in a character vector remain text.",
};
