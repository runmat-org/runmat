use crate::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UrldecodeFailure {
    WrongArity,
    InvalidText,
    TooManyOutputs,
    IncompleteEscape,
    InvalidEscape,
    InvalidUtf8,
}
impl UrldecodeFailure {
    pub const fn descriptor(self) -> &'static BuiltinErrorDescriptor {
        match self {
            Self::WrongArity => &URLDECODE_ERROR_WRONG_ARITY,
            Self::InvalidText => &URLDECODE_ERROR_INVALID_TEXT,
            Self::TooManyOutputs => &URLDECODE_ERROR_TOO_MANY_OUTPUTS,
            Self::IncompleteEscape => &URLDECODE_ERROR_INCOMPLETE_ESCAPE,
            Self::InvalidEscape => &URLDECODE_ERROR_INVALID_ESCAPE,
            Self::InvalidUtf8 => &URLDECODE_ERROR_INVALID_UTF8,
        }
    }
    pub const fn identifier(self) -> &'static str {
        match self {
            Self::WrongArity => "RunMat:urldecode:WrongArity",
            Self::InvalidText => "RunMat:urldecode:InvalidText",
            Self::TooManyOutputs => "RunMat:urldecode:TooManyOutputs",
            Self::IncompleteEscape => "RunMat:urldecode:IncompleteEscape",
            Self::InvalidEscape => "RunMat:urldecode:InvalidEscape",
            Self::InvalidUtf8 => "RunMat:urldecode:InvalidUtf8",
        }
    }
}
pub const URLDECODE_ERROR_WRONG_ARITY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.URLDECODE.WRONG_ARITY",
    identifier: Some("RunMat:urldecode:WrongArity"),
    when: "The input count is not one.",
    message: "urldecode: expected exactly one input",
};
pub const URLDECODE_ERROR_INVALID_TEXT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.URLDECODE.INVALID_TEXT",
    identifier: Some("RunMat:urldecode:InvalidText"),
    when: "Input is not a host string scalar or character row.",
    message: "urldecode: text must be a host string scalar or character vector",
};
pub const URLDECODE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.URLDECODE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:urldecode:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "urldecode: expected at most one output",
};
pub const URLDECODE_ERROR_INCOMPLETE_ESCAPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.URLDECODE.INCOMPLETE_ESCAPE",
    identifier: Some("RunMat:urldecode:IncompleteEscape"),
    when: "A percent escape has fewer than two following bytes.",
    message: "urldecode: incomplete escape",
};
pub const URLDECODE_ERROR_INVALID_ESCAPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.URLDECODE.INVALID_ESCAPE",
    identifier: Some("RunMat:urldecode:InvalidEscape"),
    when: "A percent escape does not contain two hexadecimal ASCII digits.",
    message: "urldecode: invalid escape",
};
pub const URLDECODE_ERROR_INVALID_UTF8: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.URLDECODE.INVALID_UTF8",
    identifier: Some("RunMat:urldecode:InvalidUtf8"),
    when: "Decoded bytes do not form valid UTF-8 text.",
    message: "urldecode: decoded text is not valid UTF-8",
};
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "text", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Required, default: None,
    description: "Host string scalar (including a singleton string array) or character row; numeric and provider-resident inputs reject.",
}];
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "decoded",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Fresh host character row with content-dependent width.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "decoded = urldecode(text)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
const ERRORS: [BuiltinErrorDescriptor; 6] = [
    URLDECODE_ERROR_WRONG_ARITY,
    URLDECODE_ERROR_INVALID_TEXT,
    URLDECODE_ERROR_TOO_MANY_OUTPUTS,
    URLDECODE_ERROR_INCOMPLETE_ESCAPE,
    URLDECODE_ERROR_INVALID_ESCAPE,
    URLDECODE_ERROR_INVALID_UTF8,
];
pub const URLDECODE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const URLDECODE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None,
    notes: "urldecode accepts host text only; integer and resident numeric values reject before provider access or decoding.",
};
