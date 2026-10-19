use crate::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UrlencodeFailure {
    WrongArity,
    InvalidText,
    TooManyOutputs,
}
impl UrlencodeFailure {
    pub const fn descriptor(self) -> &'static BuiltinErrorDescriptor {
        match self {
            Self::WrongArity => &URLENCODE_ERROR_WRONG_ARITY,
            Self::InvalidText => &URLENCODE_ERROR_INVALID_TEXT,
            Self::TooManyOutputs => &URLENCODE_ERROR_TOO_MANY_OUTPUTS,
        }
    }
    pub const fn identifier(self) -> &'static str {
        match self {
            Self::WrongArity => "RunMat:urlencode:WrongArity",
            Self::InvalidText => "RunMat:urlencode:InvalidText",
            Self::TooManyOutputs => "RunMat:urlencode:TooManyOutputs",
        }
    }
}
pub const URLENCODE_ERROR_WRONG_ARITY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.URLENCODE.WRONG_ARITY",
    identifier: Some("RunMat:urlencode:WrongArity"),
    when: "The input count is not one.",
    message: "urlencode: expected exactly one input",
};
pub const URLENCODE_ERROR_INVALID_TEXT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.URLENCODE.INVALID_TEXT",
    identifier: Some("RunMat:urlencode:InvalidText"),
    when: "Input is not a host string scalar or character row.",
    message: "urlencode: text must be a host string scalar or character vector",
};
pub const URLENCODE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.URLENCODE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:urlencode:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "urlencode: expected at most one output",
};
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "text", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Required, default: None,
    description: "Host string scalar (including a singleton string array) or character row; numeric and provider-resident inputs reject.",
}];
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "encoded",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Fresh host character row with content-dependent width.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "encoded = urlencode(text)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    URLENCODE_ERROR_WRONG_ARITY,
    URLENCODE_ERROR_INVALID_TEXT,
    URLENCODE_ERROR_TOO_MANY_OUTPUTS,
];
pub const URLENCODE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const URLENCODE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None,
    notes: "urlencode accepts host text only; integer and resident numeric values reject before provider access or encoding.",
};
