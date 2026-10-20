use crate::*;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "words",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Column string array of stop words for the selected language.",
}];
const LANGUAGE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "Name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("Language"),
        description: "Case-insensitive Language option name.",
    },
    BuiltinParamDescriptor {
        name: "language",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("en"),
        description: "Language code: en, ja, de, or ko.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "words = stopWords",
        inputs: &[],
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "words = stopWords('Language', language)",
        inputs: &LANGUAGE_INPUTS,
        outputs: &OUTPUTS,
    },
];

pub const STOP_WORDS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STOPWORDS.INVALID_INPUT",
    identifier: Some("RunMat:stopWords:InvalidInput"),
    when: "Inputs are not a supported stopWords form.",
    message: "stopWords: invalid input",
};
const ERRORS: [BuiltinErrorDescriptor; 1] = [STOP_WORDS_ERROR_INVALID_INPUT];
pub const STOP_WORDS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const STOP_WORDS_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "stopWords accepts only host scalar text controls. Numeric and resident controls reject before provider access; the output is host string metadata.",
};
