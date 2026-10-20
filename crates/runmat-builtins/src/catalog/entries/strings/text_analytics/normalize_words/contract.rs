use crate::*;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "updatedWords",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Normalized words or tokenizedDocument object.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "words",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Words to stem or lemmatize.",
}];
const INPUTS_REST: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "words",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Words to stem or lemmatize.",
    },
    BuiltinParamDescriptor {
        name: "NameValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value options: Language and Style.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "updatedDocuments = normalizeWords(documents)",
        inputs: &INPUTS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "updatedWords = normalizeWords(words)",
        inputs: &INPUTS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "updatedWords = normalizeWords(words, Name, Value, ...)",
        inputs: &INPUTS_REST,
        outputs: &OUTPUTS,
    },
];
pub const NORMALIZE_WORDS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NORMALIZEWORDS.INVALID_INPUT",
    identifier: Some("RunMat:normalizeWords:InvalidInput"),
    when: "Inputs are not a supported normalizeWords form.",
    message: "normalizeWords: invalid input",
};
const ERRORS: [BuiltinErrorDescriptor; 1] = [NORMALIZE_WORDS_ERROR_INVALID_INPUT];
pub const NORMALIZE_WORDS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const NORMALIZE_WORDS_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "normalizeWords accepts tokenizedDocument objects or host string, character, and cell text plus textual Language and Style values. Numeric, logical, symbolic, and provider-resident inputs reject before gather or provider access.",
};
