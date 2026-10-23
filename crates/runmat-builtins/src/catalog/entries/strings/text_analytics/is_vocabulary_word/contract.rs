use crate::*;

pub const IS_VOCABULARY_WORD_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.IS_VOCABULARY_WORD.TOO_MANY_OUTPUTS",
        identifier: Some("RunMat:isVocabularyWord:TooManyOutputs"),
        when: "The requested output count exceeds the canonical signatures.",
        message: "isVocabularyWord: too many output arguments",
    };

const OUT_LOGICAL: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical membership mask.",
}];

const IN_VOCABULARY_WORDS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "embOrEnc",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "wordEmbedding or wordEncoding object.",
    },
    BuiltinParamDescriptor {
        name: "words",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Words to test.",
    },
];

const IN_VOCABULARY_WORDS_REST: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "embOrEnc",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "wordEmbedding or wordEncoding object.",
    },
    BuiltinParamDescriptor {
        name: "words",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Words to test.",
    },
    BuiltinParamDescriptor {
        name: "NameValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value options: IgnoreCase.",
    },
];

pub const IS_VOCABULARY_WORD_ERROR_IS_VOCABULARY_WORD_INVALID_INPUT: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.ISVOCABULARYWORD.INVALID_INPUT",
        identifier: Some("RunMat:isVocabularyWord:InvalidInput"),
        when: "Inputs do not match a supported isVocabularyWord form.",
        message: "isVocabularyWord received invalid input",
    };

const IS_VOCABULARY_WORD_ERRORS: [BuiltinErrorDescriptor; 2] = [
    IS_VOCABULARY_WORD_ERROR_IS_VOCABULARY_WORD_INVALID_INPUT,
    IS_VOCABULARY_WORD_ERROR_TOO_MANY_OUTPUTS,
];

pub const IS_VOCABULARY_WORD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "tf = isVocabularyWord(emb, words)",
            inputs: &IN_VOCABULARY_WORDS,
            outputs: &OUT_LOGICAL,
        },
        BuiltinSignatureDescriptor {
            label: "tf = isVocabularyWord(enc, words)",
            inputs: &IN_VOCABULARY_WORDS,
            outputs: &OUT_LOGICAL,
        },
        BuiltinSignatureDescriptor {
            label: "tf = isVocabularyWord(___, 'IgnoreCase', true)",
            inputs: &IN_VOCABULARY_WORDS_REST,
            outputs: &OUT_LOGICAL,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IS_VOCABULARY_WORD_ERRORS,
};

pub const IS_VOCABULARY_WORD_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "isVocabularyWord accepts vocabulary objects, textual words, and logical options; integer and resident numeric values are invalid and reject before provider access.",
    };
