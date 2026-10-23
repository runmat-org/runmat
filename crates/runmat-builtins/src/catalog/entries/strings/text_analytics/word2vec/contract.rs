use crate::*;

pub const WORD2VEC_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WORD2VEC.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:word2vec:TooManyOutputs"),
    when: "The requested output count exceeds the canonical signatures.",
    message: "word2vec: too many output arguments",
};

const OUT_MATRIX: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "M",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Embedding vectors, one word per row.",
}];

const IN_WORDS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "emb",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "wordEmbedding object.",
    },
    BuiltinParamDescriptor {
        name: "words",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Words to map to vectors.",
    },
];

const IN_WORDS_REST: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "emb",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "wordEmbedding object.",
    },
    BuiltinParamDescriptor {
        name: "words",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Words to map to vectors.",
    },
    BuiltinParamDescriptor {
        name: "NameValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value options: IgnoreCase.",
    },
];

pub const WORD2VEC_ERROR_WORD2VEC_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WORD2VEC.INVALID_INPUT",
    identifier: Some("RunMat:word2vec:InvalidInput"),
    when: "Inputs do not match a supported word2vec form.",
    message: "word2vec received invalid input",
};

const WORD2VEC_ERRORS: [BuiltinErrorDescriptor; 2] = [
    WORD2VEC_ERROR_WORD2VEC_INVALID_INPUT,
    WORD2VEC_ERROR_TOO_MANY_OUTPUTS,
];

pub const WORD2VEC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "M = word2vec(emb, words)",
            inputs: &IN_WORDS,
            outputs: &OUT_MATRIX,
        },
        BuiltinSignatureDescriptor {
            label: "M = word2vec(emb, words, 'IgnoreCase', true)",
            inputs: &IN_WORDS_REST,
            outputs: &OUT_MATRIX,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &WORD2VEC_ERRORS,
};

pub const WORD2VEC_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "word2vec accepts a wordEmbedding object, textual words, and a logical IgnoreCase option and returns floating embedding rows. Integer and resident numeric word/control inputs are invalid and reject before provider access.",
};
