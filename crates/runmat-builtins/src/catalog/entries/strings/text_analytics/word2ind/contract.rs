use crate::*;

pub const WORD2IND_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WORD2IND.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:word2ind:TooManyOutputs"),
    when: "The requested output count exceeds the canonical signatures.",
    message: "word2ind: too many output arguments",
};

const OUT_INDICES: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "M",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Word encoding indices, with NaN for words outside the vocabulary.",
}];

const IN_WORDS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "enc",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "wordEncoding object.",
    },
    BuiltinParamDescriptor {
        name: "words",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Words to map to indices.",
    },
];

const IN_WORDS_REST: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "enc",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "wordEncoding object.",
    },
    BuiltinParamDescriptor {
        name: "words",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Words to map to indices.",
    },
    BuiltinParamDescriptor {
        name: "NameValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value options: IgnoreCase.",
    },
];

pub const WORD2IND_ERROR_WORD2IND_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WORD2IND.INVALID_INPUT",
    identifier: Some("RunMat:word2ind:InvalidInput"),
    when: "Inputs do not match a supported word2ind form.",
    message: "word2ind received invalid input",
};

const WORD2IND_ERRORS: [BuiltinErrorDescriptor; 2] = [
    WORD2IND_ERROR_WORD2IND_INVALID_INPUT,
    WORD2IND_ERROR_TOO_MANY_OUTPUTS,
];

pub const WORD2IND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "M = word2ind(enc, words)",
            inputs: &IN_WORDS,
            outputs: &OUT_INDICES,
        },
        BuiltinSignatureDescriptor {
            label: "M = word2ind(enc, words, 'IgnoreCase', true)",
            inputs: &IN_WORDS_REST,
            outputs: &OUT_INDICES,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &WORD2IND_ERRORS,
};

pub const WORD2IND_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "word2ind accepts a wordEncoding object, textual words, and a logical IgnoreCase option. It returns double indices or NaN; integer and resident numeric word/control inputs are invalid and reject before provider access.",
};
