use crate::*;

pub const REMOVE_STOP_WORDS_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.REMOVE_STOP_WORDS.TOO_MANY_OUTPUTS",
        identifier: Some("RunMat:removeStopWords:TooManyOutputs"),
        when: "The requested output count exceeds the canonical signatures.",
        message: "removeStopWords: too many output arguments",
    };

pub const REMOVE_STOP_WORDS_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "removeStopWords accepts a tokenizedDocument and logical IgnoreCase option; native integer and resident numeric arguments are rejected before provider access.",
    };

const OUT_DOCUMENTS_FILTERED: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "newDocuments",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Filtered tokenizedDocument object.",
}];

const IN_REMOVE_STOP: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "documents",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "tokenizedDocument object.",
    },
    BuiltinParamDescriptor {
        name: "NameValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value options: IgnoreCase.",
    },
];

const IN_REMOVE_STOP_DOCUMENTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "documents",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "tokenizedDocument object.",
}];

pub const REMOVE_STOP_WORDS_ERROR_REMOVE_STOP_WORDS_INVALID_INPUT: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.REMOVESTOPWORDS.INVALID_INPUT",
        identifier: Some("RunMat:removeStopWords:InvalidInput"),
        when: "Inputs do not match a supported removeStopWords form.",
        message: "removeStopWords: invalid input",
    };

const REMOVE_STOP_WORDS_ERRORS: [BuiltinErrorDescriptor; 2] = [
    REMOVE_STOP_WORDS_ERROR_REMOVE_STOP_WORDS_INVALID_INPUT,
    REMOVE_STOP_WORDS_ERROR_TOO_MANY_OUTPUTS,
];

pub const REMOVE_STOP_WORDS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "newDocuments = removeStopWords(documents)",
            inputs: &IN_REMOVE_STOP_DOCUMENTS,
            outputs: &OUT_DOCUMENTS_FILTERED,
        },
        BuiltinSignatureDescriptor {
            label: "newDocuments = removeStopWords(documents, Name, Value, ...)",
            inputs: &IN_REMOVE_STOP,
            outputs: &OUT_DOCUMENTS_FILTERED,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &REMOVE_STOP_WORDS_ERRORS,
};
