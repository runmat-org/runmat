use crate::*;

pub const TOKENIZED_DOCUMENT_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.TOKENIZED_DOCUMENT.TOO_MANY_OUTPUTS",
        identifier: Some("RunMat:tokenizedDocument:TooManyOutputs"),
        when: "The requested output count exceeds the canonical signatures.",
        message: "tokenizedDocument: too many output arguments",
    };

const OUT_DOCUMENTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "documents",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Tokenized document object.",
}];

const IN_TEXT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "str",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input text or pre-tokenized words.",
}];

const IN_TEXT_REST: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text or pre-tokenized words.",
    },
    BuiltinParamDescriptor {
        name: "NameValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description:
            "Name-value options: TokenizeMethod, Language, DetectPatterns, TopLevelDomains, CustomTokens, RegularExpressions.",
    },
];

pub const TOKENIZED_DOCUMENT_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEXT_ANALYTICS_DOCUMENTS.INVALID_INPUT",
    identifier: Some("RunMat:textAnalyticsDocuments:InvalidInput"),
    when: "Inputs do not match a supported Text Analytics document or model helper form.",
    message: "Text Analytics document helper received invalid input",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [
    TOKENIZED_DOCUMENT_ERROR_INVALID_INPUT,
    TOKENIZED_DOCUMENT_ERROR_TOO_MANY_OUTPUTS,
];

pub const TOKENIZED_DOCUMENT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "documents = tokenizedDocument",
            inputs: &[],
            outputs: &OUT_DOCUMENTS,
        },
        BuiltinSignatureDescriptor {
            label: "documents = tokenizedDocument(str)",
            inputs: &IN_TEXT,
            outputs: &OUT_DOCUMENTS,
        },
        BuiltinSignatureDescriptor {
            label: "documents = tokenizedDocument(str, Name, Value, ...)",
            inputs: &IN_TEXT_REST,
            outputs: &OUT_DOCUMENTS,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const TOKENIZED_DOCUMENT_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "tokenizedDocument accepts text, text containers, and text-valued tokenization options. Native integer arrays are neither document text nor valid option values; generated document counts remain double object metadata.",
    };
