use crate::*;

const OUT_DETAILS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tdetails",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Token detail table.",
}];
const OUT_DOCUMENTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "newDocuments",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Updated tokenized document object.",
}];
const IN_DOCUMENTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "documents",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "tokenizedDocument object.",
}];
const IN_TYPE_REST: [BuiltinParamDescriptor; 2] = [
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
        description: "Name-value options: DiscardKnownValues, TopLevelDomains.",
    },
];
const IN_SENTENCE_REST: [BuiltinParamDescriptor; 2] = [
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
        description: "Name-value options: Abbreviations, Starters, DiscardKnownValues.",
    },
];
pub const TOKEN_DETAILS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEXT_ANALYTICS.TOKEN_DETAILS.INVALID_INPUT",
    identifier: Some("RunMat:tokenDetails:InvalidInput"),
    when: "Input is not a supported tokenizedDocument object or option form.",
    message: "tokenDetails: invalid input",
};
pub const ADD_TYPE_DETAILS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEXT_ANALYTICS.ADD_TYPE_DETAILS.INVALID_INPUT",
    identifier: Some("RunMat:addTypeDetails:InvalidInput"),
    when: "Input is not a supported tokenizedDocument object or option form.",
    message: "addTypeDetails: invalid input",
};
pub const ADD_SENTENCE_DETAILS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.TEXT_ANALYTICS.ADD_SENTENCE_DETAILS.INVALID_INPUT",
        identifier: Some("RunMat:addSentenceDetails:InvalidInput"),
        when: "Input is not a supported tokenizedDocument object or option form.",
        message: "addSentenceDetails: invalid input",
    };
const TOKEN_ERRORS: [BuiltinErrorDescriptor; 1] = [TOKEN_DETAILS_ERROR_INVALID_INPUT];
const TYPE_ERRORS: [BuiltinErrorDescriptor; 1] = [ADD_TYPE_DETAILS_ERROR_INVALID_INPUT];
const SENTENCE_ERRORS: [BuiltinErrorDescriptor; 1] = [ADD_SENTENCE_DETAILS_ERROR_INVALID_INPUT];
const TOKEN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tdetails = tokenDetails(documents)",
    inputs: &IN_DOCUMENTS,
    outputs: &OUT_DETAILS,
}];
const TYPE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "newDocuments = addTypeDetails(documents)",
        inputs: &IN_DOCUMENTS,
        outputs: &OUT_DOCUMENTS,
    },
    BuiltinSignatureDescriptor {
        label: "newDocuments = addTypeDetails(documents,Name,Value)",
        inputs: &IN_TYPE_REST,
        outputs: &OUT_DOCUMENTS,
    },
];
const SENTENCE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "newDocuments = addSentenceDetails(documents)",
        inputs: &IN_DOCUMENTS,
        outputs: &OUT_DOCUMENTS,
    },
    BuiltinSignatureDescriptor {
        label: "newDocuments = addSentenceDetails(documents,Name,Value)",
        inputs: &IN_SENTENCE_REST,
        outputs: &OUT_DOCUMENTS,
    },
];
pub const TOKEN_DETAILS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TOKEN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TOKEN_ERRORS,
};
pub const ADD_TYPE_DETAILS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TYPE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TYPE_ERRORS,
};
pub const ADD_SENTENCE_DETAILS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SENTENCE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SENTENCE_ERRORS,
};
pub const TOKEN_DETAILS_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None,
    notes: "tokenDetails accepts only tokenizedDocument objects. Generated document, sentence, line, and dependency ordinals are table metadata stored as double columns rather than caller-provided integer data.",
};
pub const ADD_TYPE_DETAILS_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None,
    notes: "addTypeDetails accepts a tokenizedDocument plus text collections and a logical DiscardKnownValues control; integer arrays are neither document data nor valid option values. The returned object stores categorical token types.",
};
pub const ADD_SENTENCE_DETAILS_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None,
    notes: "addSentenceDetails accepts a tokenizedDocument plus text collections and a logical DiscardKnownValues control; integer arrays are neither document data nor valid option values. The returned object stores sentence numbers as double metadata.",
};
