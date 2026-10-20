use crate::*;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "updatedDocuments",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Updated tokenizedDocument object with lemma metadata.",
}];
const DOCUMENT_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "documents",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "tokenizedDocument object.",
}];
const OPTION_INPUTS: [BuiltinParamDescriptor; 2] = [
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
        description: "DiscardKnownValues logical name-value pairs.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "updatedDocuments = addLemmaDetails(documents)",
        inputs: &DOCUMENT_INPUT,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "updatedDocuments = addLemmaDetails(documents,'DiscardKnownValues',tf,...)",
        inputs: &OPTION_INPUTS,
        outputs: &OUTPUTS,
    },
];
pub const ADD_LEMMA_DETAILS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEXT_ANALYTICS.ADD_LEMMA_DETAILS.INVALID_INPUT",
    identifier: Some("RunMat:addLemmaDetails:InvalidInput"),
    when: "Input is not a supported tokenizedDocument object or option form.",
    message: "addLemmaDetails: invalid input",
};
const ERRORS: [BuiltinErrorDescriptor; 1] = [ADD_LEMMA_DETAILS_ERROR_INVALID_INPUT];
pub const ADD_LEMMA_DETAILS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const ADD_LEMMA_DETAILS_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "addLemmaDetails accepts a tokenizedDocument and a logical DiscardKnownValues control; integer arrays are neither document data nor valid option values. The returned object stores text lemma metadata.",
};
