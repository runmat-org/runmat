use crate::*;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "updatedDocuments",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Updated tokenizedDocument object with entity metadata.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "documents",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "tokenizedDocument object.",
}];
const INPUTS_REST: [BuiltinParamDescriptor; 2] = [
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
        description: "Name-value options: RetokenizeMethod, DiscardKnownValues, Model.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "updatedDocuments = addEntityDetails(documents)",
        inputs: &INPUTS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "updatedDocuments = addEntityDetails(documents,Name,Value)",
        inputs: &INPUTS_REST,
        outputs: &OUTPUTS,
    },
];
pub const ADD_ENTITY_DETAILS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEXT_ANALYTICS.ADD_ENTITY_DETAILS.INVALID_INPUT",
    identifier: Some("RunMat:addEntityDetails:InvalidInput"),
    when: "Input is not a supported tokenizedDocument object or option form.",
    message: "addEntityDetails: invalid input",
};
const ERRORS: [BuiltinErrorDescriptor; 1] = [ADD_ENTITY_DETAILS_ERROR_INVALID_INPUT];
pub const ADD_ENTITY_DETAILS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const ADD_ENTITY_DETAILS_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "addEntityDetails accepts a tokenizedDocument, text or model options, and a logical DiscardKnownValues control; integer arrays are neither document data nor valid option values. The returned object stores categorical entity tags.",
};
