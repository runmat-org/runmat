use crate::*;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "updatedDocuments",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Updated tokenizedDocument object with Head and Dependency metadata.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "documents",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "tokenizedDocument object.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "updatedDocuments = addDependencyDetails(documents)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
pub const ADD_DEPENDENCY_DETAILS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.TEXT_ANALYTICS.ADD_DEPENDENCY_DETAILS.INVALID_INPUT",
        identifier: Some("RunMat:addDependencyDetails:InvalidInput"),
        when: "Input is not a supported tokenizedDocument object.",
        message: "addDependencyDetails: invalid input",
    };
const ERRORS: [BuiltinErrorDescriptor; 1] = [ADD_DEPENDENCY_DETAILS_ERROR_INVALID_INPUT];
pub const ADD_DEPENDENCY_DETAILS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const ADD_DEPENDENCY_DETAILS_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "addDependencyDetails accepts and returns tokenizedDocument objects only; it has no numeric arguments. The Head token indices exposed through tokenDetails are double metadata rather than integer-class results.",
};
