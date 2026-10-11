use super::super::errors::*;
use crate::*;
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "prop",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Live matlab.metadata.DynamicProperty metadata handle.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "delete(prop)",
    inputs: &INPUTS,
    outputs: &[],
}];
pub const DYNAMIC_PROPERTY_DELETE_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DYNAMICPROPS.DELETE_TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:dynamicprops:TooManyOutputs"),
    when: "Any output is requested from the qualified delete method.",
    message: "matlab.metadata.DynamicProperty.delete: no outputs are available",
};
pub const DYNAMIC_PROPERTY_DELETE_ERRORS: [BuiltinErrorDescriptor; 3] = [
    DYNAMIC_ERROR_INVALID_TARGET,
    DYNAMIC_ERROR_INVALID_METADATA,
    DYNAMIC_PROPERTY_DELETE_ERROR_OUTPUTS,
];
pub const DYNAMIC_PROPERTY_DELETE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DYNAMIC_PROPERTY_DELETE_ERRORS,
};
pub const DYNAMIC_PROPERTY_DELETE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None,
    notes: "The qualified delete method accepts only a dynamic-property metadata handle; integer and resident numeric values are not arguments or outputs.",
};
