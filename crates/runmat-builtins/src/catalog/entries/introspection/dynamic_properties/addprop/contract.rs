use super::super::errors::*;
use crate::*;
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "prop",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Live matlab.metadata.DynamicProperty metadata handle.",
}];
const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "object",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Live scalar dynamicprops handle object.",
    },
    BuiltinParamDescriptor {
        name: "property_name",
        ty: BuiltinParamType::PropertyName,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "New valid identifier.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "prop = addprop(object, property_name)",
        inputs: &INPUTS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "addprop(object, property_name)",
        inputs: &INPUTS,
        outputs: &[],
    },
];
pub const ADDPROP_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ADDPROP.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:addprop:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "addprop: expected at most one output",
};
pub const ADDPROP_ERROR_ALLOCATION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ADDPROP.METADATA_ALLOCATION",
    identifier: Some("RunMat:dynamicprops:MetadataAllocationFailed"),
    when: "The metadata handle cannot be allocated before property publication.",
    message: "addprop: failed to allocate dynamic property metadata",
};
pub const ADDPROP_ERRORS: [BuiltinErrorDescriptor; 7] = [
    DYNAMIC_ERROR_INVALID_TARGET,
    DYNAMIC_ERROR_INVALID_NAME,
    DYNAMIC_ERROR_DUPLICATE,
    DYNAMIC_ERROR_INVALID_METADATA,
    DYNAMIC_ERROR_UNSUPPORTED_METADATA,
    ADDPROP_ERROR_OUTPUTS,
    ADDPROP_ERROR_ALLOCATION,
];
pub const ADDPROP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ADDPROP_ERRORS,
};
pub const ADDPROP_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None,
    notes: "addprop takes a dynamicprops handle and text name; numeric or resident values are not controls. Integer or resident property values may be assigned later through ordinary object mutation.",
};
