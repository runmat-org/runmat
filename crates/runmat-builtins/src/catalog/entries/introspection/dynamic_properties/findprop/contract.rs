use super::super::errors::*;
use crate::*;
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "prop",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Dynamic metadata handle, class property metadata object, or empty double.",
}];
const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "object",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scalar object or live handle object.",
    },
    BuiltinParamDescriptor {
        name: "property_name",
        ty: BuiltinParamType::PropertyName,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Valid property identifier.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "prop = findprop(object, property_name)",
        inputs: &INPUTS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "findprop(object, property_name)",
        inputs: &INPUTS,
        outputs: &[],
    },
];
pub const FINDPROP_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FINDPROP.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:findprop:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "findprop: expected at most one output",
};
pub const FINDPROP_ERRORS: [BuiltinErrorDescriptor; 3] = [
    DYNAMIC_ERROR_INVALID_TARGET,
    DYNAMIC_ERROR_INVALID_NAME,
    FINDPROP_ERROR_OUTPUTS,
];
pub const FINDPROP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FINDPROP_ERRORS,
};
pub const FINDPROP_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None,
    notes: "findprop takes object metadata and text; all integer classes, logical, complex and resident numeric handles reject as direct receivers without conversion.",
};
