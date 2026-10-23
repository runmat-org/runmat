use crate::*;
pub const ISMETHOD_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when the object has the requested public method.",
}];

pub const ISMETHOD_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Object instance or gpuArray object to inspect.",
    },
    BuiltinParamDescriptor {
        name: "method_name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Method name as a string scalar or row character vector.",
    },
];

pub const ISMETHOD_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = ismethod(obj, method_name)",
    inputs: &ISMETHOD_INPUTS,
    outputs: &ISMETHOD_OUTPUT,
}];

pub const ISMETHOD_ERROR_NAME_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISMETHOD.NAME_INVALID",
    identifier: Some("RunMat:IsMethodNameInvalid"),
    when: "The method-name argument is not a string scalar or row character vector.",
    message: "ismethod: method name must be a string scalar or character vector",
};

pub const ISMETHOD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISMETHOD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ISMETHOD_PUBLIC_ERRORS,
};

pub const ISMETHOD_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "ismethod is an object-method predicate rather than an integer computation: host and internally auto-resident integers return false, while explicit integer gpuArrays use the production gpuArray wrapper method registry without payload access.",
    };

pub const ISMETHOD_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISMETHOD.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:ismethod:TooManyOutputs"),
    when: "More outputs are requested than this form supports.",
    message: "ismethod: too many output arguments",
};
const ISMETHOD_PUBLIC_ERRORS: [BuiltinErrorDescriptor; 2] =
    [ISMETHOD_ERROR_NAME_INVALID, ISMETHOD_ERROR_TOO_MANY_OUTPUTS];
