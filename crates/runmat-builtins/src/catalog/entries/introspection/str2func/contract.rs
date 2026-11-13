use crate::*;
pub const STR2FUNC_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fh",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Function handle value.",
}];

pub const STR2FUNC_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "name",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Function handle text.",
}];

pub const STR2FUNC_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "fh = str2func(name)",
    inputs: &STR2FUNC_INPUTS,
    outputs: &STR2FUNC_OUTPUT,
}];

pub const STR2FUNC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &STR2FUNC_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &STR2FUNC_PUBLIC_ERRORS,
};

pub const STR2FUNC_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "str2func accepts scalar function-name text only. Integer, numeric, and provider-resident values reject before provider access, and the builtin has no integer output surface.",
    };

pub const STR2FUNC_ERROR_NAME_SHAPE_INVALID: crate::BuiltinErrorDescriptor =
    crate::BuiltinErrorDescriptor {
        code: "RM.STR2FUNC.NAME_SHAPE_INVALID",
        identifier: Some("RunMat:Str2FuncNameShapeInvalid"),
        when: "Input text value is not scalar row text.",
        message: "str2func: function name text input must be scalar row text",
    };

pub const STR2FUNC_ERROR_NAME_TYPE_INVALID: crate::BuiltinErrorDescriptor =
    crate::BuiltinErrorDescriptor {
        code: "RM.STR2FUNC.NAME_TYPE_INVALID",
        identifier: Some("RunMat:Str2FuncNameTypeInvalid"),
        when: "Input is not string or char text.",
        message: "str2func: expected string/char function name",
    };

pub const STR2FUNC_ERROR_NAME_INVALID: crate::BuiltinErrorDescriptor =
    crate::BuiltinErrorDescriptor {
        code: "RM.STR2FUNC.NAME_INVALID",
        identifier: Some("RunMat:Str2FuncNameInvalid"),
        when: "Parsed function name is empty.",
        message: "str2func: function name must not be empty",
    };

pub const STR2FUNC_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STR2FUNC.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:str2func:TooManyOutputs"),
    when: "More outputs are requested than this form supports.",
    message: "str2func: too many output arguments",
};
const STR2FUNC_PUBLIC_ERRORS: [BuiltinErrorDescriptor; 4] = [
    STR2FUNC_ERROR_NAME_SHAPE_INVALID,
    STR2FUNC_ERROR_NAME_TYPE_INVALID,
    STR2FUNC_ERROR_NAME_INVALID,
    STR2FUNC_ERROR_TOO_MANY_OUTPUTS,
];
