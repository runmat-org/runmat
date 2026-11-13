use crate::*;
pub const SYMS_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None,
    notes: "syms accepts scalar text declarations only. Numeric, integer and resident inputs are rejected without conversion or provider access.",
};
const SYMS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ans",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Empty return value for command declarations.",
}];

const SYMS_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Symbol names and optional assumption keywords.",
}];

const SYMS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "syms name ...",
    inputs: &SYMS_INPUTS,
    outputs: &SYMS_OUTPUT,
}];

pub const SYMS_ERRORS: [BuiltinErrorDescriptor; 4] = [
    BuiltinErrorDescriptor {
        code: "RM.SYMS.EMPTY",
        identifier: Some("RunMat:syms:EmptyDeclaration"),
        when: "No symbol names were supplied.",
        message: "syms: expected at least one symbolic variable name",
    },
    BuiltinErrorDescriptor {
        code: "RM.SYMS.INVALID_NAME",
        identifier: Some("RunMat:syms:InvalidName"),
        when: "A declaration token is not a valid MATLAB identifier or assumption keyword.",
        message: "syms: invalid symbolic variable name",
    },
    BuiltinErrorDescriptor {
        code: "RM.SYMS.WORKSPACE",
        identifier: Some("RunMat:syms:WorkspaceUnavailable"),
        when: "The active runtime cannot assign to the caller workspace.",
        message: "syms: workspace assignment unavailable",
    },
    SYMS_ERROR_TOO_MANY_OUTPUTS,
];

pub const SYMS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SYMS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SYMS_ERRORS,
};

pub const SYMS_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SYMS.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:syms:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "syms: too many output arguments",
};
