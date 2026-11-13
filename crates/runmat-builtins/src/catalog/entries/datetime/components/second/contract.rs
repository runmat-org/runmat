use crate::*;

pub const SECOND_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "second accepts only a datetime object and returns its double second component. Native integer host or resident values are not datetime objects and reject without conversion or provider access; legacy serial-date parsing belongs to the separately documented date conversion APIs.",
};
pub const SECOND_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SECOND.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:second:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "second: too many output arguments",
};
pub const DATETIME_SECOND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "s = second(t)",
        inputs: &[BuiltinParamDescriptor {
            name: "t", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Required,
            default: None, description: "Host datetime scalar or array; no text or numeric conversion.",
        }],
        outputs: &[BuiltinParamDescriptor {
            name: "s", ty: BuiltinParamType::NumericArray, arity: BuiltinParamArity::Required,
            default: None, description: "Real double seconds with the stored datetime array shape, subject to serial-date precision.",
        }],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[DATETIME_ERROR_INVALID_INPUT, SECOND_ERROR_TOO_MANY_OUTPUTS],
};
