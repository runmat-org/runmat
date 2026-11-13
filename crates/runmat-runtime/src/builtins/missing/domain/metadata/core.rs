use runmat_builtins::BuiltinErrorDescriptor;

#[cfg(test)]
pub(in crate::builtins::missing::domain) const MISSING_TEXT: runmat_value::StringElement =
    runmat_value::StringElement::Missing;

pub(in crate::builtins::missing::domain) const MISSING_ERROR_INVALID_ARGUMENT:
    BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MISSING.INVALID_ARGUMENT",
    identifier: Some("RunMat:missing:InvalidArgument"),
    when: "Arguments do not match a supported missing-value syntax.",
    message: "missing-value builtin: invalid argument",
};
pub(in crate::builtins::missing::domain) const MISSING_ERROR_UNSUPPORTED_TYPE:
    BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MISSING.UNSUPPORTED_TYPE",
    identifier: Some("RunMat:missing:UnsupportedType"),
    when: "The input type has no missing-value representation in RunMat.",
    message: "missing-value builtin: unsupported input type",
};
pub(in crate::builtins::missing::domain) const MISSING_ERROR_INTERNAL: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.MISSING.INTERNAL",
        identifier: Some("RunMat:missing:InternalError"),
        when: "Internal shape or table materialization fails.",
        message: "missing-value builtin: internal error",
    };
