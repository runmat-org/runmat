use runmat_builtins::{
    BuiltinErrorDescriptor, GCD_ERROR_INTERNAL, GCD_ERROR_INVALID_INPUT, GCD_ERROR_OVERFLOW,
    GCD_ERROR_SIZE_MISMATCH, LCM_ERROR_INTERNAL, LCM_ERROR_INVALID_INPUT, LCM_ERROR_OVERFLOW,
    LCM_ERROR_SIZE_MISMATCH,
};

use crate::{build_runtime_error, RuntimeError};

pub(in crate::builtins::math::discrete) struct BinaryContext {
    pub(in crate::builtins::math::discrete) name: &'static str,
    pub(in crate::builtins::math::discrete) invalid: &'static BuiltinErrorDescriptor,
    pub(in crate::builtins::math::discrete) size_mismatch: &'static BuiltinErrorDescriptor,
    pub(in crate::builtins::math::discrete) overflow: &'static BuiltinErrorDescriptor,
    pub(in crate::builtins::math::discrete) internal: &'static BuiltinErrorDescriptor,
    pub(in crate::builtins::math::discrete) accepts_zero_or_negative: bool,
}

pub(in crate::builtins::math::discrete) const LCM_CONTEXT: BinaryContext = BinaryContext {
    name: "lcm",
    invalid: &LCM_ERROR_INVALID_INPUT,
    size_mismatch: &LCM_ERROR_SIZE_MISMATCH,
    overflow: &LCM_ERROR_OVERFLOW,
    internal: &LCM_ERROR_INTERNAL,
    accepts_zero_or_negative: false,
};

pub(in crate::builtins::math::discrete) const GCD_CONTEXT: BinaryContext = BinaryContext {
    name: "gcd",
    invalid: &GCD_ERROR_INVALID_INPUT,
    size_mismatch: &GCD_ERROR_SIZE_MISMATCH,
    overflow: &GCD_ERROR_OVERFLOW,
    internal: &GCD_ERROR_INTERNAL,
    accepts_zero_or_negative: true,
};

pub(in crate::builtins::math::discrete) fn binary_error(
    context: &'static BinaryContext,
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("{}: {detail}", error.message)).with_builtin(context.name);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
