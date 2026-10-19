use runmat_builtins::{
    BuiltinErrorDescriptor, ABS_ERROR_INVALID_INPUT, ANGLE_ERROR_INVALID_INPUT,
    SIGN_ERROR_INVALID_INPUT,
};
use runmat_value::Value;

use crate::{build_runtime_error, RuntimeError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MagnitudePhaseSignOperation {
    Magnitude,
    Phase,
    Sign,
}

impl MagnitudePhaseSignOperation {
    pub(super) fn reject_complex_integer(self, value: &Value) -> crate::BuiltinResult<()> {
        if !crate::builtins::common::validation::is_typed_complex_integer(value) {
            return Ok(());
        }
        let (descriptor, detail) = match self {
            Self::Magnitude => (
                &ABS_ERROR_INVALID_INPUT,
                "typed complex integer input is not supported",
            ),
            Self::Phase => (
                &ANGLE_ERROR_INVALID_INPUT,
                "expected single or double input",
            ),
            Self::Sign => (
                &SIGN_ERROR_INVALID_INPUT,
                "typed complex integer input is not supported",
            ),
        };
        Err(self.error(descriptor, detail))
    }

    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::Magnitude => "abs",
            Self::Phase => "angle",
            Self::Sign => "sign",
        }
    }

    pub(super) fn error(
        self,
        error: &'static BuiltinErrorDescriptor,
        detail: impl std::fmt::Display,
    ) -> RuntimeError {
        let mut builder =
            build_runtime_error(format!("{}: {detail}", error.message)).with_builtin(self.name());
        if let Some(identifier) = error.identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.build()
    }
}
