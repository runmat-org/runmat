use runmat_builtins::BuiltinErrorDescriptor;

use crate::{build_runtime_error, RuntimeError};

#[derive(Clone, Copy)]
pub(super) enum FloatingConversion {
    Double,
    Single,
}

impl FloatingConversion {
    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::Double => "double",
            Self::Single => "single",
        }
    }

    pub(super) fn error_with_detail(
        self,
        error: &'static BuiltinErrorDescriptor,
        detail: impl std::fmt::Display,
    ) -> RuntimeError {
        self.error_with_message(format!("{}: {}", error.message, detail), error)
    }

    fn error_with_message(
        self,
        message: impl Into<String>,
        error: &'static BuiltinErrorDescriptor,
    ) -> RuntimeError {
        let mut builder = build_runtime_error(message).with_builtin(self.name());
        if let Some(identifier) = error.identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.build()
    }

    pub(super) fn conversion_error(
        self,
        error: &'static BuiltinErrorDescriptor,
        type_name: &str,
    ) -> RuntimeError {
        self.error_with_detail(
            error,
            format!(
                "conversion to {} from {type_name} is not possible",
                self.name()
            ),
        )
    }
}
