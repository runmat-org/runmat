use runmat_builtins::BuiltinErrorDescriptor;

use crate::{build_runtime_error, RuntimeError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MagnitudePhaseSignOperation {
    Magnitude,
    Phase,
    Sign,
}

impl MagnitudePhaseSignOperation {
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
