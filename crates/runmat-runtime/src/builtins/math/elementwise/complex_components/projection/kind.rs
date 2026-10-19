use runmat_builtins::{
    BuiltinErrorDescriptor, IMAG_ERROR_INTERNAL, IMAG_ERROR_INVALID_INPUT, REAL_ERROR_INTERNAL,
    REAL_ERROR_INVALID_INPUT,
};

use crate::{build_runtime_error, RuntimeError};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::builtins::math::elementwise::complex_components) enum ProjectionKind {
    Real,
    Imaginary,
}

impl ProjectionKind {
    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::Real => "real",
            Self::Imaginary => "imag",
        }
    }

    pub(super) const fn provider_hook(self) -> &'static str {
        match self {
            Self::Real => "unary_real",
            Self::Imaginary => "unary_imag",
        }
    }

    pub(super) fn invalid(self, detail: impl AsRef<str>) -> RuntimeError {
        let descriptor = match self {
            Self::Real => &REAL_ERROR_INVALID_INPUT,
            Self::Imaginary => &IMAG_ERROR_INVALID_INPUT,
        };
        self.error(descriptor, detail)
    }

    pub(super) fn internal(self, detail: impl AsRef<str>) -> RuntimeError {
        let descriptor = match self {
            Self::Real => &REAL_ERROR_INTERNAL,
            Self::Imaginary => &IMAG_ERROR_INTERNAL,
        };
        self.error(descriptor, detail)
    }

    fn error(
        self,
        descriptor: &'static BuiltinErrorDescriptor,
        detail: impl AsRef<str>,
    ) -> RuntimeError {
        let mut builder =
            build_runtime_error(format!("{}: {}", descriptor.message, detail.as_ref()))
                .with_builtin(self.name());
        if let Some(identifier) = descriptor.identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.build()
    }
}
