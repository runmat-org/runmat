//! The shared MAT codec retains the identity of its owning public operation.
//! Error classes are closed and each caller supplies only this typed owner.

use crate::{build_runtime_error, RuntimeError};
use runmat_builtins::{
    BuiltinErrorDescriptor, MATFILE_ERROR_IO, MATFILE_ERROR_SELECTION, MATFILE_ERROR_UNSUPPORTED,
    SAVE_ERROR_IO, SAVE_ERROR_SELECTION, SAVE_ERROR_UNSUPPORTED,
};

#[derive(Clone, Copy)]
pub(crate) enum MatEncodingOwner {
    Save,
    MatFile,
    BinaryRecord {
        identity: &'static runmat_builtins::BuiltinCatalogIdentity,
        error: &'static BuiltinErrorDescriptor,
    },
}

#[derive(Clone, Copy)]
pub(super) enum MatEncodingFailure {
    Io,
    Selection,
    Unsupported,
}

impl MatEncodingOwner {
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Save => "save",
            Self::MatFile => "matfile",
            Self::BinaryRecord { identity, .. } => identity.name,
        }
    }

    fn descriptor(self, failure: MatEncodingFailure) -> &'static BuiltinErrorDescriptor {
        match (self, failure) {
            (Self::BinaryRecord { error, .. }, _) => error,
            (Self::Save, MatEncodingFailure::Io) => &SAVE_ERROR_IO,
            (Self::Save, MatEncodingFailure::Selection) => &SAVE_ERROR_SELECTION,
            (Self::Save, MatEncodingFailure::Unsupported) => &SAVE_ERROR_UNSUPPORTED,
            (Self::MatFile, MatEncodingFailure::Io) => &MATFILE_ERROR_IO,
            (Self::MatFile, MatEncodingFailure::Selection) => &MATFILE_ERROR_SELECTION,
            (Self::MatFile, MatEncodingFailure::Unsupported) => &MATFILE_ERROR_UNSUPPORTED,
        }
    }

    pub(super) fn error(
        self,
        failure: MatEncodingFailure,
        message: impl AsRef<str>,
    ) -> RuntimeError {
        let descriptor = self.descriptor(failure);
        let mut builder = build_runtime_error(format!("{}: {}", self.name(), message.as_ref()))
            .with_builtin(self.name());
        if let Some(identifier) = descriptor.identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.build()
    }

    pub(super) fn with_source(
        self,
        failure: MatEncodingFailure,
        message: impl AsRef<str>,
        source: impl crate::runtime_error::RuntimeErrorSourceBound,
    ) -> RuntimeError {
        let descriptor = self.descriptor(failure);
        let mut builder = build_runtime_error(format!("{}: {}", self.name(), message.as_ref()))
            .with_builtin(self.name())
            .with_source(source);
        if let Some(identifier) = descriptor.identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.build()
    }
}
