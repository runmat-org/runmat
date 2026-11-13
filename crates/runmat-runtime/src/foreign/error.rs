use crate::runtime_error::RuntimeErrorSourceBound;
use crate::{build_runtime_error, RuntimeError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForeignErrorKind {
    InvalidManifest,
    AdapterUnavailable,
    CapabilityUnavailable,
    HostAlreadyRegistered,
    HostUnavailable,
    StaleHandle,
    HandleMetadataMismatch,
    AffinityViolation,
    ExecutionStackViolation,
    UnsupportedOnWasm,
    CallbackFailed,
    InvalidCall,
    LoadFailed,
    InvocationFailed,
    ConversionFailed,
}

impl ForeignErrorKind {
    pub const fn identifier(self) -> &'static str {
        match self {
            Self::InvalidManifest => "RunMat:Foreign:InvalidManifest",
            Self::AdapterUnavailable => "RunMat:Foreign:AdapterUnavailable",
            Self::CapabilityUnavailable => "RunMat:Foreign:CapabilityUnavailable",
            Self::HostAlreadyRegistered => "RunMat:Foreign:HostAlreadyRegistered",
            Self::HostUnavailable => "RunMat:Foreign:HostUnavailable",
            Self::StaleHandle => "RunMat:Foreign:StaleHandle",
            Self::HandleMetadataMismatch => "RunMat:Foreign:HandleMetadataMismatch",
            Self::AffinityViolation => "RunMat:Foreign:AffinityViolation",
            Self::ExecutionStackViolation => "RunMat:Foreign:ExecutionStackViolation",
            Self::UnsupportedOnWasm => "RunMat:Foreign:UnsupportedOnWasm",
            Self::CallbackFailed => "RunMat:Foreign:CallbackFailed",
            Self::InvalidCall => "RunMat:Foreign:InvalidCall",
            Self::LoadFailed => "RunMat:Foreign:LoadFailed",
            Self::InvocationFailed => "RunMat:Foreign:InvocationFailed",
            Self::ConversionFailed => "RunMat:Foreign:ConversionFailed",
        }
    }
}

pub fn foreign_error(kind: ForeignErrorKind, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin("foreign")
        .with_identifier(kind.identifier())
        .build()
}

/// Retain the native/host cause under the same public foreign identifier.
pub fn foreign_error_with_source(
    kind: ForeignErrorKind,
    source: impl RuntimeErrorSourceBound,
) -> RuntimeError {
    build_runtime_error(source.to_string())
        .with_builtin("foreign")
        .with_identifier(kind.identifier())
        .with_source(source)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn foreign_identifier_keeps_original_host_cause() {
        let error = foreign_error_with_source(
            ForeignErrorKind::LoadFailed,
            std::io::Error::new(std::io::ErrorKind::NotFound, "missing native library"),
        );
        assert_eq!(error.identifier(), Some("RunMat:Foreign:LoadFailed"));
        assert!(std::error::Error::source(&error)
            .and_then(|source| source.downcast_ref::<std::io::Error>())
            .is_some());
    }
}
