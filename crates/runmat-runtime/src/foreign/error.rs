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
    UnsupportedOnWasm,
    CallbackFailed,
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
            Self::UnsupportedOnWasm => "RunMat:Foreign:UnsupportedOnWasm",
            Self::CallbackFailed => "RunMat:Foreign:CallbackFailed",
        }
    }
}

pub fn foreign_error(kind: ForeignErrorKind, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin("foreign")
        .with_identifier(kind.identifier())
        .build()
}
