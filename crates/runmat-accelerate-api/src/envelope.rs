//! Typed input-domain rejection from resident envelope validation.
//! This is neither an unavailable hook nor a transport failure.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvelopeSignalDomainError {
    NonFinite,
}

impl std::fmt::Display for EnvelopeSignalDomainError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFinite => write!(formatter, "envelope: signal must contain finite values"),
        }
    }
}

impl std::error::Error for EnvelopeSignalDomainError {}
