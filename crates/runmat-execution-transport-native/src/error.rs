#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum TransportError {
    #[error("transport frame is malformed: {0}")]
    MalformedFrame(String),
    #[error("transport frame exceeds the configured bound")]
    FrameTooLarge,
    #[error("transport frame was replayed or is outside the receive window")]
    Replay,
    #[error("transport flow-control credit is exhausted")]
    FlowControl,
    #[error("transport sequence or byte count overflowed")]
    Overflow,
    #[error("object transfer does not match its admitted identity")]
    Integrity,
    #[error("application frame encryption failed: {0}")]
    Encryption(String),
    #[error("control-plane transport is unavailable: {0}")]
    Unavailable(String),
    #[error("control-plane resource is not ready")]
    NotReady,
    #[error("credential or lease is stale")]
    StaleAuthority,
}

pub type TransportResult<T> = Result<T, TransportError>;
