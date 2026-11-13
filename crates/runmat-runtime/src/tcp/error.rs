use runmat_async::runtime_error::{RuntimeErrorSource, RuntimeErrorSourceBound};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TcpTransportErrorKind {
    Io,
    Timeout,
    Closed,
    InvalidAuthority,
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TcpWriteProgress {
    NoBytesSent,
    PrefixSent(usize),
    Indeterminate,
}

#[derive(Debug)]
pub struct TcpTransportError {
    kind: TcpTransportErrorKind,
    message: String,
    source: Option<Box<RuntimeErrorSource>>,
    write_progress: Option<TcpWriteProgress>,
}

impl TcpTransportError {
    pub fn new(kind: TcpTransportErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            source: None,
            write_progress: None,
        }
    }

    pub fn with_source(
        kind: TcpTransportErrorKind,
        message: impl Into<String>,
        source: impl RuntimeErrorSourceBound,
    ) -> Self {
        Self {
            kind,
            message: message.into(),
            source: Some(Box::new(source)),
            write_progress: None,
        }
    }

    pub fn kind(&self) -> TcpTransportErrorKind {
        self.kind
    }

    pub fn with_write_progress(mut self, progress: TcpWriteProgress) -> Self {
        self.write_progress = Some(progress);
        self
    }

    pub fn write_progress(&self) -> Option<TcpWriteProgress> {
        self.write_progress
    }
}

impl fmt::Display for TcpTransportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for TcpTransportError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source.as_deref().map(|source| source as _)
    }
}
