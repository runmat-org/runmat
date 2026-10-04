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

#[derive(Debug)]
pub struct TcpTransportError {
    kind: TcpTransportErrorKind,
    message: String,
    source: Option<Box<dyn Error + Send + Sync>>,
}

impl TcpTransportError {
    pub fn new(kind: TcpTransportErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            source: None,
        }
    }

    pub fn with_source(
        kind: TcpTransportErrorKind,
        message: impl Into<String>,
        source: impl Error + Send + Sync + 'static,
    ) -> Self {
        Self {
            kind,
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }

    pub fn kind(&self) -> TcpTransportErrorKind {
        self.kind
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
