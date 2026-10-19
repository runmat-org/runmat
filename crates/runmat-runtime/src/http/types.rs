use std::time::Duration;

use thiserror::Error;
use url::Url;

use crate::runtime_error::RuntimeErrorSourceBound;

#[cfg(not(target_arch = "wasm32"))]
type HttpErrorSource = Box<dyn std::error::Error + Send + Sync + 'static>;
#[cfg(target_arch = "wasm32")]
type HttpErrorSource = Box<dyn std::error::Error + 'static>;

#[derive(Clone, Copy, Debug)]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Patch,
    Delete,
}

impl HttpMethod {
    #[cfg(target_arch = "wasm32")]
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Get => "GET",
            Self::Post => "POST",
            Self::Put => "PUT",
            Self::Patch => "PATCH",
            Self::Delete => "DELETE",
        }
    }
}

#[derive(Clone, Debug)]
pub struct HttpRequest {
    pub url: Url,
    pub method: HttpMethod,
    pub headers: Vec<(String, String)>,
    pub body: Option<Vec<u8>>,
    pub timeout: Duration,
    pub user_agent: String,
}

#[derive(Clone, Debug)]
pub struct HttpResponse {
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

#[derive(Debug)]
pub enum TransportErrorKind {
    Timeout,
    Connect,
    Status(u16),
    InvalidHeader(String),
    UnsupportedStreaming,
    BodyLimitExceeded(u64),
    Other,
}

#[derive(Debug, Error)]
#[error("{message}")]
pub struct TransportError {
    pub kind: TransportErrorKind,
    pub url: String,
    pub detail: String,
    message: String,
    #[source]
    source: Option<HttpErrorSource>,
}

impl TransportError {
    pub fn unsupported_streaming(url: &Url) -> Self {
        Self::new(
            TransportErrorKind::UnsupportedStreaming,
            url,
            "host HTTP service does not provide bounded response streaming",
        )
    }

    /// The host could not start its reusable HTTP service. No request was sent.
    pub fn host_service_unavailable(url: &Url, source: impl RuntimeErrorSourceBound) -> Self {
        Self::new(TransportErrorKind::Other, url, source.to_string()).with_source(source)
    }

    pub(super) fn new(kind: TransportErrorKind, url: &Url, detail: impl Into<String>) -> Self {
        let url = url.to_string();
        let detail = detail.into();
        let message = match &kind {
            TransportErrorKind::Timeout => format!("request to {url} timed out"),
            TransportErrorKind::Connect => format!("unable to connect to {url}: {detail}"),
            TransportErrorKind::Status(code) => {
                format!("request to {url} failed with HTTP status {code}")
            }
            TransportErrorKind::InvalidHeader(name) => {
                format!("invalid header '{name}': {detail}")
            }
            TransportErrorKind::UnsupportedStreaming => {
                format!("streaming HTTP is unavailable for {url}: {detail}")
            }
            TransportErrorKind::BodyLimitExceeded(limit) => {
                format!("response from {url} exceeds the {limit}-byte limit")
            }
            TransportErrorKind::Other => format!("failed to contact {url}: {detail}"),
        };
        Self {
            kind,
            url,
            detail,
            message,
            source: None,
        }
    }

    pub(super) fn with_source<E: RuntimeErrorSourceBound>(mut self, source: E) -> Self {
        self.source = Some(Box::new(source));
        self
    }

    pub fn message_with_prefix(&self, prefix: &str) -> String {
        format!("{prefix}: {}", self.message)
    }
}
