use std::time::Duration;

use thiserror::Error;
use url::Url;

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
    Other,
}

#[derive(Debug, Error)]
#[error("{message}")]
pub struct TransportError {
    pub kind: TransportErrorKind,
    pub url: String,
    pub detail: String,
    message: String,
}

impl TransportError {
    /// The host could not start its reusable HTTP service. No request was sent.
    pub fn host_service_unavailable(url: &Url, detail: impl Into<String>) -> Self {
        Self::new(TransportErrorKind::Other, url, detail)
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
            TransportErrorKind::Other => format!("failed to contact {url}: {detail}"),
        };
        Self {
            kind,
            url,
            detail,
            message,
        }
    }

    pub fn message_with_prefix(&self, prefix: &str) -> String {
        format!("{prefix}: {}", self.message)
    }
}
