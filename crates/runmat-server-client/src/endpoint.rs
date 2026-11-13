//! Canonical account/node endpoint identity, before credentials are attached.
use std::net::IpAddr;

use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum ServerEndpointError {
    #[error("invalid Server endpoint URL: {0}")]
    Url(#[from] url::ParseError),
    #[error("Server endpoint must use HTTPS (HTTP only on loopback), without URL credentials, query or fragment")]
    Policy,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ServerEndpoint(String);

impl ServerEndpoint {
    pub fn parse(value: &str) -> Result<Self, ServerEndpointError> {
        let url = url::Url::parse(value)?;
        let host = url.host_str().ok_or(ServerEndpointError::Policy)?;
        let loopback = host == "localhost"
            || host
                .trim_matches(['[', ']'])
                .parse::<IpAddr>()
                .is_ok_and(|ip| ip.is_loopback());
        if !(url.scheme() == "https" || (url.scheme() == "http" && loopback))
            || !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
        {
            return Err(ServerEndpointError::Policy);
        }
        Ok(Self(url.as_str().trim_end_matches('/').to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for ServerEndpoint {
    type Error = ServerEndpointError;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::parse(&value)
    }
}

impl From<ServerEndpoint> for String {
    fn from(value: ServerEndpoint) -> Self {
        value.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_identity_is_canonical_and_loopback_is_a_host_property() {
        assert_eq!(
            ServerEndpoint::parse("https://EXAMPLE.COM:443/")
                .unwrap()
                .as_str(),
            "https://example.com"
        );
        for value in [
            "http://localhost:3000",
            "http://127.0.0.1",
            "http://[::1]:3000",
        ] {
            assert!(ServerEndpoint::parse(value).is_ok());
        }
        for value in [
            "http://localhost.example",
            "http://127.0.0.1.example",
            "https://user:secret@example.com",
            "https://example.com?key=value",
            "https://example.com/#fragment",
            "file:///tmp/server",
        ] {
            assert!(ServerEndpoint::parse(value).is_err());
        }
    }
}
