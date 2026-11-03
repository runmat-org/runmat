use crate::endpoint::{ServerEndpoint, ServerEndpointError};
use reqwest::header::{HeaderMap, HeaderValue, InvalidHeaderValue, AUTHORIZATION};
use serde::de::DeserializeOwned;

#[derive(Debug, thiserror::Error)]
pub enum ServerLeaseTransportError {
    #[error("invalid RunMat Server endpoint")]
    Endpoint(#[from] ServerEndpointError),
    #[error("invalid RunMat Server credential header")]
    Credential(#[from] InvalidHeaderValue),
    #[error("RunMat Server request failed")]
    Http(#[from] reqwest::Error),
    #[error("RunMat Server returned HTTP {0}")]
    ServerStatus(u16),
    #[error("RunMat Server lease response exceeds 64 KiB")]
    BodyTooLarge,
    #[error("invalid RunMat Server lease response")]
    Decode(#[from] serde_json::Error),
}

#[derive(Clone)]
pub(crate) struct LeaseHttp {
    pub(crate) endpoint: ServerEndpoint,
    pub(crate) http: reqwest::Client,
}

impl LeaseHttp {
    pub(crate) fn new(server: &str, token: &str) -> Result<Self, ServerLeaseTransportError> {
        let endpoint = ServerEndpoint::parse(server)?;
        let mut headers = HeaderMap::new();
        let mut bearer = HeaderValue::from_str(&format!("Bearer {token}"))?;
        bearer.set_sensitive(true);
        headers.insert(AUTHORIZATION, bearer);
        let http = reqwest::Client::builder()
            .default_headers(headers)
            .redirect(reqwest::redirect::Policy::none())
            .timeout(std::time::Duration::from_secs(10))
            .build()?;
        Ok(Self { endpoint, http })
    }

    pub(crate) async fn response<T, E>(
        &self,
        request: reqwest::RequestBuilder,
        api_error: fn(u16, &[u8]) -> E,
    ) -> Result<T, E>
    where
        T: DeserializeOwned,
        E: From<ServerLeaseTransportError>,
    {
        let mut response = request
            .send()
            .await
            .map_err(ServerLeaseTransportError::from)?;
        let status = response.status();
        if status.is_server_error() {
            return Err(ServerLeaseTransportError::ServerStatus(status.as_u16()).into());
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(ServerLeaseTransportError::from)?
        {
            if bytes.len().saturating_add(chunk.len()) > 64 * 1024 {
                return Err(ServerLeaseTransportError::BodyTooLarge.into());
            }
            bytes.extend_from_slice(&chunk);
        }
        if !status.is_success() {
            return Err(api_error(status.as_u16(), &bytes));
        }
        serde_json::from_slice(&bytes)
            .map_err(ServerLeaseTransportError::from)
            .map_err(E::from)
    }
}
