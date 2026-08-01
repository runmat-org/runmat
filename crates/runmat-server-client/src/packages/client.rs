use super::artifacts::download_artifact;
use super::{
    RegistryArtifact, RegistryCandidateList, RegistryCandidateOutcome, RegistryCandidateResponse,
    RegistryClientError, RegistryReleaseMetadata, RegistryReleaseOutcome, RegistryReleaseResponse,
};
use reqwest::header::{AUTHORIZATION, ETAG, IF_NONE_MATCH};
use url::Url;

const METADATA_LIMIT: u64 = 4 * 1024 * 1024;

#[derive(Clone)]
pub struct RegistryClient {
    index: Url,
    token: Option<String>,
    http: reqwest::Client,
}

impl RegistryClient {
    pub fn new(index: &str, token: Option<String>) -> Result<Self, RegistryClientError> {
        let index = normalize_index(index)?;
        let http = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|_| RegistryClientError::Unavailable)?;
        Ok(Self { index, token, http })
    }

    pub async fn resolve(
        &self,
        namespace: &str,
        name: &str,
        requirement: &str,
        etag: Option<&str>,
    ) -> Result<RegistryReleaseOutcome, RegistryClientError> {
        let mut url = self.index.clone();
        {
            let mut segments = url
                .path_segments_mut()
                .map_err(|_| RegistryClientError::InvalidIndex)?;
            segments.pop_if_empty();
            segments.extend(["v1", "packages", "registry", namespace, name, "resolve"]);
        }
        url.query_pairs_mut()
            .append_pair("requirement", requirement);
        self.fetch_metadata(url, etag).await
    }

    pub async fn candidates(
        &self,
        namespace: &str,
        name: &str,
        etag: Option<&str>,
    ) -> Result<RegistryCandidateOutcome, RegistryClientError> {
        let mut url = self.index.clone();
        {
            let mut segments = url
                .path_segments_mut()
                .map_err(|_| RegistryClientError::InvalidIndex)?;
            segments.pop_if_empty();
            segments.extend(["v1", "packages", "registry", namespace, name, "candidates"]);
        }
        let response = self.send(url, etag).await?;
        match response {
            MetadataResponse::NotModified => Ok(RegistryCandidateOutcome::NotModified),
            MetadataResponse::Payload { bytes, etag } => {
                let list = serde_json::from_slice::<RegistryCandidateList>(&bytes)
                    .map_err(|error| RegistryClientError::InvalidResponse(error.to_string()))?;
                Ok(RegistryCandidateOutcome::Candidates(
                    RegistryCandidateResponse {
                        candidates: list.candidates,
                        etag,
                    },
                ))
            }
        }
    }

    pub async fn resolve_exact(
        &self,
        release_id: &str,
        etag: Option<&str>,
    ) -> Result<RegistryReleaseOutcome, RegistryClientError> {
        let mut url = self.index.clone();
        {
            let mut segments = url
                .path_segments_mut()
                .map_err(|_| RegistryClientError::InvalidIndex)?;
            segments.pop_if_empty();
            segments.extend(["v1", "packages", "registry", "releases", release_id]);
        }
        self.fetch_metadata(url, etag).await
    }

    pub async fn download_artifact(
        &self,
        artifact: &RegistryArtifact,
        transfer_limit: u64,
    ) -> Result<Vec<u8>, RegistryClientError> {
        download_artifact(
            &self.http,
            &self.index,
            self.token.as_deref(),
            artifact,
            transfer_limit,
        )
        .await
    }

    async fn fetch_metadata(
        &self,
        url: Url,
        etag: Option<&str>,
    ) -> Result<RegistryReleaseOutcome, RegistryClientError> {
        match self.send(url, etag).await? {
            MetadataResponse::NotModified => Ok(RegistryReleaseOutcome::NotModified),
            MetadataResponse::Payload { bytes, etag } => {
                let metadata = serde_json::from_slice::<RegistryReleaseMetadata>(&bytes)
                    .map_err(|error| RegistryClientError::InvalidResponse(error.to_string()))?;
                Ok(RegistryReleaseOutcome::Release(Box::new(
                    RegistryReleaseResponse { metadata, etag },
                )))
            }
        }
    }

    async fn send(
        &self,
        url: Url,
        etag: Option<&str>,
    ) -> Result<MetadataResponse, RegistryClientError> {
        let mut request = self.http.get(url);
        if let Some(token) = &self.token {
            request = request.header(AUTHORIZATION, format!("Bearer {token}"));
        }
        if let Some(etag) = etag {
            request = request.header(IF_NONE_MATCH, etag);
        }
        let response = request
            .send()
            .await
            .map_err(|_| RegistryClientError::Unavailable)?;
        match response.status() {
            reqwest::StatusCode::NOT_MODIFIED => Ok(MetadataResponse::NotModified),
            reqwest::StatusCode::UNAUTHORIZED => Err(RegistryClientError::Unauthorized),
            reqwest::StatusCode::FORBIDDEN => Err(RegistryClientError::Forbidden),
            reqwest::StatusCode::NOT_FOUND | reqwest::StatusCode::GONE => {
                Err(RegistryClientError::NotFound)
            }
            reqwest::StatusCode::TOO_MANY_REQUESTS => Err(RegistryClientError::RateLimited),
            status if !status.is_success() => Err(RegistryClientError::Unavailable),
            _ => {
                if response
                    .content_length()
                    .is_some_and(|length| length > METADATA_LIMIT)
                {
                    return Err(RegistryClientError::TooLarge {
                        limit: METADATA_LIMIT,
                    });
                }
                let etag = response
                    .headers()
                    .get(ETAG)
                    .and_then(|value| value.to_str().ok())
                    .map(str::to_string);
                let bytes = response
                    .bytes()
                    .await
                    .map_err(|_| RegistryClientError::Unavailable)?;
                if bytes.len() as u64 > METADATA_LIMIT {
                    return Err(RegistryClientError::TooLarge {
                        limit: METADATA_LIMIT,
                    });
                }
                Ok(MetadataResponse::Payload {
                    bytes: bytes.to_vec(),
                    etag,
                })
            }
        }
    }
}

enum MetadataResponse {
    NotModified,
    Payload {
        bytes: Vec<u8>,
        etag: Option<String>,
    },
}

fn normalize_index(value: &str) -> Result<Url, RegistryClientError> {
    let mut url = Url::parse(value).map_err(|_| RegistryClientError::InvalidIndex)?;
    if url.scheme() != "https"
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(RegistryClientError::InvalidIndex);
    }
    url.set_query(None);
    url.set_fragment(None);
    Ok(url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_client_rejects_insecure_or_secret_bearing_indexes() {
        for index in [
            "http://packages.runmat.test",
            "https://user@packages.runmat.test",
            "https://packages.runmat.test?token=secret",
            "https://packages.runmat.test/#secret",
        ] {
            assert!(RegistryClient::new(index, None).is_err());
        }
    }
}
