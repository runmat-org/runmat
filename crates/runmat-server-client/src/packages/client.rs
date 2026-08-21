use super::artifacts::download_artifact;
use super::{
    AttachKeyEnvelopesRequest, FinalizePublicationResponse, PublicationStatusResponse,
    RegistryArtifact, RegistryCandidateList, RegistryCandidateOutcome, RegistryCandidateResponse,
    RegistryClientError, RegistryRecipientKey, RegistryRecipientKeyList, RegistryReleaseMetadata,
    RegistryReleaseOutcome, RegistryReleaseResponse, StagePublicationRequest,
    StagePublicationResponse,
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

    pub async fn recipient_keys(
        &self,
        namespace: &str,
        name: &str,
    ) -> Result<Vec<RegistryRecipientKey>, RegistryClientError> {
        let url = self.recipient_keys_url(namespace, name)?;
        let response = self.authorized(self.http.get(url)).send().await;
        let bytes = response_bytes(response).await?;
        serde_json::from_slice::<RegistryRecipientKeyList>(&bytes)
            .map(|value| value.keys)
            .map_err(|error| RegistryClientError::InvalidResponse(error.to_string()))
    }

    pub async fn register_recipient_key(
        &self,
        namespace: &str,
        name: &str,
        public_key: &str,
    ) -> Result<RegistryRecipientKey, RegistryClientError> {
        let url = self.recipient_keys_url(namespace, name)?;
        let response = self
            .authorized(self.http.post(url))
            .json(&serde_json::json!({"publicKey": public_key}))
            .send()
            .await;
        let bytes = response_bytes(response).await?;
        serde_json::from_slice(&bytes)
            .map_err(|error| RegistryClientError::InvalidResponse(error.to_string()))
    }

    pub async fn revoke_recipient_key(
        &self,
        namespace: &str,
        name: &str,
        key_id: &str,
    ) -> Result<RegistryRecipientKey, RegistryClientError> {
        let mut url = self.recipient_keys_url(namespace, name)?;
        url.path_segments_mut()
            .map_err(|_| RegistryClientError::InvalidIndex)?
            .extend([key_id, "revoke"]);
        let response = self.authorized(self.http.post(url)).send().await;
        let bytes = response_bytes(response).await?;
        serde_json::from_slice(&bytes)
            .map_err(|error| RegistryClientError::InvalidResponse(error.to_string()))
    }

    pub async fn publication_recipient_keys(
        &self,
        org_id: &str,
        package_id: &str,
    ) -> Result<Vec<RegistryRecipientKey>, RegistryClientError> {
        let url = self.management_url(org_id, package_id, &["recipient-keys"])?;
        self.send_json(self.authorized(self.http.get(url)))
            .await
            .map(|value: RegistryRecipientKeyList| value.keys)
    }

    pub async fn stage_publication(
        &self,
        org_id: &str,
        package_id: &str,
        request: &StagePublicationRequest,
    ) -> Result<StagePublicationResponse, RegistryClientError> {
        let url = self.management_url(org_id, package_id, &["publications"])?;
        self.send_json(self.authorized(self.http.post(url)).json(request))
            .await
    }

    pub async fn upload_publication_artifact(
        &self,
        upload_url: &str,
        media_type: &str,
        bytes: Vec<u8>,
    ) -> Result<(), RegistryClientError> {
        let url = self.resolve_upload_url(upload_url)?;
        let response = self
            .http
            .put(url)
            .header(reqwest::header::CONTENT_TYPE, media_type)
            .header(reqwest::header::CONTENT_LENGTH, bytes.len())
            .body(bytes)
            .send()
            .await
            .map_err(|_| RegistryClientError::Unavailable)?;
        if response.status().is_success() {
            Ok(())
        } else {
            Err(RegistryClientError::Unavailable)
        }
    }

    pub async fn verify_publication(
        &self,
        org_id: &str,
        package_id: &str,
        publication_id: &str,
    ) -> Result<PublicationStatusResponse, RegistryClientError> {
        self.publication_action(org_id, package_id, publication_id, "verify", None)
            .await
    }

    pub async fn attach_publication_key_envelopes(
        &self,
        org_id: &str,
        package_id: &str,
        publication_id: &str,
        request: &AttachKeyEnvelopesRequest,
    ) -> Result<(), RegistryClientError> {
        let url = self.management_url(
            org_id,
            package_id,
            &["publications", publication_id, "key-envelopes"],
        )?;
        let response = self
            .authorized(self.http.post(url))
            .json(request)
            .send()
            .await;
        response_bytes(response).await.map(|_| ())
    }

    pub async fn approve_publication(
        &self,
        org_id: &str,
        package_id: &str,
        publication_id: &str,
    ) -> Result<PublicationStatusResponse, RegistryClientError> {
        self.publication_action(
            org_id,
            package_id,
            publication_id,
            "approve",
            Some(serde_json::json!({"approve": true})),
        )
        .await
    }

    pub async fn finalize_publication(
        &self,
        org_id: &str,
        package_id: &str,
        publication_id: &str,
    ) -> Result<FinalizePublicationResponse, RegistryClientError> {
        let url = self.management_url(
            org_id,
            package_id,
            &["publications", publication_id, "finalize"],
        )?;
        self.send_json(self.authorized(self.http.post(url))).await
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

    fn recipient_keys_url(&self, namespace: &str, name: &str) -> Result<Url, RegistryClientError> {
        let mut url = self.index.clone();
        url.path_segments_mut()
            .map_err(|_| RegistryClientError::InvalidIndex)?
            .pop_if_empty()
            .extend([
                "v1",
                "packages",
                "registry",
                namespace,
                name,
                "recipient-keys",
            ]);
        Ok(url)
    }

    fn management_url(
        &self,
        org_id: &str,
        package_id: &str,
        suffix: &[&str],
    ) -> Result<Url, RegistryClientError> {
        let mut url = self.index.clone();
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| RegistryClientError::InvalidIndex)?;
        segments
            .pop_if_empty()
            .extend(["v1", "orgs", org_id, "packages", "registry", package_id])
            .extend(suffix.iter().copied());
        drop(segments);
        Ok(url)
    }

    fn resolve_upload_url(&self, value: &str) -> Result<Url, RegistryClientError> {
        let url = self
            .index
            .join(value)
            .map_err(|_| RegistryClientError::InvalidResponse("upload URL is invalid".into()))?;
        if url.scheme() != "https"
            || !url.username().is_empty()
            || url.password().is_some()
            || url.fragment().is_some()
        {
            return Err(RegistryClientError::InvalidResponse(
                "upload URL is unsafe".to_string(),
            ));
        }
        Ok(url)
    }

    async fn publication_action(
        &self,
        org_id: &str,
        package_id: &str,
        publication_id: &str,
        action: &str,
        body: Option<serde_json::Value>,
    ) -> Result<PublicationStatusResponse, RegistryClientError> {
        let url = self.management_url(
            org_id,
            package_id,
            &["publications", publication_id, action],
        )?;
        let request = self.authorized(self.http.post(url));
        self.send_json(match body {
            Some(body) => request.json(&body),
            None => request,
        })
        .await
    }

    async fn send_json<T: serde::de::DeserializeOwned>(
        &self,
        request: reqwest::RequestBuilder,
    ) -> Result<T, RegistryClientError> {
        let bytes = response_bytes(request.send().await).await?;
        serde_json::from_slice(&bytes)
            .map_err(|error| RegistryClientError::InvalidResponse(error.to_string()))
    }

    fn authorized(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.token {
            Some(token) => request.header(AUTHORIZATION, format!("Bearer {token}")),
            None => request,
        }
    }
}

async fn response_bytes(
    response: Result<reqwest::Response, reqwest::Error>,
) -> Result<Vec<u8>, RegistryClientError> {
    let response = response.map_err(|_| RegistryClientError::Unavailable)?;
    match response.status() {
        reqwest::StatusCode::UNAUTHORIZED => return Err(RegistryClientError::Unauthorized),
        reqwest::StatusCode::FORBIDDEN => return Err(RegistryClientError::Forbidden),
        reqwest::StatusCode::NOT_FOUND | reqwest::StatusCode::GONE => {
            return Err(RegistryClientError::NotFound)
        }
        reqwest::StatusCode::TOO_MANY_REQUESTS => return Err(RegistryClientError::RateLimited),
        status if !status.is_success() => return Err(RegistryClientError::Unavailable),
        _ => {}
    }
    if response
        .content_length()
        .is_some_and(|length| length > METADATA_LIMIT)
    {
        return Err(RegistryClientError::TooLarge {
            limit: METADATA_LIMIT,
        });
    }
    let bytes = response
        .bytes()
        .await
        .map_err(|_| RegistryClientError::Unavailable)?;
    if bytes.len() as u64 > METADATA_LIMIT {
        return Err(RegistryClientError::TooLarge {
            limit: METADATA_LIMIT,
        });
    }
    Ok(bytes.to_vec())
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
