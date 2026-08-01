use super::{RegistryArtifact, RegistryClientError};
use futures::StreamExt;
use reqwest::header::AUTHORIZATION;
use sha2::{Digest as _, Sha256};
use url::Url;

pub(super) async fn download_artifact(
    http: &reqwest::Client,
    index: &Url,
    token: Option<&str>,
    artifact: &RegistryArtifact,
    transfer_limit: u64,
) -> Result<Vec<u8>, RegistryClientError> {
    if artifact.byte_len > transfer_limit {
        return Err(RegistryClientError::TooLarge {
            limit: transfer_limit,
        });
    }
    let url = artifact_url(index, &artifact.download_url)?;
    let mut request = http.get(url);
    if let Some(token) = token {
        request = request.header(AUTHORIZATION, format!("Bearer {token}"));
    }
    let response = request
        .send()
        .await
        .map_err(|_| RegistryClientError::Unavailable)?;
    match response.status() {
        reqwest::StatusCode::UNAUTHORIZED => return Err(RegistryClientError::Unauthorized),
        reqwest::StatusCode::FORBIDDEN => return Err(RegistryClientError::Forbidden),
        reqwest::StatusCode::NOT_FOUND | reqwest::StatusCode::GONE => {
            return Err(RegistryClientError::NotFound);
        }
        reqwest::StatusCode::TOO_MANY_REQUESTS => {
            return Err(RegistryClientError::RateLimited);
        }
        status if !status.is_success() => return Err(RegistryClientError::Unavailable),
        _ => {}
    }
    if response
        .content_length()
        .is_some_and(|length| length != artifact.byte_len || length > transfer_limit)
    {
        return Err(RegistryClientError::LengthMismatch);
    }
    let capacity =
        usize::try_from(artifact.byte_len).map_err(|_| RegistryClientError::TooLarge {
            limit: transfer_limit,
        })?;
    let mut bytes = Vec::with_capacity(capacity);
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|_| RegistryClientError::Unavailable)?;
        let next_len = bytes.len().saturating_add(chunk.len()) as u64;
        if next_len > transfer_limit || next_len > artifact.byte_len {
            return Err(RegistryClientError::TooLarge {
                limit: transfer_limit,
            });
        }
        bytes.extend_from_slice(&chunk);
    }
    if bytes.len() as u64 != artifact.byte_len {
        return Err(RegistryClientError::LengthMismatch);
    }
    let digest = format!("sha256:{:x}", Sha256::digest(&bytes));
    if digest != artifact.digest {
        return Err(RegistryClientError::DigestMismatch);
    }
    Ok(bytes)
}

fn artifact_url(index: &Url, value: &str) -> Result<Url, RegistryClientError> {
    if !value.starts_with('/') || value.starts_with("//") {
        return Err(RegistryClientError::UnsafeArtifactUrl);
    }
    let url = index
        .join(value)
        .map_err(|_| RegistryClientError::UnsafeArtifactUrl)?;
    if url.scheme() != index.scheme()
        || url.host_str() != index.host_str()
        || url.port_or_known_default() != index.port_or_known_default()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
    {
        return Err(RegistryClientError::UnsafeArtifactUrl);
    }
    Ok(url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn artifact_urls_are_relative_and_same_origin() {
        let index = Url::parse("https://api.runmat.test/base").unwrap();
        assert_eq!(
            artifact_url(
                &index,
                "/v1/packages/registry/releases/rel_123/artifact?signature=s"
            )
            .unwrap()
            .host_str(),
            Some("api.runmat.test")
        );
        for value in [
            "https://evil.test/artifact",
            "//evil.test/artifact",
            "artifact",
            "/artifact#fragment",
        ] {
            assert!(artifact_url(&index, value).is_err());
        }
    }
}
