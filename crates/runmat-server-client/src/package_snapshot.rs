use base64::Engine as _;
use reqwest::header::{AUTHORIZATION, ETAG, IF_NONE_MATCH};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use url::Url;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProjectSnapshotEntryKind {
    File,
    Directory,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ProjectSnapshotEntry {
    pub path: String,
    pub kind: ProjectSnapshotEntryKind,
    #[serde(default, skip_serializing_if = "Vec::is_empty", with = "base64_bytes")]
    pub bytes: Vec<u8>,
    #[serde(default)]
    pub executable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ProjectSnapshotInventory {
    pub project: String,
    pub snapshot: String,
    pub tree_digest: String,
    pub entries: Vec<ProjectSnapshotEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectSnapshotResponse {
    pub inventory: ProjectSnapshotInventory,
    pub etag: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProjectSnapshotOutcome {
    Snapshot(ProjectSnapshotResponse),
    NotModified,
}

#[derive(Debug, Error)]
pub enum ProjectSnapshotClientError {
    #[error("Server URL is invalid")]
    InvalidService,
    #[error("project snapshot request is unauthorized")]
    Unauthorized,
    #[error("project snapshot request is forbidden")]
    Forbidden,
    #[error("project snapshot was not found or was deleted")]
    NotFound,
    #[error("project snapshot exceeds transfer limits")]
    TooLarge,
    #[error("project snapshot is corrupt")]
    Corrupt,
    #[error("project snapshot service is unavailable")]
    Unavailable,
    #[error("project snapshot response is invalid: {0}")]
    InvalidResponse(String),
}

#[derive(Clone)]
pub struct ProjectSnapshotClient {
    service: Url,
    token: Option<String>,
    http: reqwest::Client,
}

impl ProjectSnapshotClient {
    pub fn new(service: &str, token: Option<String>) -> Result<Self, ProjectSnapshotClientError> {
        let mut service =
            Url::parse(service).map_err(|_| ProjectSnapshotClientError::InvalidService)?;
        if service.scheme() != "https"
            || !service.username().is_empty()
            || service.password().is_some()
            || service.query().is_some()
            || service.fragment().is_some()
        {
            return Err(ProjectSnapshotClientError::InvalidService);
        }
        service.set_query(None);
        service.set_fragment(None);
        Ok(Self {
            service,
            token,
            http: reqwest::Client::new(),
        })
    }

    pub async fn fetch(
        &self,
        project: &str,
        selector: &str,
        etag: Option<&str>,
    ) -> Result<ProjectSnapshotOutcome, ProjectSnapshotClientError> {
        let mut url = self.service.clone();
        {
            let mut segments = url
                .path_segments_mut()
                .map_err(|_| ProjectSnapshotClientError::InvalidService)?;
            segments.pop_if_empty();
            segments.extend(["v1", "packages", "projects", project, "snapshots", selector]);
        }
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
            .map_err(|_| ProjectSnapshotClientError::Unavailable)?;
        match response.status() {
            reqwest::StatusCode::NOT_MODIFIED => Ok(ProjectSnapshotOutcome::NotModified),
            reqwest::StatusCode::UNAUTHORIZED => Err(ProjectSnapshotClientError::Unauthorized),
            reqwest::StatusCode::FORBIDDEN => Err(ProjectSnapshotClientError::Forbidden),
            reqwest::StatusCode::NOT_FOUND | reqwest::StatusCode::GONE => {
                Err(ProjectSnapshotClientError::NotFound)
            }
            reqwest::StatusCode::PAYLOAD_TOO_LARGE => Err(ProjectSnapshotClientError::TooLarge),
            reqwest::StatusCode::UNPROCESSABLE_ENTITY => Err(ProjectSnapshotClientError::Corrupt),
            status if !status.is_success() => Err(ProjectSnapshotClientError::Unavailable),
            _ => {
                let etag = response
                    .headers()
                    .get(ETAG)
                    .and_then(|value| value.to_str().ok())
                    .map(str::to_string);
                let inventory =
                    response
                        .json::<ProjectSnapshotInventory>()
                        .await
                        .map_err(|error| {
                            ProjectSnapshotClientError::InvalidResponse(error.to_string())
                        })?;
                Ok(ProjectSnapshotOutcome::Snapshot(ProjectSnapshotResponse {
                    inventory,
                    etag,
                }))
            }
        }
    }
}

mod base64_bytes {
    use super::*;
    use serde::{Deserializer, Serializer};

    pub fn serialize<S>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&base64::engine::general_purpose::STANDARD.encode(bytes))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        base64::engine::general_purpose::STANDARD
            .decode(value)
            .map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_wire_decodes_file_bytes_and_defaults_optional_fields() {
        let inventory: ProjectSnapshotInventory = serde_json::from_str(
            r#"{
                "project":"proj_0123456789abcdef0123456789abcdef",
                "snapshot":"snap_0123456789abcdef0123456789abcdef",
                "treeDigest":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "entries":[
                    {"path":"src","kind":"directory"},
                    {"path":"src/helper.m","kind":"file","bytes":"YW5zd2VyID0gNDI7Cg=="}
                ]
            }"#,
        )
        .unwrap();
        assert_eq!(inventory.entries[0].bytes, Vec::<u8>::new());
        assert!(!inventory.entries[0].executable);
        assert_eq!(inventory.entries[1].bytes, b"answer = 42;\n");
    }

    #[test]
    fn client_rejects_insecure_or_secret_bearing_services() {
        for service in [
            "http://api.runmat.test",
            "https://user@api.runmat.test",
            "https://api.runmat.test?token=secret",
            "https://api.runmat.test/#secret",
        ] {
            assert!(ProjectSnapshotClient::new(service, None).is_err());
        }
    }
}
