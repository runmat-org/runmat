use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Artifact {
    pub id: String,
    pub name: String,
    pub media_type: String,
    pub byte_len: u64,
    pub content_digest: String,
    pub location: ArtifactLocation,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum ArtifactLocation {
    Inline(String),
    StoreKey(String),
    DownloadHandle(String),
}
