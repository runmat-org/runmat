use super::{CanonicalPackageId, ContentDigest, PackageVersion, RegistryId};
use crate::IdentityError;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::path::Path;
use std::str::FromStr;
use url::Url;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct NormalizedRelativePath(String);

impl NormalizedRelativePath {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, IdentityError> {
        let path = path.as_ref();
        let Some(raw) = path.to_str() else {
            return Err(invalid_path(&path.to_string_lossy(), "must be valid UTF-8"));
        };
        let portable = raw.replace('\\', "/");
        if portable.is_empty() {
            return Ok(Self(".".to_string()));
        }
        if portable.starts_with('/') || portable.as_bytes().get(1).is_some_and(|byte| *byte == b':')
        {
            return Err(invalid_path(&portable, "must be relative"));
        }
        let mut segments = Vec::new();
        for segment in portable.split('/') {
            match segment {
                "" | "." => {}
                ".." => {
                    return Err(invalid_path(&portable, "must not contain parent traversal"));
                }
                _ if segment
                    .chars()
                    .any(|character| character.is_control() || character == ':') =>
                {
                    return Err(invalid_path(&portable, "contains a non-portable character"));
                }
                _ => segments.push(segment),
            }
        }
        Ok(Self(if segments.is_empty() {
            ".".to_string()
        } else {
            segments.join("/")
        }))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Display for NormalizedRelativePath {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl FromStr for NormalizedRelativePath {
    type Err = IdentityError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::new(Path::new(value))
    }
}

impl TryFrom<String> for NormalizedRelativePath {
    type Error = IdentityError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        value.parse()
    }
}

impl From<NormalizedRelativePath> for String {
    fn from(value: NormalizedRelativePath) -> Self {
        value.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PathSourceId {
    pub workspace_path: NormalizedRelativePath,
    pub manifest_digest: ContentDigest,
    pub tree_digest: ContentDigest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GitObjectAlgorithm {
    Sha1,
    Sha256,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct GitCommitId {
    pub algorithm: GitObjectAlgorithm,
    pub hex: String,
}

impl FromStr for GitCommitId {
    type Err = IdentityError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let (algorithm, length) = match value.len() {
            40 => (GitObjectAlgorithm::Sha1, 40),
            64 => (GitObjectAlgorithm::Sha256, 64),
            _ => {
                return Err(IdentityError::InvalidGitObjectId {
                    value: value.to_string(),
                    reason: "must be a full 40-digit SHA-1 or 64-digit SHA-256 object ID",
                });
            }
        };
        if value.len() != length
            || value
                .bytes()
                .any(|byte| !byte.is_ascii_hexdigit() || byte.is_ascii_uppercase())
        {
            return Err(IdentityError::InvalidGitObjectId {
                value: value.to_string(),
                reason: "must use lowercase hexadecimal",
            });
        }
        Ok(Self {
            algorithm,
            hex: value.to_string(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct GitRepositoryUrl(String);

impl GitRepositoryUrl {
    pub fn new(value: &str) -> Result<Self, IdentityError> {
        normalize_git_repository(value).map(Self)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Display for GitRepositoryUrl {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl FromStr for GitRepositoryUrl {
    type Err = IdentityError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::new(value)
    }
}

impl TryFrom<String> for GitRepositoryUrl {
    type Error = IdentityError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(&value)
    }
}

impl From<GitRepositoryUrl> for String {
    fn from(value: GitRepositoryUrl) -> Self {
        value.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct GitSourceId {
    pub repository: GitRepositoryUrl,
    pub commit: GitCommitId,
    pub subdir: NormalizedRelativePath,
    pub tree_digest: ContentDigest,
}

impl GitSourceId {
    pub fn new(
        repository: &str,
        commit: GitCommitId,
        subdir: NormalizedRelativePath,
        tree_digest: ContentDigest,
    ) -> Result<Self, IdentityError> {
        Ok(Self {
            repository: GitRepositoryUrl::new(repository)?,
            commit,
            subdir,
            tree_digest,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ServerProjectSourceId {
    pub service: String,
    pub project: String,
    pub snapshot: String,
    pub tree_digest: ContentDigest,
}

impl ServerProjectSourceId {
    pub fn new(
        service: &str,
        project: impl Into<String>,
        snapshot: impl Into<String>,
        tree_digest: ContentDigest,
    ) -> Result<Self, IdentityError> {
        let service = normalize_service_origin(service)?;
        let project = project.into();
        let snapshot = snapshot.into();
        if project.trim().is_empty() || snapshot.trim().is_empty() {
            return Err(IdentityError::InvalidServerProjectSource {
                value: format!("{service}/{project}@{snapshot}"),
                reason: "project and snapshot IDs must be non-empty",
            });
        }
        Ok(Self {
            service,
            project,
            snapshot,
            tree_digest,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RegistrySourceId {
    pub registry: RegistryId,
    pub package: CanonicalPackageId,
    pub version: PackageVersion,
    pub release_digest: ContentDigest,
    pub artifact_digest: ContentDigest,
    pub tree_digest: ContentDigest,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", content = "source", rename_all = "kebab-case")]
pub enum SourceId {
    Path(PathSourceId),
    Git(GitSourceId),
    ServerProject(ServerProjectSourceId),
    Registry(RegistrySourceId),
}

impl SourceId {
    pub fn tree_digest(&self) -> &ContentDigest {
        match self {
            Self::Path(source) => &source.tree_digest,
            Self::Git(source) => &source.tree_digest,
            Self::ServerProject(source) => &source.tree_digest,
            Self::Registry(source) => &source.tree_digest,
        }
    }

    pub fn validate(&self) -> Result<(), IdentityError> {
        match self {
            Self::Path(_) => Ok(()),
            Self::Git(source) => {
                let parsed: GitCommitId = source.commit.hex.parse()?;
                if parsed.algorithm != source.commit.algorithm {
                    return Err(IdentityError::InvalidGitObjectId {
                        value: source.commit.hex.clone(),
                        reason: "object algorithm does not match object ID length",
                    });
                }
                GitRepositoryUrl::new(source.repository.as_str())?;
                Ok(())
            }
            Self::ServerProject(source) => {
                ServerProjectSourceId::new(
                    &source.service,
                    source.project.clone(),
                    source.snapshot.clone(),
                    source.tree_digest.clone(),
                )?;
                Ok(())
            }
            Self::Registry(source) => {
                if source.registry != *source.package.registry() {
                    return Err(IdentityError::InvalidName {
                        kind: "registry source",
                        value: source.package.to_string(),
                        reason: "source registry and package registry disagree",
                    });
                }
                Ok(())
            }
        }
    }
}

impl Display for SourceId {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Path(source) => write!(
                formatter,
                "path:{}#{}",
                source.workspace_path, source.tree_digest
            ),
            Self::Git(source) => write!(
                formatter,
                "git:{}@{}:{}#{}",
                source.repository, source.commit.hex, source.subdir, source.tree_digest
            ),
            Self::ServerProject(source) => write!(
                formatter,
                "project:{}:{}@{}#{}",
                source.service, source.project, source.snapshot, source.tree_digest
            ),
            Self::Registry(source) => write!(
                formatter,
                "registry:{}@{}#{}",
                source.package, source.version, source.tree_digest
            ),
        }
    }
}

fn normalize_git_repository(value: &str) -> Result<String, IdentityError> {
    let mut url = Url::parse(value).map_err(|_| IdentityError::InvalidGitSource {
        value: value.to_string(),
        reason: "must be an absolute URL",
    })?;
    reject_url_secrets(&url, value)?;
    if !matches!(url.scheme(), "https" | "ssh") {
        return Err(IdentityError::InvalidGitSource {
            value: value.to_string(),
            reason: "scheme must be `https` or `ssh`",
        });
    }
    url.set_fragment(None);
    url.set_query(None);
    normalize_default_port(&mut url);
    let normalized_path = url.path().trim_end_matches('/').to_string();
    url.set_path(if normalized_path.is_empty() {
        "/"
    } else {
        &normalized_path
    });
    Ok(url.to_string())
}

fn normalize_service_origin(value: &str) -> Result<String, IdentityError> {
    let mut url = Url::parse(value).map_err(|_| IdentityError::InvalidServerProjectSource {
        value: value.to_string(),
        reason: "service must be an absolute HTTPS origin",
    })?;
    reject_url_secrets(&url, value).map_err(|_| IdentityError::InvalidServerProjectSource {
        value: value.to_string(),
        reason: "service origin cannot contain credentials, query, or fragment",
    })?;
    if url.scheme() != "https" {
        return Err(IdentityError::InvalidServerProjectSource {
            value: value.to_string(),
            reason: "service origin must use HTTPS",
        });
    }
    normalize_default_port(&mut url);
    let normalized_path = url.path().trim_end_matches('/').to_string();
    url.set_path(&normalized_path);
    url.set_query(None);
    url.set_fragment(None);
    Ok(url.to_string().trim_end_matches('/').to_string())
}

fn reject_url_secrets(url: &Url, value: &str) -> Result<(), IdentityError> {
    if !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(IdentityError::InvalidGitSource {
            value: value.to_string(),
            reason: "credentials, query parameters, and fragments are prohibited",
        });
    }
    Ok(())
}

fn normalize_default_port(url: &mut Url) {
    let is_default = matches!(
        (url.scheme(), url.port()),
        ("https", Some(443)) | ("ssh", Some(22))
    );
    if is_default {
        let _ = url.set_port(None);
    }
}

fn invalid_path(value: &str, reason: &'static str) -> IdentityError {
    IdentityError::InvalidRelativePath {
        value: value.to_string(),
        reason,
    }
}
