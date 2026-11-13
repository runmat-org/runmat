use std::collections::BTreeSet;
use std::io::Read as _;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use crate::{JAVA_ADAPTER_ID, JAVA_ADAPTER_VERSION};

pub const JAVA_ARCHIVE_MEDIA_TYPE: &str = "application/java-archive";
pub const JAVA_ARTIFACT_BUNDLE_SCHEMA_VERSION: u16 = 1;
const MAX_ARTIFACTS: usize = 4_096;
const MAX_ARTIFACT_BYTES: usize = 1024 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct JavaArtifactIdentity(String);

impl JavaArtifactIdentity {
    pub fn for_bytes(bytes: &[u8]) -> Self {
        Self(format!(
            "{JAVA_ADAPTER_ID}:v{JAVA_ADAPTER_VERSION}:sha256:{}",
            hex_digest(bytes)
        ))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn validate_bytes(&self, bytes: &[u8]) -> Result<(), JavaArtifactError> {
        if self == &Self::for_bytes(bytes) {
            Ok(())
        } else {
            Err(JavaArtifactError::Invalid(
                "Java artifact bytes do not match their identity".into(),
            ))
        }
    }

    pub fn validate_file(&self, path: &Path) -> Result<(), JavaArtifactError> {
        let mut file = std::fs::File::open(path).map_err(|error| {
            JavaArtifactError::Invalid(format!(
                "cannot read Java artifact `{}`: {error}",
                path.display()
            ))
        })?;
        let length = file
            .metadata()
            .map_err(|error| {
                JavaArtifactError::Invalid(format!(
                    "cannot inspect Java artifact `{}`: {error}",
                    path.display()
                ))
            })?
            .len();
        if length > MAX_ARTIFACT_BYTES as u64 {
            return Err(JavaArtifactError::Invalid(format!(
                "Java artifact `{}` exceeds its byte limit",
                path.display()
            )));
        }
        let mut header = [0_u8; 4];
        file.read_exact(&mut header).map_err(|error| {
            JavaArtifactError::Invalid(format!(
                "cannot read Java artifact `{}`: {error}",
                path.display()
            ))
        })?;
        if !is_zip_archive(&header) {
            return Err(JavaArtifactError::Invalid(format!(
                "Java artifact `{}` is not a JAR archive",
                path.display()
            )));
        }
        let mut digest = Sha256::new();
        digest.update(header);
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let read = file.read(&mut buffer).map_err(|error| {
                JavaArtifactError::Invalid(format!(
                    "cannot read Java artifact `{}`: {error}",
                    path.display()
                ))
            })?;
            if read == 0 {
                break;
            }
            digest.update(&buffer[..read]);
        }
        let actual = Self(format!(
            "{JAVA_ADAPTER_ID}:v{JAVA_ADAPTER_VERSION}:sha256:{}",
            hex_encoded_digest(digest.finalize())
        ));
        if self == &actual {
            Ok(())
        } else {
            Err(JavaArtifactError::Invalid(
                "Java artifact file does not match its identity".into(),
            ))
        }
    }
}

impl std::fmt::Display for JavaArtifactIdentity {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JavaArtifactBundleEntry {
    pub logical_name: String,
    pub identity: JavaArtifactIdentity,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JavaArtifactBundle {
    pub schema_version: u16,
    pub artifacts: Vec<JavaArtifactBundleEntry>,
}

impl JavaArtifactBundle {
    pub fn empty() -> Self {
        Self {
            schema_version: JAVA_ARTIFACT_BUNDLE_SCHEMA_VERSION,
            artifacts: Vec::new(),
        }
    }

    pub fn new(mut artifacts: Vec<JavaArtifactBundleEntry>) -> Result<Self, JavaArtifactError> {
        artifacts.sort_by(|left, right| left.logical_name.cmp(&right.logical_name));
        let bundle = Self {
            schema_version: JAVA_ARTIFACT_BUNDLE_SCHEMA_VERSION,
            artifacts,
        };
        bundle.validate()?;
        Ok(bundle)
    }

    pub fn validate(&self) -> Result<(), JavaArtifactError> {
        if self.schema_version != JAVA_ARTIFACT_BUNDLE_SCHEMA_VERSION {
            return Err(JavaArtifactError::Invalid(format!(
                "unsupported Java artifact bundle schema {}",
                self.schema_version
            )));
        }
        if self.artifacts.len() > MAX_ARTIFACTS {
            return Err(JavaArtifactError::Invalid(
                "Java artifact bundle exceeds its entry limit".into(),
            ));
        }
        if self
            .artifacts
            .windows(2)
            .any(|pair| pair[0].logical_name >= pair[1].logical_name)
        {
            return Err(JavaArtifactError::Invalid(
                "Java artifacts must be sorted and unique by logical name".into(),
            ));
        }
        let mut identities = BTreeSet::new();
        let mut total_bytes = 0_usize;
        for artifact in &self.artifacts {
            if artifact.logical_name.is_empty()
                || artifact.logical_name.len() > 512
                || artifact.logical_name.chars().any(char::is_control)
            {
                return Err(JavaArtifactError::Invalid(
                    "Java artifact logical name is invalid".into(),
                ));
            }
            total_bytes = total_bytes
                .checked_add(artifact.bytes.len())
                .ok_or_else(|| {
                    JavaArtifactError::Invalid("Java artifact byte total overflowed".into())
                })?;
            if total_bytes > MAX_ARTIFACT_BYTES {
                return Err(JavaArtifactError::Invalid(
                    "Java artifact bundle exceeds its byte limit".into(),
                ));
            }
            if !is_zip_archive(&artifact.bytes) {
                return Err(JavaArtifactError::Invalid(format!(
                    "Java artifact `{}` is not a JAR archive",
                    artifact.logical_name
                )));
            }
            artifact.identity.validate_bytes(&artifact.bytes)?;
            if !identities.insert(&artifact.identity) {
                return Err(JavaArtifactError::Invalid(
                    "Java artifact identity appears more than once".into(),
                ));
            }
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, JavaArtifactError> {
        self.validate()?;
        serde_json::to_vec(self).map_err(|error| JavaArtifactError::Encoding(error.to_string()))
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, JavaArtifactError> {
        let bundle: Self = serde_json::from_slice(bytes)
            .map_err(|error| JavaArtifactError::Encoding(error.to_string()))?;
        bundle.validate()?;
        if bundle.canonical_bytes()? != bytes {
            return Err(JavaArtifactError::Invalid(
                "Java artifact bundle encoding is not canonical".into(),
            ));
        }
        Ok(bundle)
    }
}

impl Default for JavaArtifactBundle {
    fn default() -> Self {
        Self::empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum JavaArtifactError {
    #[error("invalid Java artifact: {0}")]
    Invalid(String),
    #[error("failed to encode Java artifact: {0}")]
    Encoding(String),
}

fn hex_digest(bytes: &[u8]) -> String {
    hex_encoded_digest(Sha256::digest(bytes))
}

fn hex_encoded_digest(digest: impl AsRef<[u8]>) -> String {
    let digest = digest.as_ref();
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("write to string");
    }
    output
}

fn is_zip_archive(bytes: &[u8]) -> bool {
    bytes.starts_with(b"PK\x03\x04")
        || bytes.starts_with(b"PK\x05\x06")
        || bytes.starts_with(b"PK\x07\x08")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundle_is_canonical_and_content_addressed() {
        let bytes = b"PK\x05\x06\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0".to_vec();
        let bundle = JavaArtifactBundle::new(vec![JavaArtifactBundleEntry {
            logical_name: "fixture".into(),
            identity: JavaArtifactIdentity::for_bytes(&bytes),
            bytes,
        }])
        .unwrap();
        let encoded = bundle.canonical_bytes().unwrap();
        assert_eq!(
            JavaArtifactBundle::from_canonical_bytes(&encoded).unwrap(),
            bundle
        );
    }

    #[test]
    fn identity_rejects_changed_bytes() {
        let identity = JavaArtifactIdentity::for_bytes(b"first");
        assert!(identity.validate_bytes(b"second").is_err());
    }

    #[test]
    fn file_validation_checks_archive_shape_and_exact_identity() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("fixture.jar");
        let bytes = b"PK\x05\x06\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
        std::fs::write(&path, bytes).unwrap();
        let identity = JavaArtifactIdentity::for_bytes(bytes);
        identity.validate_file(&path).unwrap();

        std::fs::write(&path, b"not a jar").unwrap();
        assert!(identity.validate_file(&path).is_err());
    }
}
