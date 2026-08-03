use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use crate::{ArtifactError, ArtifactResult};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum ObjectNamespace {
    ProgramSource,
    PackageRelease,
    ProgramArtifact,
    InputValue,
    ResultValue,
    DetailedEvent,
    Log,
    Checkpoint,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectDescriptor {
    pub namespace: ObjectNamespace,
    pub logical_name: String,
    pub digest: Digest,
    pub encoded_length: u64,
    pub media_type: String,
}

impl ObjectDescriptor {
    pub fn new(
        namespace: ObjectNamespace,
        logical_name: impl Into<String>,
        media_type: impl Into<String>,
        bytes: &[u8],
    ) -> ArtifactResult<Self> {
        let descriptor = Self {
            namespace,
            logical_name: logical_name.into(),
            digest: Digest::sha256(bytes),
            encoded_length: bytes.len() as u64,
            media_type: media_type.into(),
        };
        descriptor.validate()?;
        Ok(descriptor)
    }

    pub fn validate(&self) -> ArtifactResult<()> {
        validate_logical_name(&self.logical_name)?;
        if self.media_type.is_empty()
            || self.media_type.len() > 128
            || !self.media_type.is_ascii()
            || self.media_type.chars().any(char::is_control)
            || self.media_type.chars().any(char::is_whitespace)
        {
            return Err(ArtifactError::Invalid(
                "object media type is invalid".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LogicalObject {
    pub descriptor: ObjectDescriptor,
    pub bytes: Vec<u8>,
}

impl LogicalObject {
    pub fn new(
        namespace: ObjectNamespace,
        logical_name: impl Into<String>,
        media_type: impl Into<String>,
        bytes: Vec<u8>,
    ) -> ArtifactResult<Self> {
        let descriptor = ObjectDescriptor::new(namespace, logical_name, media_type, &bytes)?;
        Ok(Self { descriptor, bytes })
    }

    pub fn validate(&self) -> ArtifactResult<()> {
        self.descriptor.validate()?;
        if self.descriptor.encoded_length != self.bytes.len() as u64
            || self.descriptor.digest != Digest::sha256(&self.bytes)
        {
            return Err(ArtifactError::Identity(
                "logical object bytes do not match their descriptor".into(),
            ));
        }
        Ok(())
    }
}

fn validate_logical_name(name: &str) -> ArtifactResult<()> {
    if name.is_empty()
        || name.len() > 4096
        || !name.is_ascii()
        || name.starts_with('/')
        || name.starts_with('\\')
        || name.as_bytes().get(1).is_some_and(|byte| *byte == b':')
        || name.contains('\\')
        || name
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
        || name.chars().any(char::is_control)
    {
        return Err(ArtifactError::Invalid(format!(
            "object logical name is not a normalized relative path: {name:?}"
        )));
    }
    Ok(())
}
