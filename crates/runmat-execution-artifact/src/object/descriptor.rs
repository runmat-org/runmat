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
    ForeignArtifact,
    ForeignPolicy,
    ProjectRuntimePolicy,
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
    /// Requested owned string capacities; the descriptor's inline storage and
    /// object payload have separate owners.
    pub fn owned_buffer_bytes(&self) -> ArtifactResult<u64> {
        let Self {
            namespace: _,
            logical_name,
            digest: _,
            encoded_length: _,
            media_type,
        } = self;
        logical_name
            .capacity()
            .checked_add(media_type.capacity())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| ArtifactError::Limit("object descriptor storage overflow".into()))
    }

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

    /// Requested string bytes for content_named; inline descriptor slots are separate.
    pub fn content_named_text_bytes(prefix: &str, media_type: &str) -> ArtifactResult<u64> {
        prefix
            .len()
            .checked_add(65)
            .and_then(|length| length.checked_add(media_type.len()))
            .and_then(|length| u64::try_from(length).ok())
            .ok_or_else(|| ArtifactError::Limit("object descriptor text storage overflow".into()))
    }

    /// Construct metadata for an already identified content buffer. This does
    /// not certify or read content; LogicalObject::validate remains the raw check.
    pub fn content_named(
        namespace: ObjectNamespace,
        prefix: &str,
        media_type: &str,
        digest: Digest,
        encoded_length: u64,
    ) -> ArtifactResult<Self> {
        let length = prefix
            .len()
            .checked_add(65)
            .ok_or_else(|| ArtifactError::Limit("object name length overflow".into()))?;
        let mut name = String::with_capacity(length);
        name.push_str(prefix);
        name.push('/');
        const HEX: &[u8; 16] = b"0123456789abcdef";
        for byte in digest.bytes() {
            name.push(HEX[(byte >> 4) as usize] as char);
            name.push(HEX[(byte & 15) as usize] as char);
        }
        let descriptor = Self {
            namespace,
            logical_name: name,
            digest,
            encoded_length,
            media_type: media_type.to_owned(),
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

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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

pub(crate) fn validate_logical_name(name: &str) -> ArtifactResult<()> {
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
