use minicbor::Encoder;
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use crate::{ArtifactError, ArtifactResult};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum EncryptionPurpose {
    Bundle,
    Input,
    Result,
    DetailedEvent,
    Log,
    Checkpoint,
    TransferFrame,
    RunKeyEnvelope,
    Program,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EncryptionContext {
    pub schema_version: u16,
    pub run_identity: String,
    pub purpose: EncryptionPurpose,
    pub object_digest: Digest,
    pub task_identity: Option<String>,
    pub attempt_identity: Option<String>,
    pub chunk_index: u64,
    pub total_length: u64,
    pub key_epoch: u32,
}

impl EncryptionContext {
    pub fn validate(&self) -> ArtifactResult<()> {
        if self.schema_version != 1
            || !valid_identity(&self.run_identity)
            || self
                .task_identity
                .as_deref()
                .is_some_and(|value| !valid_identity(value))
            || self
                .attempt_identity
                .as_deref()
                .is_some_and(|value| !valid_identity(value))
            || self.attempt_identity.is_some() && self.task_identity.is_none()
            || self.key_epoch == 0
        {
            return Err(ArtifactError::Invalid(
                "execution encryption context is malformed".into(),
            ));
        }
        Ok(())
    }

    pub fn aad(&self) -> ArtifactResult<Vec<u8>> {
        self.validate()?;
        let mut bytes = b"runmat-execution-encryption-context-v1\0".to_vec();
        let mut encoder = Encoder::new(&mut bytes);
        encoder
            .array(9)
            .and_then(|encoder| encoder.u16(self.schema_version))
            .and_then(|encoder| encoder.str(&self.run_identity))
            .and_then(|encoder| encoder.u8(self.purpose as u8))
            .and_then(|encoder| encoder.bytes(self.object_digest.bytes()))
            .map_err(encoding)?;
        encode_optional_text(&mut encoder, self.task_identity.as_deref())?;
        encode_optional_text(&mut encoder, self.attempt_identity.as_deref())?;
        encoder
            .u64(self.chunk_index)
            .and_then(|encoder| encoder.u64(self.total_length))
            .and_then(|encoder| encoder.u32(self.key_epoch))
            .map_err(encoding)?;
        Ok(bytes)
    }
}

fn valid_identity(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 256
        && value.is_ascii()
        && !value.chars().any(char::is_control)
}

fn encode_optional_text(
    encoder: &mut Encoder<&mut Vec<u8>>,
    value: Option<&str>,
) -> ArtifactResult<()> {
    match value {
        Some(value) => encoder.str(value),
        None => encoder.null(),
    }
    .map(|_| ())
    .map_err(encoding)
}

fn encoding(error: minicbor::encode::Error<std::convert::Infallible>) -> ArtifactError {
    ArtifactError::Encoding(error.to_string())
}
