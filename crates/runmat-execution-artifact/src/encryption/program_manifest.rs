use runmat_execution::resource::{MAX_GRAPH_PROGRAM_CHUNKS, MAX_GRAPH_PROGRAM_WIRE_BYTES};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use super::{
    EncryptedRunObject, EncryptionContext, EncryptionPurpose, RunKeyMaterial, RunObjectEncryption,
    MAX_PROGRAM_CHUNK_ENCODED_OBJECT_BYTES, MAX_PROGRAM_CHUNK_PLAINTEXT_BYTES,
};
use crate::{ArtifactError, ArtifactResult};

/// The encrypted manifest is a small authority record; archive bytes remain
/// in independently authenticated objects and are never placed in control frames.
pub const MAX_PROGRAM_CHUNK_MANIFEST_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_PROGRAM_CHUNK_MANIFEST_ENCODED_OBJECT_BYTES: usize =
    MAX_PROGRAM_CHUNK_MANIFEST_BYTES + 16 + super::MAX_PROGRAM_CHUNK_CBOR_OVERHEAD_BYTES;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramChunkObject {
    pub artifact_id: String,
    pub plaintext_digest: Digest,
    pub plaintext_length: u32,
    /// Digest/length of the complete encoded EncryptedRunObject HTTP object,
    /// not its inner AES-GCM ciphertext (which is plaintext_length + 16).
    pub encoded_object_digest: Digest,
    pub encoded_object_length: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramChunkManifest {
    pub schema_version: u16,
    pub program_digest: Digest,
    pub program_length: u64,
    pub chunks: Vec<ProgramChunkObject>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProgramManifestAuthority {
    pub run_identity: String,
    pub key_epoch: u32,
}

/// A manifest opened under the exact run/key authority. Only the authenticated
/// opener may issue this staging capability; parsing JSON alone cannot.
pub struct OpenedProgramChunkManifest {
    manifest: ProgramChunkManifest,
    run_identity: String,
    key_epoch: u32,
}

impl OpenedProgramChunkManifest {
    pub fn manifest(&self) -> &ProgramChunkManifest {
        &self.manifest
    }

    pub fn run_identity(&self) -> &str {
        &self.run_identity
    }

    pub fn key_epoch(&self) -> u32 {
        self.key_epoch
    }
}

impl ProgramManifestAuthority {
    fn context(&self, bytes: &[u8]) -> ArtifactResult<EncryptionContext> {
        let context = EncryptionContext {
            schema_version: 1,
            run_identity: self.run_identity.clone(),
            purpose: EncryptionPurpose::Program,
            object_digest: Digest::sha256(bytes),
            task_identity: None,
            attempt_identity: None,
            chunk_index: 0,
            total_length: bytes.len() as u64,
            key_epoch: self.key_epoch,
        };
        context.validate()?;
        Ok(context)
    }

    fn matches_context(&self, context: &EncryptionContext) -> bool {
        context.schema_version == 1
            && context.run_identity == self.run_identity
            && context.purpose == EncryptionPurpose::Program
            && context.task_identity.is_none()
            && context.attempt_identity.is_none()
            && context.chunk_index == 0
            && context.total_length <= MAX_PROGRAM_CHUNK_MANIFEST_BYTES as u64
            && context.key_epoch == self.key_epoch
    }
}

impl ProgramChunkManifest {
    pub fn validate(&self, maximum_program_bytes: u64) -> ArtifactResult<()> {
        if self.schema_version != 1
            || self.program_length == 0
            || self.program_length > maximum_program_bytes
            || self.program_length > MAX_GRAPH_PROGRAM_WIRE_BYTES
            || self.chunks.is_empty()
            || self.chunks.len() > MAX_GRAPH_PROGRAM_CHUNKS
        {
            return Err(ArtifactError::Invalid(
                "Graph Program chunk manifest exceeds its admitted bound".into(),
            ));
        }
        let chunk_size = MAX_PROGRAM_CHUNK_PLAINTEXT_BYTES as u64;
        let expected_count = self.program_length.div_ceil(chunk_size);
        if self.chunks.len() as u64 != expected_count {
            return Err(ArtifactError::Invalid(
                "Graph Program chunk count does not match its length".into(),
            ));
        }
        let mut total = 0_u64;
        let mut artifact_ids = HashSet::with_capacity(self.chunks.len());
        for (index, chunk) in self.chunks.iter().enumerate() {
            let expected_length = if index + 1 == self.chunks.len() {
                self.program_length - total
            } else {
                chunk_size
            };
            if chunk.plaintext_length as u64 != expected_length
                || chunk.encoded_object_length <= expected_length + 16
                || chunk.encoded_object_length > MAX_PROGRAM_CHUNK_ENCODED_OBJECT_BYTES as u64
                || !valid_artifact_id(&chunk.artifact_id)
                || !artifact_ids.insert(&chunk.artifact_id)
            {
                return Err(ArtifactError::Invalid(
                    "Graph Program chunk descriptor is malformed".into(),
                ));
            }
            total = total.checked_add(expected_length).ok_or_else(|| {
                ArtifactError::Invalid("Graph Program chunk length overflow".into())
            })?;
        }
        if total != self.program_length {
            return Err(ArtifactError::Identity(
                "Graph Program chunks do not cover the complete request".into(),
            ));
        }
        Ok(())
    }
}

pub fn seal_program_chunk_manifest(
    run_key: &RunKeyMaterial,
    derivation_salt: [u8; 32],
    authority: &ProgramManifestAuthority,
    manifest: &ProgramChunkManifest,
    maximum_program_bytes: u64,
) -> ArtifactResult<EncryptedRunObject> {
    let bytes = encode_program_chunk_manifest(manifest, maximum_program_bytes)?;
    let context = authority.context(&bytes)?;
    RunObjectEncryption.seal_with_entropy(run_key, derivation_salt, context, &bytes)
}

pub fn open_program_chunk_manifest(
    run_key: &RunKeyMaterial,
    authority: &ProgramManifestAuthority,
    encrypted: &EncryptedRunObject,
    maximum_program_bytes: u64,
) -> ArtifactResult<OpenedProgramChunkManifest> {
    if !authority.matches_context(&encrypted.context) {
        return Err(ArtifactError::Identity(
            "Graph Program manifest has different run or attempt authority".into(),
        ));
    }
    if encrypted.ciphertext.len() > MAX_PROGRAM_CHUNK_MANIFEST_BYTES + 16 {
        return Err(ArtifactError::Invalid(
            "Graph Program encrypted manifest is too large".into(),
        ));
    }
    let bytes = RunObjectEncryption.open(run_key, encrypted)?;
    Ok(OpenedProgramChunkManifest {
        manifest: decode_program_chunk_manifest(&bytes, maximum_program_bytes)?,
        run_identity: authority.run_identity.clone(),
        key_epoch: authority.key_epoch,
    })
}

pub fn encode_program_chunk_manifest(
    manifest: &ProgramChunkManifest,
    maximum_program_bytes: u64,
) -> ArtifactResult<Vec<u8>> {
    manifest.validate(maximum_program_bytes)?;
    let bytes = serde_json::to_vec(manifest).map_err(encoding)?;
    if bytes.len() > MAX_PROGRAM_CHUNK_MANIFEST_BYTES {
        return Err(ArtifactError::Invalid(
            "Graph Program chunk manifest is too large".into(),
        ));
    }
    Ok(bytes)
}

pub fn decode_program_chunk_manifest(
    bytes: &[u8],
    maximum_program_bytes: u64,
) -> ArtifactResult<ProgramChunkManifest> {
    if bytes.len() > MAX_PROGRAM_CHUNK_MANIFEST_BYTES {
        return Err(ArtifactError::Invalid(
            "Graph Program chunk manifest is too large".into(),
        ));
    }
    let manifest: ProgramChunkManifest = serde_json::from_slice(bytes).map_err(encoding)?;
    manifest.validate(maximum_program_bytes)?;
    Ok(manifest)
}

fn valid_artifact_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 256
        && value.is_ascii()
        && !value.chars().any(char::is_control)
}

fn encoding(error: impl std::fmt::Display) -> ArtifactError {
    ArtifactError::Encoding(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_requires_exact_chunk_coverage_and_bounded_descriptors() {
        let manifest = ProgramChunkManifest {
            schema_version: 1,
            program_digest: Digest::sha256(b"complete request"),
            program_length: 5,
            chunks: vec![ProgramChunkObject {
                artifact_id: "program-chunk-0".into(),
                plaintext_digest: Digest::sha256(b"chunk"),
                plaintext_length: 5,
                encoded_object_digest: Digest::sha256(b"encrypted"),
                encoded_object_length: 128,
            }],
        };
        let encoded = encode_program_chunk_manifest(&manifest, 5).unwrap();
        assert_eq!(
            decode_program_chunk_manifest(&encoded, 5).unwrap(),
            manifest
        );
        assert!(decode_program_chunk_manifest(&encoded, 4).is_err());
        let mut missing = manifest.clone();
        missing.chunks.clear();
        assert!(missing.validate(5).is_err());
        let mut wrong_size = manifest;
        wrong_size.chunks[0].plaintext_length = 4;
        assert!(wrong_size.validate(5).is_err());
        wrong_size.chunks[0].plaintext_length = 5;
        wrong_size.chunks[0].encoded_object_length = 21;
        assert!(wrong_size.validate(5).is_err());
    }

    #[test]
    fn encrypted_manifest_is_bound_to_run_and_full_request() {
        let manifest = ProgramChunkManifest {
            schema_version: 1,
            program_digest: Digest::sha256(b"chunk"),
            program_length: 5,
            chunks: vec![ProgramChunkObject {
                artifact_id: "chunk-0".into(),
                plaintext_digest: Digest::sha256(b"chunk"),
                plaintext_length: 5,
                encoded_object_digest: Digest::sha256(b"sealed"),
                encoded_object_length: 128,
            }],
        };
        let authority = ProgramManifestAuthority {
            run_identity: "run-1".into(),
            key_epoch: 1,
        };
        let key = RunKeyMaterial::from_entropy([7; 32]).unwrap();
        let encrypted =
            seal_program_chunk_manifest(&key, [9; 32], &authority, &manifest, 5).unwrap();
        assert_eq!(
            open_program_chunk_manifest(&key, &authority, &encrypted, 5)
                .unwrap()
                .manifest(),
            &manifest
        );
        let mut wrong = authority;
        wrong.run_identity = "run-2".into();
        assert!(open_program_chunk_manifest(&key, &wrong, &encrypted, 5).is_err());
    }
}
