use aes_gcm::aead::{Aead as _, KeyInit as _, Payload};
use aes_gcm::{Aes256Gcm, Nonce};
use runmat_execution::Digest;

use super::{
    run_key::derive, EncryptedRunObject, EncryptionContext, EncryptionPurpose, RunKeyMaterial,
    RunObjectEncryptionSuite,
};
use crate::{ArtifactError, ArtifactResult};

/// A Program chunk is one bounded member of a separately admitted, ordered
/// Graph Program manifest. Direct V7 objects continue to use RunObjectEncryption.
pub const MAX_PROGRAM_CHUNK_PLAINTEXT_BYTES: usize =
    runmat_execution::resource::GRAPH_PROGRAM_CHUNK_PLAINTEXT_BYTES as usize;
/// Exact upper bound for the binary CBOR envelope emitted by
/// `encode_encrypted_run_object` for a sealed Program chunk. The chunk context
/// has one <=256-byte run identity and null task/attempt identities; the salt
/// is exactly 32 bytes. Each term below is the maximum encoded width in the
/// canonical 13-field object, including its domain prefix.
pub const MAX_PROGRAM_CHUNK_CBOR_OVERHEAD_BYTES: usize = b"runmat-encrypted-run-object-v1\0".len()
    + 1 // array(13)
    + 3 // schema_version: u16
    + 2 // suite: u8
    + 3 + 256 // run_identity: bounded text
    + 2 // purpose: u8
    + 2 + 32 // object_digest: 32-byte string
    + 1 + 1 // null task_identity and attempt_identity
    + 9 + 9 // chunk_index and total_length: u64
    + 5 // key_epoch: u32
    + 2 + 32 // derivation_salt: 32-byte string
    + 5 // ciphertext byte-string header (length <= 4 MiB + 16)
    + 3; // context schema_version: u16
pub const MAX_PROGRAM_CHUNK_ENCODED_OBJECT_BYTES: usize =
    MAX_PROGRAM_CHUNK_PLAINTEXT_BYTES + 16 + MAX_PROGRAM_CHUNK_CBOR_OVERHEAD_BYTES;
const CHUNK_AAD_DOMAIN: &[u8] = b"runmat-graph-program-chunk-v1\0";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProgramChunkAuthority {
    pub run_identity: String,
    pub program_digest: Digest,
    pub program_length: u64,
    pub chunk_index: u64,
    pub chunk_count: u64,
    pub chunk_length: u32,
    pub chunk_digest: Digest,
    pub key_epoch: u32,
}

impl ProgramChunkAuthority {
    fn context(&self) -> ArtifactResult<EncryptionContext> {
        if self.program_length == 0
            || self.chunk_count == 0
            || self.chunk_index >= self.chunk_count
            || self.chunk_length == 0
            || self.chunk_length as usize > MAX_PROGRAM_CHUNK_PLAINTEXT_BYTES
            || self.chunk_count
                != self
                    .program_length
                    .div_ceil(MAX_PROGRAM_CHUNK_PLAINTEXT_BYTES as u64)
        {
            return Err(ArtifactError::Invalid(
                "Graph Program chunk authority is malformed".into(),
            ));
        }
        let expected_length = if self.chunk_index + 1 == self.chunk_count {
            self.program_length - (self.chunk_count - 1) * MAX_PROGRAM_CHUNK_PLAINTEXT_BYTES as u64
        } else {
            MAX_PROGRAM_CHUNK_PLAINTEXT_BYTES as u64
        };
        if self.chunk_length as u64 != expected_length {
            return Err(ArtifactError::Invalid(
                "Graph Program chunk length differs from its ordinal".into(),
            ));
        }
        let context = EncryptionContext {
            schema_version: 1,
            run_identity: self.run_identity.clone(),
            purpose: EncryptionPurpose::Program,
            object_digest: self.program_digest,
            task_identity: None,
            attempt_identity: None,
            chunk_index: self.chunk_index,
            total_length: self.program_length,
            key_epoch: self.key_epoch,
        };
        context.validate()?;
        Ok(context)
    }

    fn aad(&self, context: &EncryptionContext) -> ArtifactResult<Vec<u8>> {
        let mut aad = context.aad()?;
        aad.extend_from_slice(CHUNK_AAD_DOMAIN);
        aad.extend_from_slice(&self.chunk_count.to_be_bytes());
        aad.extend_from_slice(&self.chunk_length.to_be_bytes());
        aad.extend_from_slice(self.chunk_digest.bytes());
        Ok(aad)
    }
}

pub fn seal_program_chunk(
    run_key: &RunKeyMaterial,
    derivation_salt: [u8; 32],
    authority: &ProgramChunkAuthority,
    plaintext: &[u8],
) -> ArtifactResult<EncryptedRunObject> {
    let context = authority.context()?;
    if plaintext.len() != authority.chunk_length as usize
        || Digest::sha256(plaintext) != authority.chunk_digest
    {
        return Err(ArtifactError::Identity(
            "Graph Program chunk differs from its manifest entry".into(),
        ));
    }
    if derivation_salt.iter().all(|byte| *byte == 0) {
        return Err(ArtifactError::Encryption(
            "Program chunk encryption salt must not be all zero".into(),
        ));
    }
    let (key, nonce) = derive(run_key, &derivation_salt, &context)?;
    let cipher = Aes256Gcm::new_from_slice(&key).map_err(encryption)?;
    let aad = authority.aad(&context)?;
    let ciphertext = cipher
        .encrypt(
            Nonce::from_slice(&nonce),
            Payload {
                msg: plaintext,
                aad: &aad,
            },
        )
        .map_err(encryption)?;
    Ok(EncryptedRunObject {
        schema_version: 1,
        suite: RunObjectEncryptionSuite::HkdfSha256Aes256GcmV1,
        context,
        derivation_salt: derivation_salt.to_vec(),
        ciphertext,
    })
}

pub fn open_program_chunk(
    run_key: &RunKeyMaterial,
    authority: &ProgramChunkAuthority,
    encrypted: &EncryptedRunObject,
) -> ArtifactResult<Vec<u8>> {
    let expected_context = authority.context()?;
    if encrypted.schema_version != 1
        || encrypted.suite != RunObjectEncryptionSuite::HkdfSha256Aes256GcmV1
        || encrypted.context != expected_context
        || encrypted.derivation_salt.len() != 32
        || encrypted.ciphertext.len() != authority.chunk_length as usize + 16
    {
        return Err(ArtifactError::Identity(
            "encrypted Graph Program chunk has different authority".into(),
        ));
    }
    let (key, nonce) = derive(run_key, &encrypted.derivation_salt, &encrypted.context)?;
    let cipher = Aes256Gcm::new_from_slice(&key).map_err(encryption)?;
    let aad = authority.aad(&encrypted.context)?;
    let plaintext = cipher
        .decrypt(
            Nonce::from_slice(&nonce),
            Payload {
                msg: &encrypted.ciphertext,
                aad: &aad,
            },
        )
        .map_err(encryption)?;
    if plaintext.len() != authority.chunk_length as usize
        || Digest::sha256(&plaintext) != authority.chunk_digest
    {
        return Err(ArtifactError::Identity(
            "decrypted Graph Program chunk differs from its manifest entry".into(),
        ));
    }
    Ok(plaintext)
}

fn encryption(error: impl std::fmt::Display) -> ArtifactError {
    ArtifactError::Encryption(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encryption::{encode_encrypted_run_object, RunObjectEncryption};

    #[test]
    fn encoded_chunk_bound_covers_the_largest_canonical_context() {
        let mut expected = authority(0);
        expected.run_identity = "r".repeat(256);
        expected.program_length = MAX_PROGRAM_CHUNK_PLAINTEXT_BYTES as u64;
        expected.chunk_length = MAX_PROGRAM_CHUNK_PLAINTEXT_BYTES as u32;
        let encoded = encode_encrypted_run_object(&EncryptedRunObject {
            schema_version: 1,
            suite: RunObjectEncryptionSuite::HkdfSha256Aes256GcmV1,
            context: expected.context().unwrap(),
            derivation_salt: vec![1; 32],
            ciphertext: vec![0; MAX_PROGRAM_CHUNK_PLAINTEXT_BYTES + 16],
        })
        .unwrap();
        assert_eq!(MAX_PROGRAM_CHUNK_CBOR_OVERHEAD_BYTES, 399);
        assert!(encoded.len() <= MAX_PROGRAM_CHUNK_ENCODED_OBJECT_BYTES);
    }

    fn authority(index: u64) -> ProgramChunkAuthority {
        ProgramChunkAuthority {
            run_identity: "run-1".into(),
            program_digest: Digest::sha256(b"full canonical Graph request"),
            program_length: 5,
            chunk_index: index,
            chunk_count: 1,
            chunk_length: 5,
            chunk_digest: Digest::sha256(b"chunk"),
            key_epoch: 1,
        }
    }

    #[test]
    fn chunk_is_bound_to_full_program_and_exact_manifest_entry() {
        let key = RunKeyMaterial::from_entropy([7; 32]).unwrap();
        let expected = authority(0);
        let encrypted = seal_program_chunk(&key, [9; 32], &expected, b"chunk").unwrap();
        assert_eq!(
            open_program_chunk(&key, &expected, &encrypted).unwrap(),
            b"chunk"
        );

        let mut other = expected.clone();
        other.chunk_index = 1;
        assert!(open_program_chunk(&key, &other, &encrypted).is_err());
        other = expected.clone();
        other.program_digest = Digest::sha256(b"different Graph request");
        assert!(open_program_chunk(&key, &other, &encrypted).is_err());
        other = expected.clone();
        other.chunk_count = 3;
        assert!(open_program_chunk(&key, &other, &encrypted).is_err());
        other = expected.clone();
        other.chunk_digest = Digest::sha256(b"other");
        assert!(open_program_chunk(&key, &other, &encrypted).is_err());

        // The whole-object API cannot accidentally admit a chunk as a Direct object.
        assert!(RunObjectEncryption.open(&key, &encrypted).is_err());
    }

    #[test]
    fn oversize_or_wrong_chunk_is_rejected_before_sealing() {
        let key = RunKeyMaterial::from_entropy([7; 32]).unwrap();
        let expected = authority(0);
        assert!(seal_program_chunk(&key, [9; 32], &expected, b"wrong").is_err());
        let mut oversize = expected;
        oversize.chunk_length = MAX_PROGRAM_CHUNK_PLAINTEXT_BYTES as u32 + 1;
        assert!(seal_program_chunk(&key, [9; 32], &oversize, b"chunk").is_err());
    }
}
