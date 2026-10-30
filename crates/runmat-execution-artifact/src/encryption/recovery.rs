//! Offline recovery using the same canonical HPKE envelope and run-object
//! codecs as submission and execution. No server or filesystem is required.

use super::{
    decode_encrypted_run_object, decode_run_key_envelope, EncryptionPurpose,
    PortableExecutionEncryption, PortableExecutionPrivateKey, RunObjectEncryption,
};
use crate::{ArtifactError, ArtifactResult};
use runmat_execution::security::recipient_fingerprint;

/// Expected authenticated identity supplied by the caller, not by the envelope.
pub struct RecoveryArtifactScope<'a> {
    pub run_identity: &'a str,
    pub key_epoch: u32,
    pub purpose: EncryptionPurpose,
}

/// Open a canonical binary recovery envelope and an encrypted result or
/// diagnostic. The private key is an exact 32-byte X25519 secret, not keygen
/// entropy. Callers retain responsibility for secure key input and plaintext
/// storage; returned plaintext must not be logged automatically.
pub fn open_recovery_artifact(
    private_key: &PortableExecutionPrivateKey,
    envelope_bytes: &[u8],
    artifact_bytes: &[u8],
    scope: RecoveryArtifactScope<'_>,
    maximum_artifact_bytes: usize,
) -> ArtifactResult<Vec<u8>> {
    if !matches!(
        scope.purpose,
        EncryptionPurpose::Result | EncryptionPurpose::DetailedEvent
    ) {
        return Err(ArtifactError::Invalid(
            "recovery only opens results or diagnostics".into(),
        ));
    }
    let fingerprint = recipient_fingerprint(&private_key.public_key_bytes());
    let envelope = decode_run_key_envelope(envelope_bytes, 64 * 1024)?;
    let key = PortableExecutionEncryption.open_run_key(
        private_key,
        &envelope,
        &fingerprint,
        scope.run_identity,
        scope.key_epoch,
    )?;
    let artifact = decode_encrypted_run_object(artifact_bytes, maximum_artifact_bytes)?;
    if artifact.context.run_identity != scope.run_identity
        || artifact.context.key_epoch != scope.key_epoch
        || artifact.context.purpose != scope.purpose
    {
        return Err(ArtifactError::Identity(
            "recovery artifact has the wrong run, epoch, or purpose".into(),
        ));
    }
    RunObjectEncryption.open(&key, &artifact)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encryption::{
        encode_encrypted_run_object, encode_run_key_envelope, EncryptionContext, RunKeyMaterial,
    };
    use runmat_execution::Digest;

    #[test]
    fn canonical_recovery_authenticates_recipient_scope_and_ciphertext() {
        let (recipient, private) = PortableExecutionEncryption
            .recipient_from_entropy_with_derived_fingerprint([7; 32], 1, 1000)
            .unwrap();
        let run_key = RunKeyMaterial::from_entropy([9; 32]).unwrap();
        let envelope = PortableExecutionEncryption
            .seal_run_key_with_entropy([11; 32], &recipient, &run_key, "run-recovery", 1)
            .unwrap();
        let plaintext = b"authenticated result";
        let artifact = RunObjectEncryption
            .seal_with_entropy(
                &run_key,
                [13; 32],
                EncryptionContext {
                    schema_version: 1,
                    run_identity: "run-recovery".into(),
                    purpose: EncryptionPurpose::Result,
                    object_digest: Digest::sha256(plaintext),
                    task_identity: None,
                    attempt_identity: None,
                    chunk_index: 0,
                    total_length: plaintext.len() as u64,
                    key_epoch: 1,
                },
                plaintext,
            )
            .unwrap();
        let envelope = encode_run_key_envelope(&envelope).unwrap();
        let artifact = encode_encrypted_run_object(&artifact).unwrap();
        let open = |key: &PortableExecutionPrivateKey, run, epoch, purpose, bytes: &[u8]| {
            open_recovery_artifact(
                key,
                &envelope,
                bytes,
                RecoveryArtifactScope {
                    run_identity: run,
                    key_epoch: epoch,
                    purpose,
                },
                4096,
            )
        };
        assert_eq!(
            open(
                &private,
                "run-recovery",
                1,
                EncryptionPurpose::Result,
                &artifact
            )
            .unwrap(),
            plaintext
        );
        assert!(open(
            &private,
            "another-run",
            1,
            EncryptionPurpose::Result,
            &artifact
        )
        .is_err());
        assert!(open(
            &private,
            "run-recovery",
            2,
            EncryptionPurpose::Result,
            &artifact
        )
        .is_err());
        assert!(open(
            &private,
            "run-recovery",
            1,
            EncryptionPurpose::DetailedEvent,
            &artifact
        )
        .is_err());
        let wrong = PortableExecutionPrivateKey::from_bytes(&[17; 32]).unwrap();
        assert!(open(
            &wrong,
            "run-recovery",
            1,
            EncryptionPurpose::Result,
            &artifact
        )
        .is_err());
        let mut tampered = artifact.clone();
        let last = tampered.len() - 1;
        tampered[last] ^= 1;
        assert!(open(
            &private,
            "run-recovery",
            1,
            EncryptionPurpose::Result,
            &tampered
        )
        .is_err());
    }
}
