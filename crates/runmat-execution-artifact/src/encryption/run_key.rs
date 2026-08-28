use aes_gcm::aead::{Aead as _, KeyInit as _, Payload};
use aes_gcm::{Aes256Gcm, Nonce};
use hkdf::Hkdf;
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use zeroize::{Zeroize, ZeroizeOnDrop};

use super::{
    EncryptedArtifact, EncryptionContext, EncryptionPurpose, ExecutionEncryptionProvider,
    ExecutionRecipientKey,
};
use crate::{ArtifactError, ArtifactResult};

const KEY_SCHEDULE_INFO: &[u8] = b"runmat-execution-run-object-v1";

/// The content secret for one remote run.
///
/// It is created by the submitter, wrapped independently to each verified
/// execution/recovery recipient with the execution HPKE provider, and never
/// sent to the RunMat control plane in plaintext.
#[derive(Clone, PartialEq, Eq, Zeroize, ZeroizeOnDrop)]
pub struct RunKeyMaterial([u8; 32]);

impl RunKeyMaterial {
    /// Construct key material from bytes supplied by the platform CSPRNG.
    ///
    /// Browser callers must use `crypto.getRandomValues`; native callers use
    /// [`super::NativeExecutionEncryption`] or their OS CSPRNG.
    pub fn from_entropy(entropy: [u8; 32]) -> ArtifactResult<Self> {
        if entropy.iter().all(|byte| *byte == 0) {
            return Err(ArtifactError::Encryption(
                "run key entropy must not be all zero".into(),
            ));
        }
        Ok(Self(entropy))
    }

    pub fn expose_for_recipient_envelope(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunKeyEnvelope {
    pub schema_version: u16,
    pub recipient_fingerprint: String,
    pub encrypted_key: EncryptedArtifact,
}

pub fn seal_run_key<P: ExecutionEncryptionProvider>(
    provider: &P,
    recipient: &ExecutionRecipientKey,
    run_key: &RunKeyMaterial,
    run_identity: impl Into<String>,
    key_epoch: u32,
) -> ArtifactResult<RunKeyEnvelope> {
    let plaintext = run_key.expose_for_recipient_envelope();
    let context = run_key_context(run_identity.into(), plaintext, key_epoch);
    Ok(envelope(
        recipient,
        provider.seal(recipient, context, plaintext)?,
    ))
}

pub(crate) fn run_key_context(
    run_identity: String,
    plaintext: &[u8],
    key_epoch: u32,
) -> EncryptionContext {
    EncryptionContext {
        schema_version: 1,
        run_identity,
        purpose: EncryptionPurpose::RunKeyEnvelope,
        object_digest: runmat_execution::Digest::sha256(plaintext),
        task_identity: None,
        attempt_identity: None,
        chunk_index: 0,
        total_length: plaintext.len() as u64,
        key_epoch,
    }
}

pub(crate) fn envelope(
    recipient: &ExecutionRecipientKey,
    encrypted_key: EncryptedArtifact,
) -> RunKeyEnvelope {
    RunKeyEnvelope {
        schema_version: 1,
        recipient_fingerprint: recipient.fingerprint.clone(),
        encrypted_key,
    }
}

pub fn open_run_key<P: ExecutionEncryptionProvider>(
    provider: &P,
    private_key: &P::PrivateKey,
    envelope: &RunKeyEnvelope,
    expected_recipient_fingerprint: &str,
    expected_run_identity: &str,
    expected_key_epoch: u32,
) -> ArtifactResult<RunKeyMaterial> {
    validate_envelope(
        envelope,
        expected_recipient_fingerprint,
        expected_run_identity,
        expected_key_epoch,
    )?;
    material_from_plaintext(provider.open(private_key, &envelope.encrypted_key)?)
}

pub(crate) fn validate_envelope(
    envelope: &RunKeyEnvelope,
    expected_recipient_fingerprint: &str,
    expected_run_identity: &str,
    expected_key_epoch: u32,
) -> ArtifactResult<()> {
    if envelope.schema_version != 1
        || envelope.recipient_fingerprint != expected_recipient_fingerprint
        || envelope.encrypted_key.context.run_identity != expected_run_identity
        || envelope.encrypted_key.context.key_epoch != expected_key_epoch
        || envelope.encrypted_key.context.purpose != EncryptionPurpose::RunKeyEnvelope
    {
        return Err(ArtifactError::Identity(
            "run key envelope authority does not match the expected recipient and run".into(),
        ));
    }
    Ok(())
}

pub(crate) fn material_from_plaintext(plaintext: Vec<u8>) -> ArtifactResult<RunKeyMaterial> {
    let bytes: [u8; 32] = plaintext
        .try_into()
        .map_err(|_| ArtifactError::Encryption("decrypted run key has an invalid length".into()))?;
    RunKeyMaterial::from_entropy(bytes)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[repr(u8)]
pub enum RunObjectEncryptionSuite {
    HkdfSha256Aes256GcmV1,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EncryptedRunObject {
    pub schema_version: u16,
    pub suite: RunObjectEncryptionSuite,
    pub context: EncryptionContext,
    /// A unique 256-bit value supplied by the platform CSPRNG. It is public
    /// and drives a context-bound key and nonce through HKDF.
    pub derivation_salt: Vec<u8>,
    pub ciphertext: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RunObjectEncryption;

impl RunObjectEncryption {
    pub fn seal_with_entropy(
        &self,
        run_key: &RunKeyMaterial,
        derivation_salt: [u8; 32],
        context: EncryptionContext,
        plaintext: &[u8],
    ) -> ArtifactResult<EncryptedRunObject> {
        validate_context(&context, plaintext)?;
        if derivation_salt.iter().all(|byte| *byte == 0) {
            return Err(ArtifactError::Encryption(
                "object encryption salt must not be all zero".into(),
            ));
        }
        let (key, nonce) = derive(run_key, &derivation_salt, &context)?;
        let cipher = Aes256Gcm::new_from_slice(&key).map_err(encryption)?;
        let aad = context.aad()?;
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

    pub fn open(
        &self,
        run_key: &RunKeyMaterial,
        object: &EncryptedRunObject,
    ) -> ArtifactResult<Vec<u8>> {
        if object.schema_version != 1
            || object.suite != RunObjectEncryptionSuite::HkdfSha256Aes256GcmV1
            || object.derivation_salt.len() != 32
        {
            return Err(ArtifactError::Invalid(
                "unsupported encrypted run object".into(),
            ));
        }
        object.context.validate()?;
        let (key, nonce) = derive(run_key, &object.derivation_salt, &object.context)?;
        let cipher = Aes256Gcm::new_from_slice(&key).map_err(encryption)?;
        let aad = object.context.aad()?;
        let plaintext = cipher
            .decrypt(
                Nonce::from_slice(&nonce),
                Payload {
                    msg: &object.ciphertext,
                    aad: &aad,
                },
            )
            .map_err(encryption)?;
        validate_context(&object.context, &plaintext)?;
        Ok(plaintext)
    }
}

fn derive(
    run_key: &RunKeyMaterial,
    salt: &[u8],
    context: &EncryptionContext,
) -> ArtifactResult<([u8; 32], [u8; 12])> {
    let aad = context.aad()?;
    let hkdf = Hkdf::<Sha256>::new(Some(salt), &run_key.0);
    let mut output = [0_u8; 44];
    let mut info = KEY_SCHEDULE_INFO.to_vec();
    info.extend_from_slice(&aad);
    hkdf.expand(&info, &mut output)
        .map_err(|_| ArtifactError::Encryption("run object key derivation failed".into()))?;
    let mut key = [0_u8; 32];
    key.copy_from_slice(&output[..32]);
    let mut nonce = [0_u8; 12];
    nonce.copy_from_slice(&output[32..]);
    Ok((key, nonce))
}

fn validate_context(context: &EncryptionContext, plaintext: &[u8]) -> ArtifactResult<()> {
    context.validate()?;
    if context.total_length != plaintext.len() as u64
        || context.object_digest != runmat_execution::Digest::sha256(plaintext)
    {
        return Err(ArtifactError::Identity(
            "run object does not match its authenticated identity".into(),
        ));
    }
    Ok(())
}

fn encryption(error: impl std::fmt::Display) -> ArtifactError {
    ArtifactError::Encryption(error.to_string())
}
