use core::convert::Infallible;

use hpke::rand_core::{TryCryptoRng, TryRng};
use hpke::{
    aead::AesGcm128,
    kdf::HkdfSha256,
    kem::{Kem as _, X25519HkdfSha256},
    setup_receiver, setup_sender_with_rng, Deserializable, OpModeR, OpModeS, Serializable,
};
use runmat_execution::security::recipient_fingerprint;

use super::{
    run_key::{
        envelope, material_from_plaintext, run_key_context, validate_envelope, RunKeyEnvelope,
        RunKeyMaterial,
    },
    EncryptedArtifact, EncryptionContext, ExecutionHpkeSuite, ExecutionRecipientKey,
};
use crate::{ArtifactError, ArtifactResult};

const INFO: &[u8] = b"runmat-execution-artifact-hpke-v1";

#[derive(Clone)]
pub struct PortableExecutionPrivateKey(<X25519HkdfSha256 as hpke::Kem>::PrivateKey);

impl PortableExecutionPrivateKey {
    /// Restore a portable X25519 recipient private key from an exact 32-byte
    /// secret previously returned by [`Self::to_bytes`].
    ///
    /// Callers own secure storage, access control, and zeroization of the
    /// serialized bytes. The execution artifact layer deliberately does not
    /// choose a filesystem, browser storage, or KMS policy.
    pub fn from_bytes(bytes: &[u8]) -> ArtifactResult<Self> {
        let private =
            <X25519HkdfSha256 as hpke::Kem>::PrivateKey::from_bytes(bytes).map_err(encryption)?;
        Ok(Self(private))
    }

    /// Export the exact 32-byte recipient secret for storage by a platform
    /// key custodian.
    pub fn to_bytes(&self) -> [u8; 32] {
        self.0
            .to_bytes()
            .as_slice()
            .try_into()
            .expect("X25519 private keys are always 32 bytes")
    }

    /// Return the public X25519 recipient key corresponding to this secret.
    pub fn public_key_bytes(&self) -> [u8; 32] {
        X25519HkdfSha256::sk_to_pk(&self.0)
            .to_bytes()
            .as_slice()
            .try_into()
            .expect("X25519 public keys are always 32 bytes")
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PortableExecutionEncryption;

impl PortableExecutionEncryption {
    pub fn recipient_from_entropy_with_derived_fingerprint(
        &self,
        entropy: [u8; 32],
        valid_after_unix_millis: u64,
        valid_before_unix_millis: u64,
    ) -> ArtifactResult<(ExecutionRecipientKey, PortableExecutionPrivateKey)> {
        let (private, public) = X25519HkdfSha256::derive_keypair(&entropy);
        let public_key = public.to_bytes().to_vec();
        let recipient = ExecutionRecipientKey {
            suite: ExecutionHpkeSuite::X25519HkdfSha256Aes128GcmV1,
            fingerprint: recipient_fingerprint(&public_key),
            public_key,
            valid_after_unix_millis,
            valid_before_unix_millis,
            custodian_uri: None,
        };
        recipient.validate()?;
        Ok((recipient, PortableExecutionPrivateKey(private)))
    }

    /// Derive a recipient key from 32 bytes supplied by the platform CSPRNG.
    ///
    /// Browser hosts must source these bytes from `crypto.getRandomValues`.
    pub fn recipient_from_entropy(
        &self,
        entropy: [u8; 32],
        fingerprint: impl Into<String>,
        valid_after_unix_millis: u64,
        valid_before_unix_millis: u64,
    ) -> ArtifactResult<(ExecutionRecipientKey, PortableExecutionPrivateKey)> {
        let (private, public) = X25519HkdfSha256::derive_keypair(&entropy);
        let recipient = ExecutionRecipientKey {
            suite: ExecutionHpkeSuite::X25519HkdfSha256Aes128GcmV1,
            public_key: public.to_bytes().to_vec(),
            fingerprint: fingerprint.into(),
            valid_after_unix_millis,
            valid_before_unix_millis,
            custodian_uri: None,
        };
        recipient.validate()?;
        Ok((recipient, PortableExecutionPrivateKey(private)))
    }

    /// Seal with 32 fresh bytes supplied by the platform CSPRNG.
    ///
    /// Reusing `ephemeral_entropy` for the same or another recipient is forbidden.
    pub fn seal_with_entropy(
        &self,
        ephemeral_entropy: [u8; 32],
        recipient: &ExecutionRecipientKey,
        context: EncryptionContext,
        plaintext: &[u8],
    ) -> ArtifactResult<EncryptedArtifact> {
        recipient.validate()?;
        validate_context(&context, plaintext)?;
        let public = <X25519HkdfSha256 as hpke::Kem>::PublicKey::from_bytes(&recipient.public_key)
            .map_err(encryption)?;
        let mut entropy = ExactEntropy::new(ephemeral_entropy);
        let (encapsulated, mut sender) = setup_sender_with_rng::<
            AesGcm128,
            HkdfSha256,
            X25519HkdfSha256,
        >(&OpModeS::Base, &public, INFO, &mut entropy)
        .map_err(encryption)?;
        if !entropy.consumed {
            return Err(ArtifactError::Encryption(
                "HPKE provider did not consume ephemeral entropy".into(),
            ));
        }
        let ciphertext = sender
            .seal(plaintext, &context.aad()?)
            .map_err(encryption)?;
        Ok(EncryptedArtifact {
            schema_version: 1,
            suite: recipient.suite,
            context,
            encapsulated_key: encapsulated.to_bytes().to_vec(),
            ciphertext,
        })
    }

    pub fn open(
        &self,
        private_key: &PortableExecutionPrivateKey,
        artifact: &EncryptedArtifact,
    ) -> ArtifactResult<Vec<u8>> {
        if artifact.schema_version != 1
            || artifact.suite != ExecutionHpkeSuite::X25519HkdfSha256Aes128GcmV1
        {
            return Err(ArtifactError::Invalid(
                "unsupported encrypted execution artifact".into(),
            ));
        }
        let encapsulated =
            <X25519HkdfSha256 as hpke::Kem>::EncappedKey::from_bytes(&artifact.encapsulated_key)
                .map_err(encryption)?;
        let mut receiver = setup_receiver::<AesGcm128, HkdfSha256, X25519HkdfSha256>(
            &OpModeR::Base,
            &private_key.0,
            &encapsulated,
            INFO,
        )
        .map_err(encryption)?;
        let plaintext = receiver
            .open(&artifact.ciphertext, &artifact.context.aad()?)
            .map_err(encryption)?;
        validate_context(&artifact.context, &plaintext)?;
        Ok(plaintext)
    }

    pub fn seal_run_key_with_entropy(
        &self,
        ephemeral_entropy: [u8; 32],
        recipient: &ExecutionRecipientKey,
        run_key: &RunKeyMaterial,
        run_identity: impl Into<String>,
        key_epoch: u32,
    ) -> ArtifactResult<RunKeyEnvelope> {
        let plaintext = run_key.expose_for_recipient_envelope();
        let encrypted = self.seal_with_entropy(
            ephemeral_entropy,
            recipient,
            run_key_context(run_identity.into(), plaintext, key_epoch),
            plaintext,
        )?;
        Ok(envelope(recipient, encrypted))
    }

    pub fn open_run_key(
        &self,
        private_key: &PortableExecutionPrivateKey,
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
        material_from_plaintext(self.open(private_key, &envelope.encrypted_key)?)
    }
}

struct ExactEntropy {
    bytes: [u8; 32],
    consumed: bool,
}

impl ExactEntropy {
    fn new(bytes: [u8; 32]) -> Self {
        Self {
            bytes,
            consumed: false,
        }
    }
}

impl TryRng for ExactEntropy {
    type Error = Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        let mut bytes = [0_u8; 4];
        self.try_fill_bytes(&mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        let mut bytes = [0_u8; 8];
        self.try_fill_bytes(&mut bytes)?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn try_fill_bytes(&mut self, destination: &mut [u8]) -> Result<(), Self::Error> {
        assert!(
            !self.consumed && destination.len() == self.bytes.len(),
            "X25519 HPKE requested an unexpected amount of entropy"
        );
        destination.copy_from_slice(&self.bytes);
        self.consumed = true;
        Ok(())
    }
}

impl TryCryptoRng for ExactEntropy {}

fn validate_context(context: &EncryptionContext, plaintext: &[u8]) -> ArtifactResult<()> {
    context.validate()?;
    if context.total_length != plaintext.len() as u64
        || context.object_digest != runmat_execution::Digest::sha256(plaintext)
    {
        return Err(ArtifactError::Identity(
            "encryption context does not identify the plaintext".into(),
        ));
    }
    Ok(())
}

fn encryption(error: impl std::fmt::Display) -> ArtifactError {
    ArtifactError::Encryption(error.to_string())
}
