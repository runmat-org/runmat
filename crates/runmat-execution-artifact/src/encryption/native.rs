use hpke::{
    aead::AesGcm128,
    kdf::HkdfSha256,
    kem::{Kem as _, X25519HkdfSha256},
    setup_receiver, setup_sender, Deserializable, OpModeR, OpModeS, Serializable,
};

use super::{
    EncryptedArtifact, EncryptionContext, ExecutionEncryptionProvider, ExecutionHpkeSuite,
    ExecutionRecipientKey,
};
use crate::{ArtifactError, ArtifactResult};

const INFO: &[u8] = b"runmat-execution-artifact-hpke-v1";

#[derive(Clone)]
pub struct NativeExecutionPrivateKey(<X25519HkdfSha256 as hpke::Kem>::PrivateKey);

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeExecutionEncryption;

impl NativeExecutionEncryption {
    pub fn generate_recipient(
        &self,
        fingerprint: impl Into<String>,
        valid_after_unix_millis: u64,
        valid_before_unix_millis: u64,
    ) -> ArtifactResult<(ExecutionRecipientKey, NativeExecutionPrivateKey)> {
        let (private, public) = X25519HkdfSha256::gen_keypair();
        let recipient = ExecutionRecipientKey {
            suite: ExecutionHpkeSuite::X25519HkdfSha256Aes128GcmV1,
            public_key: public.to_bytes().to_vec(),
            fingerprint: fingerprint.into(),
            valid_after_unix_millis,
            valid_before_unix_millis,
            custodian_uri: None,
        };
        recipient.validate()?;
        Ok((recipient, NativeExecutionPrivateKey(private)))
    }
}

impl ExecutionEncryptionProvider for NativeExecutionEncryption {
    type PrivateKey = NativeExecutionPrivateKey;

    fn seal(
        &self,
        recipient: &ExecutionRecipientKey,
        context: EncryptionContext,
        plaintext: &[u8],
    ) -> ArtifactResult<EncryptedArtifact> {
        recipient.validate()?;
        context.validate()?;
        if context.total_length != plaintext.len() as u64
            || context.object_digest != runmat_execution::Digest::sha256(plaintext)
        {
            return Err(ArtifactError::Identity(
                "encryption context does not identify the plaintext".into(),
            ));
        }
        let public = <X25519HkdfSha256 as hpke::Kem>::PublicKey::from_bytes(&recipient.public_key)
            .map_err(encryption)?;
        let (encapsulated, mut sender) =
            setup_sender::<AesGcm128, HkdfSha256, X25519HkdfSha256>(&OpModeS::Base, &public, INFO)
                .map_err(encryption)?;
        let aad = context.aad()?;
        let ciphertext = sender.seal(plaintext, &aad).map_err(encryption)?;
        Ok(EncryptedArtifact {
            schema_version: 1,
            suite: recipient.suite,
            context,
            encapsulated_key: encapsulated.to_bytes().to_vec(),
            ciphertext,
        })
    }

    fn open(
        &self,
        private_key: &Self::PrivateKey,
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
        let aad = artifact.context.aad()?;
        let plaintext = receiver
            .open(&artifact.ciphertext, &aad)
            .map_err(encryption)?;
        artifact.context.validate()?;
        if artifact.context.total_length != plaintext.len() as u64
            || artifact.context.object_digest != runmat_execution::Digest::sha256(&plaintext)
        {
            return Err(ArtifactError::Identity(
                "decrypted artifact does not match its authenticated identity".into(),
            ));
        }
        Ok(plaintext)
    }
}

fn encryption(error: impl std::fmt::Display) -> ArtifactError {
    ArtifactError::Encryption(error.to_string())
}
