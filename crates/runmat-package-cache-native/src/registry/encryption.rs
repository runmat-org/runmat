use aes_gcm::aead::{Aead as _, KeyInit as _, Payload};
use aes_gcm::{Aes256Gcm, Nonce};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use hkdf::Hkdf;
use p256::ecdh::{diffie_hellman, EphemeralSecret};
use p256::elliptic_curve::sec1::ToEncodedPoint as _;
use p256::{PublicKey, SecretKey};
use rand_core::{OsRng, RngCore as _};
use runmat_package::{
    ContentDigest, EncryptedArtifactMetadata, KeyEnvelopeAlgorithm, PackageKeyEnvelope,
    RecipientEncryptionKey, RecipientKeyAlgorithm, RegistrySourceId, AES_256_GCM_NONCE_BYTE_LEN,
    PACKAGE_KEY_ENVELOPE_SCHEMA_VERSION,
};
use sha2::Sha256;
use zeroize::Zeroizing;

const CONTENT_KEY_BYTE_LEN: usize = 32;
const KEY_WRAP_INFO: &[u8] = b"runmat-package-key-wrap-v1";

pub struct RecipientKeyPair {
    id: String,
    secret: Zeroizing<[u8; 32]>,
}

impl RecipientKeyPair {
    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn generate(id: impl Into<String>) -> Result<Self, String> {
        let secret = SecretKey::random(&mut OsRng);
        Self::from_secret_bytes(id, secret.to_bytes().into())
    }

    pub fn from_secret_bytes(id: impl Into<String>, secret: [u8; 32]) -> Result<Self, String> {
        SecretKey::from_slice(&secret)
            .map_err(|_| "recipient encryption secret key is invalid".to_string())?;
        let value = Self {
            id: id.into(),
            secret: Zeroizing::new(secret),
        };
        value.public_key()?.validate()?;
        Ok(value)
    }

    pub fn public_key(&self) -> Result<RecipientEncryptionKey, String> {
        let secret = self.secret_key()?;
        let encoded = secret.public_key().to_encoded_point(false);
        let bytes = encoded.as_bytes();
        Ok(RecipientEncryptionKey {
            id: self.id.clone(),
            algorithm: RecipientKeyAlgorithm::P256,
            public_key: URL_SAFE_NO_PAD.encode(bytes),
            fingerprint: ContentDigest::sha256(bytes),
        })
    }

    pub fn secret_bytes(&self) -> Zeroizing<[u8; 32]> {
        Zeroizing::new(*self.secret)
    }

    fn secret_key(&self) -> Result<SecretKey, String> {
        SecretKey::from_slice(self.secret.as_ref())
            .map_err(|_| "recipient encryption secret key is invalid".to_string())
    }
}

pub trait PrivateArtifactDecryptor: Send + Sync {
    fn decrypt(
        &self,
        source: &RegistrySourceId,
        ciphertext: &[u8],
        metadata: &EncryptedArtifactMetadata,
        envelopes: &[PackageKeyEnvelope],
    ) -> Result<Zeroizing<Vec<u8>>, String>;
}

#[derive(Default)]
pub struct InMemoryRecipientKeyRing {
    keys: Vec<RecipientKeyPair>,
}

impl InMemoryRecipientKeyRing {
    pub fn new(keys: Vec<RecipientKeyPair>) -> Self {
        Self { keys }
    }

    pub fn insert(&mut self, key: RecipientKeyPair) {
        if let Some(existing) = self.keys.iter_mut().find(|value| value.id() == key.id()) {
            *existing = key;
        } else {
            self.keys.push(key);
        }
    }
}

impl PrivateArtifactDecryptor for InMemoryRecipientKeyRing {
    fn decrypt(
        &self,
        source: &RegistrySourceId,
        ciphertext: &[u8],
        metadata: &EncryptedArtifactMetadata,
        envelopes: &[PackageKeyEnvelope],
    ) -> Result<Zeroizing<Vec<u8>>, String> {
        for envelope in envelopes {
            if let Some(key) = self
                .keys
                .iter()
                .find(|key| key.id() == envelope.recipient_key_id)
            {
                return decrypt_private_artifact(source, ciphertext, metadata, envelope, key);
            }
        }
        Err("no local recipient key matches an authorized package key envelope".to_string())
    }
}

pub struct EncryptedArtifactBundle {
    pub ciphertext: Vec<u8>,
    pub metadata: EncryptedArtifactMetadata,
    pub envelopes: Vec<PackageKeyEnvelope>,
}

pub fn encrypt_private_artifact(
    source: &RegistrySourceId,
    plaintext: &[u8],
    key_version: u64,
    recipients: &[RecipientEncryptionKey],
) -> Result<EncryptedArtifactBundle, String> {
    if recipients.is_empty() {
        return Err("private artifact requires at least one recipient key".to_string());
    }
    let mut nonce = [0u8; AES_256_GCM_NONCE_BYTE_LEN];
    OsRng.fill_bytes(&mut nonce);
    let metadata = EncryptedArtifactMetadata::new(key_version, plaintext, nonce, source)?;
    let aad = metadata.aad_bytes(source)?;
    let mut content_key = Zeroizing::new([0u8; CONTENT_KEY_BYTE_LEN]);
    OsRng.fill_bytes(content_key.as_mut());
    let ciphertext = cipher(content_key.as_ref())?
        .encrypt(
            Nonce::from_slice(&nonce),
            Payload {
                msg: plaintext,
                aad: &aad,
            },
        )
        .map_err(|_| "private artifact encryption failed".to_string())?;

    let mut sorted = recipients.to_vec();
    sorted.sort_by(|left, right| left.id.cmp(&right.id));
    if sorted.windows(2).any(|pair| pair[0].id == pair[1].id) {
        return Err("private artifact recipient keys must be unique".to_string());
    }
    let envelopes = sorted
        .iter()
        .map(|recipient| wrap_content_key(&content_key, recipient, &metadata))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(EncryptedArtifactBundle {
        ciphertext,
        metadata,
        envelopes,
    })
}

pub fn decrypt_private_artifact(
    source: &RegistrySourceId,
    ciphertext: &[u8],
    metadata: &EncryptedArtifactMetadata,
    envelope: &PackageKeyEnvelope,
    recipient: &RecipientKeyPair,
) -> Result<Zeroizing<Vec<u8>>, String> {
    if ContentDigest::sha256(ciphertext) != source.artifact_digest {
        return Err("private artifact ciphertext digest is invalid".to_string());
    }
    metadata.validate(source)?;
    let public = recipient.public_key()?;
    envelope.validate(&public, metadata)?;
    let ephemeral = PublicKey::from_sec1_bytes(&envelope.decoded_ephemeral_public_key()?)
        .map_err(|_| "package key envelope ephemeral public key is invalid".to_string())?;
    let secret = recipient.secret_key()?;
    let shared = diffie_hellman(secret.to_nonzero_scalar(), ephemeral.as_affine());
    let wrapping_key = derive_wrapping_key(shared.raw_secret_bytes(), &envelope.context_digest)?;
    let wrapped_key = envelope.decoded_wrapped_key()?;
    let context = envelope.context_bytes(metadata)?;
    let content_key = Zeroizing::new(
        cipher(wrapping_key.as_ref())?
            .decrypt(
                Nonce::from_slice(&envelope.decoded_nonce()?),
                Payload {
                    msg: &wrapped_key,
                    aad: &context,
                },
            )
            .map_err(|_| "package content key envelope is corrupt or unauthorized".to_string())?,
    );
    if content_key.len() != CONTENT_KEY_BYTE_LEN {
        return Err("unwrapped package content key has an invalid length".to_string());
    }
    let plaintext = Zeroizing::new(
        cipher(content_key.as_slice())?
            .decrypt(
                Nonce::from_slice(&metadata.decoded_nonce()?),
                Payload {
                    msg: ciphertext,
                    aad: &metadata.aad_bytes(source)?,
                },
            )
            .map_err(|_| "private artifact is corrupt or unauthorized".to_string())?,
    );
    if plaintext.len() as u64 != metadata.plaintext_byte_len
        || ContentDigest::sha256(plaintext.as_slice()) != metadata.plaintext_digest
    {
        return Err("private artifact plaintext integrity check failed".to_string());
    }
    Ok(plaintext)
}

fn wrap_content_key(
    content_key: &[u8; CONTENT_KEY_BYTE_LEN],
    recipient: &RecipientEncryptionKey,
    metadata: &EncryptedArtifactMetadata,
) -> Result<PackageKeyEnvelope, String> {
    recipient.validate()?;
    let recipient_public = PublicKey::from_sec1_bytes(
        &URL_SAFE_NO_PAD
            .decode(&recipient.public_key)
            .map_err(|_| "recipient encryption public key is invalid".to_string())?,
    )
    .map_err(|_| "recipient encryption public key is invalid".to_string())?;
    let ephemeral = EphemeralSecret::random(&mut OsRng);
    let ephemeral_public = PublicKey::from(&ephemeral);
    let ephemeral_encoded = ephemeral_public.to_encoded_point(false);
    let shared = ephemeral.diffie_hellman(&recipient_public);
    let mut envelope = PackageKeyEnvelope {
        schema_version: PACKAGE_KEY_ENVELOPE_SCHEMA_VERSION,
        algorithm: KeyEnvelopeAlgorithm::P256HkdfSha256Aes256Gcm,
        recipient_key_id: recipient.id.clone(),
        recipient_key_fingerprint: recipient.fingerprint.clone(),
        ephemeral_public_key: URL_SAFE_NO_PAD.encode(ephemeral_encoded.as_bytes()),
        nonce: String::new(),
        wrapped_key: String::new(),
        context_digest: ContentDigest::sha256([]),
    };
    envelope.context_digest = ContentDigest::sha256(envelope.context_bytes(metadata)?);
    let wrapping_key = derive_wrapping_key(shared.raw_secret_bytes(), &envelope.context_digest)?;
    let mut nonce = [0u8; AES_256_GCM_NONCE_BYTE_LEN];
    OsRng.fill_bytes(&mut nonce);
    envelope.nonce = URL_SAFE_NO_PAD.encode(nonce);
    envelope.wrapped_key = URL_SAFE_NO_PAD.encode(
        cipher(wrapping_key.as_ref())?
            .encrypt(
                Nonce::from_slice(&nonce),
                Payload {
                    msg: content_key,
                    aad: &envelope.context_bytes(metadata)?,
                },
            )
            .map_err(|_| "package content key wrapping failed".to_string())?,
    );
    envelope.validate(recipient, metadata)?;
    Ok(envelope)
}

fn derive_wrapping_key(
    shared_secret: impl AsRef<[u8]>,
    context_digest: &ContentDigest,
) -> Result<Zeroizing<[u8; CONTENT_KEY_BYTE_LEN]>, String> {
    let hkdf = Hkdf::<Sha256>::new(Some(context_digest.bytes()), shared_secret.as_ref());
    let mut key = Zeroizing::new([0u8; CONTENT_KEY_BYTE_LEN]);
    hkdf.expand(KEY_WRAP_INFO, key.as_mut())
        .map_err(|_| "package key derivation failed".to_string())?;
    Ok(key)
}

fn cipher(key: &[u8]) -> Result<Aes256Gcm, String> {
    Aes256Gcm::new_from_slice(key).map_err(|_| "AES-256-GCM key is invalid".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_package::{
        CanonicalPackageId, PackageVersion, RegistryId, RegistryOrigin, RegistryReleaseId,
    };

    fn source(artifact: ContentDigest) -> RegistrySourceId {
        RegistrySourceId {
            registry_origin: RegistryOrigin::new("https://packages.runmat.test").unwrap(),
            package: CanonicalPackageId::new(RegistryId::default(), "acme", "private").unwrap(),
            release: RegistryReleaseId::new("rel_0123456789abcdef0123456789abcdef").unwrap(),
            version: "1.2.3".parse::<PackageVersion>().unwrap(),
            release_digest: ContentDigest::sha256("release"),
            artifact_digest: artifact,
            tree_digest: ContentDigest::sha256("tree"),
        }
    }

    #[test]
    fn encrypted_artifact_round_trip_rejects_wrong_recipient_and_tampering() {
        let alice = RecipientKeyPair::from_secret_bytes("pkr_alice", [3; 32]).unwrap();
        let bob = RecipientKeyPair::from_secret_bytes("pkr_bob", [4; 32]).unwrap();
        let plaintext = br#"{"schemaVersion":1,"entries":[]}"#;
        let source_before_encryption = source(ContentDigest::sha256(plaintext));
        let bundle = encrypt_private_artifact(
            &source_before_encryption,
            plaintext,
            7,
            &[alice.public_key().unwrap()],
        )
        .unwrap();
        let encrypted_source = source(ContentDigest::sha256(&bundle.ciphertext));

        assert_eq!(
            decrypt_private_artifact(
                &encrypted_source,
                &bundle.ciphertext,
                &bundle.metadata,
                &bundle.envelopes[0],
                &alice,
            )
            .unwrap()
            .as_slice(),
            plaintext
        );
        assert!(decrypt_private_artifact(
            &encrypted_source,
            &bundle.ciphertext,
            &bundle.metadata,
            &bundle.envelopes[0],
            &bob,
        )
        .is_err());
        let mut corrupt = bundle.ciphertext;
        corrupt[0] ^= 1;
        assert!(decrypt_private_artifact(
            &encrypted_source,
            &corrupt,
            &bundle.metadata,
            &bundle.envelopes[0],
            &alice,
        )
        .is_err());
    }

    #[test]
    fn encrypted_artifact_metadata_aad_is_cross_host_stable() {
        let source = source(ContentDigest::sha256("ciphertext"));
        let metadata = EncryptedArtifactMetadata::new(9, b"plaintext", [7; 12], &source).unwrap();
        assert_eq!(
            metadata.aad_digest.to_string(),
            "sha256:c4a5d7fdd5c0e335e325a1383846508a9f0b6617ed28db742d7cbb1a22e9e037"
        );
    }
}
