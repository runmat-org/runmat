use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use zeroize::Zeroizing;

use super::{decrypt_private_artifact, PrivateArtifactDecryptor, RecipientKeyPair};

pub struct OsCredentialPrivateArtifactDecryptor;

impl PrivateArtifactDecryptor for OsCredentialPrivateArtifactDecryptor {
    fn decrypt(
        &self,
        source: &runmat_package::RegistrySourceId,
        ciphertext: &[u8],
        metadata: &runmat_package::EncryptedArtifactMetadata,
        envelopes: &[runmat_package::PackageKeyEnvelope],
    ) -> Result<Zeroizing<Vec<u8>>, String> {
        for envelope in envelopes {
            let Some(key) = load_key(source.registry_origin.as_str(), &envelope.recipient_key_id)?
            else {
                continue;
            };
            return decrypt_private_artifact(source, ciphertext, metadata, envelope, &key);
        }
        Err("no matching private package key is available in the OS credential store".into())
    }
}

fn load_key(origin: &str, key_id: &str) -> Result<Option<RecipientKeyPair>, String> {
    let account = credential_account(origin, key_id);
    let entry = keyring::Entry::new("runmat", &account)
        .map_err(|error| format!("failed to open the OS credential store: {error}"))?;
    let encoded = match entry.get_password() {
        Ok(value) => Zeroizing::new(value),
        Err(keyring::Error::NoEntry) => return Ok(None),
        Err(error) => return Err(format!("failed to access the OS credential store: {error}")),
    };
    let decoded = Zeroizing::new(
        URL_SAFE_NO_PAD
            .decode(encoded.as_bytes())
            .map_err(|_| "stored private package key is corrupt".to_string())?,
    );
    let secret: [u8; 32] = decoded
        .as_slice()
        .try_into()
        .map_err(|_| "stored private package key has an invalid length".to_string())?;
    RecipientKeyPair::from_secret_bytes(key_id, secret).map(Some)
}

fn credential_account(origin: &str, key_id: &str) -> String {
    let origin_digest = runmat_package::ContentDigest::sha256(origin);
    format!("package-key:{origin_digest}:{key_id}")
}
