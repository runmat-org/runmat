use super::publication_manifest::PreparedPublication;
use anyhow::{bail, Context, Result};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use runmat_package::{
    ContentDigest, PackageVersion, RegistryOrigin, RegistryReleaseId, RegistrySourceId,
};
use runmat_package_cache_native::registry::{encrypt_private_artifact, EncryptedArtifactBundle};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct PrivateRetryState {
    schema_version: u32,
    registry_origin: String,
    key_version: u64,
    plaintext_digest: ContentDigest,
    tree_digest: ContentDigest,
    recipient_fingerprints: Vec<ContentDigest>,
    ciphertext: String,
    pub(super) artifact_digest: ContentDigest,
    pub(super) metadata: runmat_package::EncryptedArtifactMetadata,
    pub(super) envelopes: Vec<runmat_package::PackageKeyEnvelope>,
}

pub(super) fn load_or_create_encrypted(
    path: &Path,
    origin: &str,
    prepared: &PreparedPublication,
    key_version: u64,
    recipients: &[runmat_package::RecipientEncryptionKey],
) -> Result<PrivateRetryState> {
    let mut recipient_fingerprints = recipients
        .iter()
        .map(|key| key.fingerprint.clone())
        .collect::<Vec<_>>();
    recipient_fingerprints.sort();
    match std::fs::read(path) {
        Ok(bytes) => {
            let state: PrivateRetryState = serde_json::from_slice(&bytes)
                .context("encrypted publication retry state is corrupt")?;
            if state.schema_version != 1
                || state.registry_origin != origin
                || state.key_version != key_version
                || state.plaintext_digest != prepared.bundle.artifact_digest
                || state.tree_digest != prepared.bundle.tree_digest
                || state.recipient_fingerprints != recipient_fingerprints
                || ContentDigest::sha256(
                    &URL_SAFE_NO_PAD
                        .decode(&state.ciphertext)
                        .context("encrypted publication retry ciphertext is corrupt")?,
                ) != state.artifact_digest
            {
                bail!(
                    "encrypted publication retry state differs from this release; reject the prior staged publication before removing {}",
                    path.display()
                );
            }
            return Ok(state);
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error)
                .with_context(|| format!("failed to read private retry state {}", path.display()))
        }
    }
    let source = encryption_source(origin, prepared)?;
    let encrypted = encrypt_private_artifact(
        &source,
        &prepared.bundle.artifact_bytes,
        key_version,
        recipients,
    )
    .map_err(anyhow::Error::msg)?;
    let state = retry_state(
        encrypted,
        origin,
        prepared.bundle.artifact_digest.clone(),
        prepared.bundle.tree_digest.clone(),
        recipient_fingerprints,
    );
    write_private_state(path, &state)?;
    Ok(state)
}

fn retry_state(
    encrypted: EncryptedArtifactBundle,
    registry_origin: &str,
    plaintext_digest: ContentDigest,
    tree_digest: ContentDigest,
    recipient_fingerprints: Vec<ContentDigest>,
) -> PrivateRetryState {
    let artifact_digest = ContentDigest::sha256(&encrypted.ciphertext);
    PrivateRetryState {
        schema_version: 1,
        registry_origin: registry_origin.to_string(),
        key_version: encrypted.metadata.key_version,
        plaintext_digest,
        tree_digest,
        recipient_fingerprints,
        ciphertext: URL_SAFE_NO_PAD.encode(encrypted.ciphertext),
        artifact_digest,
        metadata: encrypted.metadata,
        envelopes: encrypted.envelopes,
    }
}

fn encryption_source(origin: &str, prepared: &PreparedPublication) -> Result<RegistrySourceId> {
    Ok(RegistrySourceId {
        registry_origin: RegistryOrigin::new(origin)?,
        package: prepared
            .manifest
            .canonical_id
            .clone()
            .context("publishing requires package.organization")?,
        release: RegistryReleaseId::new("rel_00000000000000000000000000000000")?,
        version: prepared.release_manifest.version.clone(),
        release_digest: ContentDigest::sha256([]),
        artifact_digest: prepared.bundle.artifact_digest.clone(),
        tree_digest: prepared.bundle.tree_digest.clone(),
    })
}

pub(super) fn private_state_path(
    root: &Path,
    package_id: &str,
    version: &PackageVersion,
) -> Result<PathBuf> {
    if package_id.is_empty()
        || package_id.len() > 128
        || !package_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        bail!("registry package ID is invalid");
    }
    Ok(root
        .join(".runmat")
        .join("publications")
        .join(format!("{package_id}-{version}.json")))
}

fn write_private_state(path: &Path, state: &PrivateRetryState) -> Result<()> {
    let parent = path.parent().context("private retry state has no parent")?;
    std::fs::create_dir_all(parent)
        .with_context(|| format!("failed to create {}", parent.display()))?;
    let mut temporary =
        tempfile::NamedTempFile::new_in(parent).context("failed to stage private retry state")?;
    use std::io::Write as _;
    temporary.write_all(&serde_json::to_vec(state)?)?;
    temporary.as_file().sync_all()?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        temporary
            .as_file()
            .set_permissions(std::fs::Permissions::from_mode(0o600))?;
    }
    temporary
        .persist(path)
        .map_err(|error| error.error)
        .with_context(|| format!("failed to persist {}", path.display()))?;
    Ok(())
}

impl PrivateRetryState {
    pub(super) fn ciphertext(&self) -> Result<Vec<u8>> {
        URL_SAFE_NO_PAD
            .decode(&self.ciphertext)
            .context("encrypted publication retry ciphertext is corrupt")
    }
}
