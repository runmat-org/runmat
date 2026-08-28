use super::private_keys::configured_registry_origin;
use super::publication_manifest::{publication_metadata, PreparedPublication};
use super::publication_retry::{load_or_create_encrypted, private_state_path};
use super::registry_transport::registry_client;
use crate::cli::{PackageInspectArgs, PackagePublishArgs};
use anyhow::{bail, Context, Result};
use runmat_server_client::packages::{
    AttachKeyEnvelopesRequest, KeyEnvelopeRequest, PublicationArtifactRequest,
    StagePublicationRequest,
};

const PACKAGE_MEDIA_TYPE: &str = "application/vnd.runmat.package";
const ENCRYPTED_PACKAGE_MEDIA_TYPE: &str = "application/vnd.runmat.package.encrypted";

pub(super) fn inspect(args: &PackageInspectArgs) -> Result<()> {
    let prepared = PreparedPublication::build(args)?;
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "manifest": prepared.release_manifest,
                "inventory": prepared.bundle.inventory,
            }))?
        );
    } else {
        println!("package: {}", prepared.release_manifest.package);
        println!("version: {}", prepared.release_manifest.version);
        println!("files: {}", prepared.bundle.inventory.file_count);
        println!("bytes: {}", prepared.bundle.inventory.total_bytes);
        println!("tree: {}", prepared.bundle.tree_digest);
        println!("artifact: {}", prepared.bundle.artifact_digest);
        println!("inventory: {}", prepared.bundle.inventory.digest);
    }
    Ok(())
}

pub(super) async fn publish(args: &PackagePublishArgs) -> Result<()> {
    let prepared = PreparedPublication::build(&args.artifact)?;
    let origin = configured_registry_origin(args.registry.as_deref())?;
    let client = registry_client(&origin).await.map_err(anyhow::Error::msg)?;
    let (artifact_bytes, artifact_digest, media_type, encryption, envelopes, state_path) = if args
        .private_package
    {
        let recipients = client
            .publication_recipient_keys(&args.org_id, &args.package_id)
            .await
            .context("failed to list private package recipient keys")?
            .into_iter()
            .filter(|key| key.revoked_at.is_none())
            .map(|key| {
                Ok(runmat_package::RecipientEncryptionKey {
                    id: key.id,
                    algorithm: match key.algorithm.as_str() {
                        "p256" => runmat_package::RecipientKeyAlgorithm::P256,
                        _ => bail!("registry returned an unsupported recipient key algorithm"),
                    },
                    public_key: key.public_key,
                    fingerprint: key.fingerprint.parse()?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if recipients.is_empty() {
            bail!(
                "private publication has no active recipient keys; recipients must register keys before publication"
            );
        }
        let state_path = private_state_path(
            &prepared.root,
            &args.package_id,
            &prepared.release_manifest.version,
        )?;
        let encrypted = load_or_create_encrypted(
            &state_path,
            &origin,
            &prepared,
            args.key_version,
            &recipients,
        )?;
        let ciphertext = encrypted.ciphertext()?;
        (
            ciphertext,
            encrypted.artifact_digest,
            ENCRYPTED_PACKAGE_MEDIA_TYPE,
            Some(encrypted.metadata),
            encrypted.envelopes,
            Some(state_path),
        )
    } else {
        (
            prepared.bundle.artifact_bytes.clone(),
            prepared.bundle.artifact_digest.clone(),
            PACKAGE_MEDIA_TYPE,
            None,
            Vec::new(),
            None,
        )
    };
    let request = StagePublicationRequest {
        version: prepared.release_manifest.version.to_string(),
        artifact: PublicationArtifactRequest {
            digest: artifact_digest.to_string(),
            tree_digest: prepared.bundle.tree_digest.to_string(),
            byte_len: artifact_bytes.len() as u64,
            media_type: media_type.to_string(),
            encryption,
        },
        metadata: publication_metadata(&prepared.manifest, &prepared.bundle)?,
        idempotency_key: format!(
            "runmat-publish-{}-{}",
            prepared.release_manifest.version, artifact_digest
        ),
    };
    let staged = client
        .stage_publication(&args.org_id, &args.package_id, &request)
        .await
        .context("failed to stage package publication")?;
    let mut status = staged.status;
    if status == "staged" {
        let upload_url = staged
            .upload_url
            .as_deref()
            .context("staged publication response omitted its upload URL")?;
        client
            .upload_publication_artifact(upload_url, media_type, artifact_bytes)
            .await
            .context("failed to upload package artifact")?;
        status = client
            .verify_publication(&args.org_id, &args.package_id, &staged.id)
            .await
            .context("failed to verify package publication")?
            .status;
    }
    if status == "verified" {
        if !envelopes.is_empty() {
            client
                .attach_publication_key_envelopes(
                    &args.org_id,
                    &args.package_id,
                    &staged.id,
                    &AttachKeyEnvelopesRequest {
                        envelopes: envelopes
                            .into_iter()
                            .map(KeyEnvelopeRequest::from)
                            .collect(),
                    },
                )
                .await
                .context("failed to attach private package key envelopes")?;
        }
        status = client
            .approve_publication(&args.org_id, &args.package_id, &staged.id)
            .await
            .context("failed to approve package publication")?
            .status;
    }
    if status != "approved" && status != "finalized" {
        bail!("publication stopped in unexpected state `{status}`");
    }
    let finalized = client
        .finalize_publication(&args.org_id, &args.package_id, &staged.id)
        .await
        .context("failed to finalize package publication")?;
    remove_retry_state(state_path)?;
    println!(
        "{} {} {}",
        finalized.release_id, finalized.version, finalized.release_digest
    );
    Ok(())
}

fn remove_retry_state(path: Option<std::path::PathBuf>) -> Result<()> {
    let Some(path) = path else {
        return Ok(());
    };
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| {
            format!(
                "release finalized, but encrypted retry state could not be removed from {}",
                path.display()
            )
        }),
    }
}
