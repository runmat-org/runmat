use std::path::Path;
use std::sync::Arc;

use runmat_execution::Digest;
use runmat_execution_artifact::archive::{read_bundle, ArchiveLimits};
use runmat_execution_artifact::ExecutionBundle;

use super::RemoteBundleReceipt;
use crate::materialized_project::MaterializedProject;

pub(super) struct InstalledBundle {
    pub(super) bundle: ExecutionBundle,
    pub(super) materialized_project: Arc<MaterializedProject>,
    pub(super) digest: Digest,
    pub(super) receipt: RemoteBundleReceipt,
}

pub(super) fn install(
    cache: Option<&Path>,
    bundle_digest: Digest,
    bytes: &[u8],
) -> Result<InstalledBundle, String> {
    if Digest::sha256(bytes) != bundle_digest {
        return Err("remote bundle digest does not match its bytes".into());
    }
    let bundle = read_bundle(bytes, ArchiveLimits::default())
        .map_err(|error| format!("remote bundle is invalid: {error}"))?;
    let installed = materialize(bundle_digest, bundle, bytes.len() as u64)?;
    if let Some(cache) = cache {
        super::bundle_cache::store(cache, bundle_digest, bytes)
            .map_err(|error| error.to_string())?;
    }
    Ok(installed)
}

pub(super) fn activate(cache: &Path, bundle_digest: Digest) -> Result<InstalledBundle, String> {
    let (bundle, stored_bytes) =
        super::bundle_cache::load(cache, bundle_digest).map_err(|error| error.to_string())?;
    materialize(bundle_digest, bundle, stored_bytes)
}

fn materialize(
    bundle_digest: Digest,
    bundle: ExecutionBundle,
    stored_bytes: u64,
) -> Result<InstalledBundle, String> {
    let materialized_project = Arc::new(
        MaterializedProject::from_bundle(&bundle)
            .map_err(|error| format!("remote bundle could not be materialized: {error}"))?,
    );
    let receipt = RemoteBundleReceipt {
        bundle_digest,
        bundle_identity: bundle.identity().map_err(|error| error.to_string())?,
        project_revision: bundle.manifest.project_revision.clone(),
        stored_bytes,
    };
    Ok(InstalledBundle {
        bundle,
        materialized_project,
        digest: bundle_digest,
        receipt,
    })
}
