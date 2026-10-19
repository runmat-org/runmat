//! Private immutable records become visible only after one atomic commit marker.
use super::*;
mod cleanup;
mod inventory;
pub(in crate::analysis::storage) mod transaction;

pub(in crate::analysis::storage) fn directory(
    store: &FilesystemAnalysisArtifactStore,
    identity: Digest,
) -> PathBuf {
    store.root.join("publications").join(digest_hex(identity))
}
fn put(
    store: &access::FilesystemAccess<'_>,
    path: impl AsRef<Path>,
    bytes: &[u8],
) -> Result<(), AnalysisPublicationError> {
    match store
        .atomic_create_new(path, bytes)
        .map_err(AnalysisPublicationError::from_io)?
    {
        AtomicCreateResult::Published { .. } => Ok(()),
        AtomicCreateResult::Existing(existing) if existing == bytes => Ok(()),
        AtomicCreateResult::Existing(_) => Err("immutable publication content collision".into()),
    }
}
pub(super) fn descriptor(
    store: &access::FilesystemAccess<'_>,
    identity: Digest,
    maximum_bytes: u64,
) -> Result<Option<PublicationDescriptor>, AnalysisPublicationError> {
    let Some((value, _)) = descriptor_with_length(store, identity, maximum_bytes)
        .map_err(AnalysisPublicationError::from)?
    else {
        return Ok(None);
    };
    Ok(Some(value))
}
pub(super) fn descriptor_with_length(
    store: &access::FilesystemAccess<'_>,
    identity: Digest,
    maximum_bytes: u64,
) -> Result<Option<(PublicationDescriptor, u64)>, AssessmentStoreError> {
    let root = directory(store, identity);
    for path in [&store.root, &store.root.join("publications"), &root] {
        if !store
            .owned_directory_exists(path, "publication directory")
            .map_err(AssessmentStoreError::from)?
        {
            return Ok(None);
        }
    }
    let Some(bytes) = store.read_bounded_artifact(&root.join("commit.json"), maximum_bytes)? else {
        return Ok(None);
    };
    let value: PublicationDescriptor = store
        .decode_metadata(&bytes)
        .map_err(|e| AssessmentStoreError::from(e))?;
    if value.schema_version != 1
        || value.records.is_empty()
        || value.identity().map_err(AssessmentStoreError::Invalid)? != identity
    {
        return Err(AssessmentStoreError::Invalid(
            "malformed publication commit identity".into(),
        ));
    }
    for id in value.records.keys() {
        validate_run_id_token(id).map_err(AssessmentStoreError::Invalid)?;
    }
    Ok(Some((value, bytes.len() as u64)))
}
pub(in crate::analysis::storage) fn commit(
    store: &access::FilesystemAccess<'_>,
    prepared: PreparedAnalysisPublication,
) -> Result<AnalysisPublicationReceipt, AnalysisPublicationError> {
    prepared.validate()?;
    let identity = prepared.descriptor.identity()?;
    cleanup::uncommitted_roots(store, Some(identity))?;
    let root = directory(store, identity);
    let manifest_path = root.join("manifest.json");
    let receipt = prepared.descriptor.receipt(Some(manifest_path.clone()))?;
    // Plan the complete retained inventory before allocating private files or evicting.
    let evictions = transaction::plan(store, &prepared.runs, Some(identity))?;
    if reclamation::sweep(store, prepared.chunks.keys().copied())?
        == reclamation::SweepOutcome::DeferredLegacyInventory
    {
        log::debug!("analysis object reclamation deferred: unsupported legacy inventory");
    }
    if let Some(existing) = descriptor(store, identity, prepared.maximum_bytes)? {
        if existing != prepared.descriptor {
            return Err("publication identity contains different content".into());
        }
        store.visit_publication_records(|_| Ok(()))?;
        load(store, identity, prepared.maximum_bytes)?
            .ok_or("committed publication manifest absent")?;
        for (id, expected) in &existing.observations {
            let bytes = store
                .read_bounded_artifact(
                    &root.join("observations").join(format!("{id}.json")),
                    prepared.maximum_bytes,
                )
                .map_err(AnalysisPublicationError::from)?
                .ok_or("committed publication observation absent")?;
            let attachment: observations::ObservationAttachment = store.decode_metadata(&bytes)?;
            if canonical_digest("analysis-publication-observation", &attachment)? != *expected {
                return Err("observation differs from its committed publication".into());
            }
        }
        transaction::transition(store, None, evictions)
            .map_err(|e| e.with_publication(receipt.clone()))?;
        return Ok(receipt);
    }
    store.ensure_runs_directory()?;
    store.ensure_objects_directory()?;
    store.ensure_owned_directory(&store.root.join("publications"), "publication directory")?;
    store.ensure_owned_directory(&root, "publication root")?;
    store.ensure_owned_directory(&root.join("runs"), "publication records")?;
    store.ensure_owned_directory(&root.join("observations"), "publication observations")?;
    for id in prepared.runs.keys() {
        store.reserve_study(id, identity)?;
    }
    for (digest, bytes) in &prepared.chunks {
        if Digest::sha256(bytes.as_ref()) != *digest {
            return Err("publication chunk digest differs".into());
        }
        put(store, store.object_path(*digest), bytes)?;
    }
    for (id, record) in &prepared.runs {
        let path = root.join("runs").join(format!("{id}.json"));
        let bytes = resources::json(&record_value(record)?, prepared.maximum_bytes)?;
        match store
            .atomic_create_new(&path, &bytes)
            .map_err(AnalysisPublicationError::from_io)?
        {
            AtomicCreateResult::Published { .. } => {}
            AtomicCreateResult::Existing(existing) => {
                if record_digest(&store.decode_record(&existing, id)?)? != record_digest(record)? {
                    return Err("publication record collision".into());
                }
            }
        }
    }
    for (id, attachment) in &prepared.observations {
        put(
            store,
            root.join("observations").join(format!("{id}.json")),
            &resources::json(attachment, prepared.maximum_bytes)?,
        )?;
    }
    put(
        store,
        &manifest_path,
        &resources::json(&prepared.manifest, prepared.maximum_bytes)?,
    )?;
    if crate::interrupt::is_cancelled() {
        return Err("study publication was cancelled".into());
    }
    // The shared durable journal owns visibility and recovery of the whole edit set.
    transaction::transition(
        store,
        Some((
            root.join("commit.json"),
            resources::json(&prepared.descriptor, prepared.maximum_bytes)?,
        )),
        evictions,
    )
    .map_err(|e| e.with_publication(receipt.clone()))?;
    Ok(receipt)
}
pub(in crate::analysis::storage) fn load(
    store: &access::FilesystemAccess<'_>,
    identity: Digest,
    maximum_bytes: u64,
) -> Result<Option<serde_json::Value>, AnalysisPublicationError> {
    load_bounded(store, identity, maximum_bytes).map_err(Into::into)
}
pub(in crate::analysis::storage) fn load_bounded(
    store: &access::FilesystemAccess<'_>,
    identity: Digest,
    maximum_bytes: u64,
) -> Result<Option<serde_json::Value>, AssessmentStoreError> {
    let Some((descriptor, descriptor_bytes)) =
        descriptor_with_length(store, identity, maximum_bytes)?
    else {
        return Ok(None);
    };
    let remaining = maximum_bytes
        .checked_sub(descriptor_bytes)
        .ok_or(AssessmentStoreError::ResourceLimit)?;
    let bytes = store
        .read_bounded_artifact(&directory(store, identity).join("manifest.json"), remaining)?
        .ok_or_else(|| {
            AssessmentStoreError::Invalid("committed publication manifest absent".into())
        })?;
    let manifest = store
        .decode_metadata(&bytes)
        .map_err(|e| AssessmentStoreError::from(e))?;
    validate_manifest(identity, &descriptor, &manifest).map_err(AssessmentStoreError::Invalid)?;
    Ok(Some(manifest))
}

pub(in crate::analysis::storage) mod reclamation;
