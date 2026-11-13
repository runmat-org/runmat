//! Install a closed inventory while all reader-visible tables remain write-locked.
use super::*;

pub(in crate::analysis::storage) fn commit(
    store: &InMemoryAnalysisArtifactStore,
    mut prepared: PreparedAnalysisPublication,
) -> Result<AnalysisPublicationReceipt, AnalysisPublicationError> {
    prepared.validate()?;
    // The memory backend retains every chunk in memory; streamed chunks are read back after
    // the staging admission has been validated.
    let spilled = std::mem::take(&mut prepared.spilled);
    for digest in spilled.digests() {
        let bytes = spilled
            .read(&digest)?
            .ok_or("publication spilled chunk is absent")?;
        prepared.chunks.insert(digest, Arc::from(bytes));
    }
    let receipt = prepared.descriptor.receipt(None)?;
    let bytes = resources::json(
        &(&prepared.descriptor, &prepared.manifest),
        prepared.maximum_bytes,
    )?;
    // Observation readers acquire attachment locks before a second run validation.
    let mut observations = store
        .observations
        .write()
        .map_err(|_| "observation lock poisoned")?;
    let mut runs = store.runs.write().map_err(|_| "run lock poisoned")?;
    let mut chunks = store.chunks.write().map_err(|_| "chunk lock poisoned")?;
    let mut publications = store
        .publications
        .write()
        .map_err(|_| "publication lock poisoned")?;
    let mut owners = store
        .publication_runs
        .write()
        .map_err(|_| "publication ownership lock poisoned")?;
    let retry = publications.get(&receipt.identity);
    if retry.is_some_and(|existing| existing != &bytes) {
        return Err("publication identity contains different content".into());
    }
    for (id, record) in &prepared.runs {
        if let Some(existing) = runs.get(id) {
            if retry.is_none() || record_digest(existing)? != record_digest(record)? {
                return Err("publication run identity already belongs to different content".into());
            }
        }
    }
    for (digest, bytes) in &prepared.chunks {
        if Digest::sha256(bytes.as_ref()) != *digest
            || chunks
                .get(digest)
                .is_some_and(|existing| existing.as_ref() != bytes.as_ref())
        {
            return Err("publication chunk digest collision".into());
        }
    }
    for (id, attachment) in &prepared.observations {
        if observations
            .get(id)
            .is_some_and(|existing| existing != attachment)
        {
            return Err("publication observation identity collision".into());
        }
    }
    let evictions =
        retention::plan_memory(&prepared, receipt.identity, &runs, &owners, &publications)?;
    let mut reachable = reclamation::memory_reachable(&runs, &observations, &evictions)?;
    if reachable.iter().any(|digest| !chunks.contains_key(digest)) {
        return Err("retained publication references absent objects".into());
    }
    reachable.extend(prepared.chunks.keys().copied());
    crate::analysis::fea_run_admission::FeaRunScope::check_publication_checked()
        .map_err(|denied| AnalysisPublicationError::from(denied.to_string()))?;
    if crate::interrupt::is_cancelled() {
        return Err("study publication was cancelled".into());
    }
    // All fallible validation and cancellation precede the first table mutation.
    for eviction in evictions {
        for id in eviction.runs {
            runs.remove(&id);
            owners.remove(&id);
            observations.remove(&id);
        }
        if let Some(root) = eviction.publication {
            publications.remove(&root);
        }
    }
    chunks.retain(|digest, _| reachable.contains(digest));
    // Retained and incoming originals share content-addressed chunks.
    owners.extend(
        prepared
            .runs
            .keys()
            .map(ToOwned::to_owned)
            .map(|id| (id, receipt.identity)),
    );
    runs.extend(prepared.runs);
    chunks.extend(prepared.chunks);
    observations.extend(prepared.observations);
    publications.insert(receipt.identity, bytes);
    Ok(receipt)
}

pub(in crate::analysis::storage) fn load(
    store: &InMemoryAnalysisArtifactStore,
    identity: Digest,
    maximum_bytes: u64,
) -> Result<Option<serde_json::Value>, AnalysisPublicationError> {
    let publications = store
        .publications
        .read()
        .map_err(|_| "publication lock poisoned")?;
    let Some(bytes) = publications.get(&identity) else {
        return Ok(None);
    };
    if bytes.len() as u64 > maximum_bytes {
        return Err(AnalysisPublicationError::ResourceLimit {
            message: "publication exceeds its artifact byte budget".into(),
        });
    }
    let (descriptor, manifest): (PublicationDescriptor, serde_json::Value) =
        serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    validate_manifest(identity, &descriptor, &manifest)?;
    Ok(Some(manifest))
}
