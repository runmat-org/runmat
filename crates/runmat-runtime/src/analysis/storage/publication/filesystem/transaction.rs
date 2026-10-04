//! Whole inventory selection and visibility changes share the provider journal fence.
use super::*;
use runmat_filesystem::workspace::{
    WorkspaceCaptureRequest, WorkspaceEdit, WorkspacePublicationId, WorkspacePublicationOutcome,
    WorkspacePublicationRequest, MAX_WORKSPACE_PUBLICATION_BYTES,
};

mod edits;

#[cfg(test)]
mod tests;

pub(super) struct EvictionPlan<'a> {
    entries: Vec<retention::Eviction>,
    _reservation: access::Reservation<'a>,
}

pub(super) fn plan<'a>(
    store: &'a access::FilesystemAccess<'_>,
    incoming: &HashMap<String, PersistedAnalysisArtifact>,
    identity: Option<Digest>,
) -> Result<EvictionPlan<'a>, AnalysisPublicationError> {
    plan_facts(store, &retention::inventory_facts(incoming), identity)
}

pub(super) fn plan_facts<'a>(
    store: &'a access::FilesystemAccess<'_>,
    incoming: &HashMap<String, retention::RunFacts>,
    identity: Option<Digest>,
) -> Result<EvictionPlan<'a>, AnalysisPublicationError> {
    let mut inventory = store.reserve_resident(0)?;
    let mut runs = HashMap::new();
    let mut owners = HashMap::new();
    store.visit_persisted_runs(|record| {
        let id = match &record {
            PersistedAnalysisArtifact::Standard(r) => &r.metadata.run_id,
            PersistedAnalysisArtifact::StructuralFrequency(r) => &r.step.run_id,
            PersistedAnalysisArtifact::AcousticFrequency(r) => &r.step.run_id,
        }
        .clone();
        let owner = store.publication_owner(&id)?;
        let facts = retention::facts(&record);
        // Charge borrowed entry shapes before growing retained maps. Each entry
        // uses the shared conservative materialization profile, including overhead.
        inventory.grow(store.retained_estimate(&(&id, &facts, &owner))?)?;
        if let Some(owner) = owner {
            owners.insert(id.clone(), owner);
        }
        if runs.insert(id, facts).is_some() {
            return Err("retention inventory repeats a run identity".into());
        }
        Ok(())
    })?;
    let mut descriptors = HashMap::new();
    let roots = roots(store)?;
    for root in roots.iter() {
        let name = root
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or("publication directory name invalid")?;
        let digest = format!("sha256:{name}")
            .parse::<Digest>()
            .map_err(|e| e.to_string())?;
        if let Some(descriptor) = descriptor(store, digest, MAX_WORKSPACE_PUBLICATION_BYTES as u64)?
        {
            inventory.grow(store.retained_estimate(&(digest, &descriptor))?)?;
            descriptors.insert(digest, descriptor);
        }
    }
    // Selector working copies coexist with ranking maps and the admitted original.
    let _selector_reservation = store.reserve_resident(store.retained_estimate(&(
        incoming,
        &runs,
        &owners,
        &descriptors,
    ))?)?;
    let entries = retention::plan_inventory(incoming, identity, &runs, &owners, &descriptors)?;
    let reservation = store.reserve_resident(if entries.is_empty() {
        0
    } else {
        store.retained_estimate(&entries)?
    })?;
    Ok(EvictionPlan {
        entries,
        _reservation: reservation,
    })
}

pub(super) fn roots<'a, 'store>(
    store: &'a access::FilesystemAccess<'store>,
) -> Result<inventory::Paths<'a, 'store>, AnalysisPublicationError> {
    let path = store.root.join("publications");
    let mut roots = inventory::Paths::new(store)?;
    if !store.owned_directory_exists(&path, "publication directory")? {
        return Ok(roots);
    }
    let entries = store
        .fs_read_dir(path)
        .map_err(AnalysisPublicationError::from_io)?;
    let _directory_reservation = store.retain_directory(&entries)?;
    for entry in entries {
        if entry.file_type() != FsFileType::Directory {
            return Err("publication root must be a directory".into());
        }
        roots.push(entry.path())?;
    }
    Ok(roots)
}

fn exists(
    store: &access::FilesystemAccess<'_>,
    path: &Path,
) -> Result<bool, AnalysisPublicationError> {
    match store.fs_symlink_metadata(path) {
        Ok(metadata) if metadata.file_type() == FsFileType::File => Ok(true),
        Ok(_) => Err("retention metadata must be a regular file".into()),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error.to_string().into()),
    }
}

pub(super) fn transition(
    store: &access::FilesystemAccess<'_>,
    marker: Option<(PathBuf, Vec<u8>)>,
    evictions: EvictionPlan<'_>,
) -> Result<(), AnalysisPublicationError> {
    let relative = |path: PathBuf| {
        path.strip_prefix(&store.root)
            .map(Path::to_path_buf)
            .map_err(|_| "publication transaction escapes its original root".to_owned())
    };
    let mut edits = edits::Edits::new(store)?;
    let EvictionPlan {
        entries,
        _reservation,
    } = evictions;
    for eviction in entries {
        if let Some(root) = eviction.publication {
            edits.push(WorkspaceEdit::Remove {
                path: relative(directory(store, root).join("commit.json"))?,
            })?;
        } else {
            for id in &eviction.runs {
                edits.push(WorkspaceEdit::Remove {
                    path: relative(store.run_path(id)?)?,
                })?;
                let observation = store.root.join("observations").join(format!("{id}.json"));
                if exists(store, &observation)? {
                    edits.push(WorkspaceEdit::Remove {
                        path: relative(observation)?,
                    })?;
                }
            }
        }
        for id in eviction.runs {
            let path = store
                .root
                .join("publication-bindings")
                .join(format!("{id}.json"));
            if exists(store, &path)? {
                edits.push(WorkspaceEdit::Remove {
                    path: relative(path)?,
                })?;
            }
        }
    }
    if let Some((path, bytes)) = marker {
        edits.push(WorkspaceEdit::Write {
            path: relative(path)?,
            bytes,
        })?;
    }
    if edits.is_empty() {
        return Ok(());
    }
    // Original numerical objects and private child records are not transaction inputs.
    // Legacy visible records remain included because their removal is part of this batch.
    let mut exclusions = inventory::Paths::new(store)?;
    exclusions.push(Path::new("objects"))?;
    // The admitted artifact root may also host producer workspaces. They are not
    // artifact authority or metadata transaction inputs; never gather their fields.
    let entries = store
        .fs_read_dir(&store.root)
        .map_err(AnalysisPublicationError::from_io)?;
    let directory_reservation = store.retain_directory(&entries)?;
    for entry in entries {
        if !matches!(
            entry.file_name().to_str(),
            Some("runs" | "observations" | "publications" | "publication-bindings")
        ) {
            exclusions.push(&relative(entry.path().to_path_buf())?)?;
        }
    }
    let publication_roots = roots(store)?;
    for root in publication_roots.iter() {
        for child in ["runs", "observations", "manifest.json"] {
            exclusions.push(&relative(root.join(child))?)?;
        }
    }
    drop(directory_reservation);
    drop(publication_roots);
    let (excluded_subtrees, exclusions_reservation) = exclusions.into_parts();
    let snapshot = store.capture_workspace(WorkspaceCaptureRequest {
        root: "/".into(),
        excluded_subtrees,
        max_files: 10_000,
        max_bytes: MAX_WORKSPACE_PUBLICATION_BYTES as u64,
    })?;
    // The provider rederives the fenced inventory; retained field/record buffers
    // are not needed while its transaction validation and journal plan execute.
    drop(snapshot.files);
    drop(exclusions_reservation);
    let _snapshot_reservation =
        store.reserve_resident(store.retained_estimate(&snapshot.excluded_subtrees)?)?;
    if crate::interrupt::is_cancelled() {
        return Err("study publication was cancelled".into());
    }
    let (edits, _edits_reservation) = edits.into_parts();
    match store.publish_workspace(WorkspacePublicationRequest {
        id: WorkspacePublicationId::new(),
        root: "/".into(),
        excluded_subtrees: snapshot.excluded_subtrees,
        expected_revision: snapshot.revision,
        edits,
    })? {
        WorkspacePublicationOutcome::Published { .. }
        | WorkspacePublicationOutcome::CommittedPendingRecovery { .. } => Ok(()),
        WorkspacePublicationOutcome::DurabilityIndeterminate { receipt } => {
            Err(AnalysisPublicationError::DurabilityIndeterminate {
                workspace: receipt,
                publication: None,
            })
        }
        WorkspacePublicationOutcome::Conflict { .. } => {
            Err("publication inventory changed inside its provider fence".into())
        }
        WorkspacePublicationOutcome::IdReused => {
            Err("publication transaction identity was reused".into())
        }
    }
}

pub(in crate::analysis::storage) fn prune_standalone(
    store: &access::FilesystemAccess<'_>,
    run_id: &str,
) -> Result<(), AnalysisPublicationError> {
    let limits = crate::analysis::storage::retention::resolved_limits()?;
    if limits.total == 0 && limits.per_kind == 0 {
        return Ok(());
    }
    let bytes = store
        .fs_read(store.run_path(run_id)?)
        .map_err(AnalysisPublicationError::from_io)?;
    let record = store.decode_record(&bytes, run_id)?;
    let incoming = HashMap::from([(run_id.to_owned(), record)]);
    transition(store, None, plan(store, &incoming, None)?)
}
