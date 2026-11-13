//! Deferred object sweep under shared recovery/write fencing, before incoming visibility.
use super::*;

/// Unknown legacy records prevent a proven reachability set, but cannot poison
/// unrelated supported publication. No deletion occurs for this disposition.
#[derive(Debug, PartialEq, Eq)]
pub(in crate::analysis::storage) enum SweepOutcome {
    Reclaimed,
    DeferredLegacyInventory,
}

pub(in crate::analysis::storage) fn sweep(
    store: &access::FilesystemAccess<'_>,
    protected: impl IntoIterator<Item = Digest>,
) -> Result<SweepOutcome, AnalysisPublicationError> {
    let mut protected_set = inventory::Digests::new(store)?;
    for digest in protected {
        protected_set.insert(digest)?;
    }
    let protected = protected_set;
    let mut reachable = inventory::Digests::new(store)?;
    // Ordinary discovery permits retired or unsupported legacy records, and a supported run may
    // carry a legacy observation attachment. Their unknown closure defers all deletion; never
    // infer that their objects are orphans.
    let legacy_attachment = std::cell::Cell::new(false);
    let mut visit = |record: PersistedAnalysisArtifact| {
        super::super::reclamation::visit_record_objects(&record, &mut |digest| {
            reachable.insert(digest)
        })?;
        if let PersistedAnalysisArtifact::Standard(run) = &record {
            let outcome =
                store.visit_stored_observation_attachment(&run.metadata, |attachment| {
                    let _attachment_reservation =
                        store.reserve_resident(store.retained_estimate(&attachment)?)?;
                    for digest in super::super::reclamation::attachment_digests(&attachment) {
                        reachable.insert(digest)?;
                    }
                    Ok(())
                })?;
            if matches!(
                outcome,
                observations::ObservationAttachmentReadOutcome::Legacy
            ) {
                legacy_attachment.set(true);
            }
        }
        Ok(())
    };
    let mut unresolved_legacy = false;
    discovery::visit_supported_artifacts_admitted(
        store,
        |bytes| store.reserve_resident(store.admit_json_metadata(bytes)?),
        |bytes, id| Ok(decode_persisted_artifact(bytes, id)),
        |record| match record {
            Ok(record) => visit(record),
            Err(_) => {
                unresolved_legacy = true;
                Ok(())
            }
        },
    )?;
    // Committed descriptors/reservations/digests remain strict even when legacy
    // collection is deferred. They define supported authority, not optional discovery.
    store.visit_publication_records(visit)?;
    if unresolved_legacy || legacy_attachment.get() {
        return Ok(SweepOutcome::DeferredLegacyInventory);
    }
    let directory = store.objects_directory();
    if !store.owned_directory_exists(&directory, "analysis objects")? {
        return if reachable.is_empty() {
            Ok(SweepOutcome::Reclaimed)
        } else {
            Err("retained publication references absent objects".into())
        };
    }
    let mut available = inventory::Digests::new(store)?;
    let mut obsolete = inventory::Paths::new(store)?;
    let entries = store
        .fs_read_dir(&directory)
        .map_err(AnalysisPublicationError::from_io)?;
    let _entries_reservation = store.retain_directory(&entries)?;
    for entry in entries {
        if entry.file_type() != FsFileType::File {
            return Err("analysis object must be a regular content-addressed file".into());
        }
        let name = entry
            .file_name()
            .to_str()
            .ok_or("analysis object filename invalid")?;
        let digest = format!("sha256:{name}")
            .parse::<Digest>()
            .map_err(|e| e.to_string())?;
        if digest_hex(digest) != name {
            return Err("analysis object filename is not canonical".into());
        }
        available.insert(digest)?;
        if !reachable.contains(&digest) && !protected.contains(&digest) {
            obsolete.push(entry.path())?;
        }
    }
    if reachable.iter().any(|digest| !available.contains(digest)) {
        return Err("retained publication references absent objects".into());
    }
    // Complete reference and ownership validation precedes deletion. Partial sweep
    // is retryable; retained and protected originals are never in the deletion set.
    for path in obsolete.iter() {
        store
            .fs_remove_file(path)
            .map_err(AnalysisPublicationError::from_io)?;
    }
    Ok(SweepOutcome::Reclaimed)
}

/// Source authentication is already complete. Caller owns the write/recovery scope.
/// This maintenance cannot convert an installed result into a reported rejection.
pub(in crate::analysis::storage) fn before_standalone(
    store: &access::FilesystemAccess<'_>,
    run_id: &str,
    identity: &AnalysisRunIdentity,
    created_at: &str,
    protected: impl IntoIterator<Item = Digest>,
) -> Result<(), AnalysisPublicationError> {
    let incoming = HashMap::from([(
        run_id.to_owned(),
        retention::RunFacts::from_metadata(identity, created_at),
    )]);
    // Validate committed closure before collecting anything; retained visibility
    // changes are applied only after successful original record installation.
    transaction::plan_facts(store, &incoming, None)?;
    cleanup::uncommitted_roots(store, None)?;
    if sweep(store, protected)? == SweepOutcome::DeferredLegacyInventory {
        log::debug!("analysis object reclamation deferred: unsupported legacy inventory");
    }
    Ok(())
}
