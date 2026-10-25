//! Reachability follows original record inventories, never filenames or field projections.
use super::*;

pub(super) fn record_objects(
    record: &PersistedAnalysisArtifact,
    reachable: &mut HashSet<Digest>,
) -> Result<(), String> {
    visit_record_objects(record, &mut |digest| {
        reachable.insert(digest);
        Ok(())
    })
}

/// The shared original-inventory traversal also admits fallible retained consumers.
pub(super) fn visit_record_objects<E: From<String>>(
    record: &PersistedAnalysisArtifact,
    visit: &mut impl FnMut(Digest) -> Result<(), E>,
) -> Result<(), E> {
    match record {
        PersistedAnalysisArtifact::Standard(run) => {
            validate_persisted_run(run, &run.metadata.run_id)?;
            for manifest in run
                .fields
                .iter()
                .map(|field| &field.manifest)
                .chain(
                    run.observation_histories
                        .iter()
                        .map(|history| &history.manifest),
                )
                .chain(run.dynamic_balance.iter().map(|balance| &balance.manifest))
                .chain(run.observation_payload.iter())
            {
                for chunk in &manifest.chunks {
                    visit(chunk.digest)?;
                }
            }
            if let Some(index) = &run.render_tiles {
                for mesh in &index.meshes {
                    for tile in &mesh.tiles {
                        visit(tile.digest)?;
                    }
                    for node in &mesh.lod_nodes {
                        visit(node.tile.digest)?;
                    }
                }
            }
        }
        PersistedAnalysisArtifact::StructuralFrequency(run) => {
            frequency::validate_artifact(run, &run.step.run_id)?;
            for object in &run.object_catalog.objects {
                visit(object.logical_digest)?;
            }
        }
        PersistedAnalysisArtifact::AcousticFrequency(run) => {
            frequency_sample::validate_artifact(run, &run.step.run_id)?;
            for object in &run.object_catalog.objects {
                visit(object.logical_digest)?;
            }
        }
    }
    Ok(())
}

pub(super) fn attachment_objects(
    attachment: &observations::ObservationAttachment,
    reachable: &mut HashSet<Digest>,
) {
    reachable.extend(attachment_digests(attachment));
}

/// One original attachment inventory, reused by admitted and in-memory consumers.
pub(super) fn attachment_digests(
    attachment: &observations::ObservationAttachment,
) -> impl Iterator<Item = Digest> + '_ {
    attachment
        .objects
        .objects
        .iter()
        .map(|object| object.logical_digest)
}

/// Plan under all table locks; validation must finish before the first mutation.
pub(in crate::analysis::storage) fn memory_reachable(
    runs: &HashMap<String, PersistedAnalysisArtifact>,
    observations: &HashMap<String, observations::ObservationAttachment>,
    evictions: &[retention::Eviction],
) -> Result<HashSet<Digest>, String> {
    let evicted = evictions
        .iter()
        .flat_map(|group| group.runs.iter())
        .collect::<HashSet<_>>();
    let mut reachable = HashSet::new();
    for (id, run) in runs {
        if evicted.contains(id) {
            continue;
        }
        record_objects(run, &mut reachable)?;
        if let Some(attachment) = observations.get(id) {
            let PersistedAnalysisArtifact::Standard(run) = run else {
                return Err("observation attachment has no standard run".into());
            };
            attachment.validate(&run.metadata, run.run_digest)?;
            attachment_objects(attachment, &mut reachable);
        }
    }
    if observations.keys().any(|id| !runs.contains_key(id)) {
        return Err("observation attachment owns an absent run".into());
    }
    Ok(reachable)
}
