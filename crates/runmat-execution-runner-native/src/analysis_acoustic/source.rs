//! Authenticated bridge-to-runtime boundary, without a runtime artifact dependency.
use std::collections::BTreeMap;

use runmat_analysis_execution::{
    import_acoustic_study_completion, prepare_acoustic_study_root, CompletedAcousticStudy,
};
use runmat_execution::{
    value::{PublishedObjectSource, ValueRef},
    Digest,
};
use runmat_execution_artifact::{cache::CacheImport, ArtifactResult};
use runmat_runtime::analysis::{
    AcousticStudyCompletion, AcousticStudyEntryCompletion, AcousticStudyObjectSource,
    AcousticStudySource, AnalysisArtifactRoot, FrequencySampleArtifactRootInventory,
    FrequencySampleArtifactRoots, FrequencySamplePublicationAccess,
};

struct RetainedAcousticSource<S> {
    artifact_access: runmat_execution_artifact::ArtifactAccess,
    completed: CompletedAcousticStudy,
    store: S,
    objects: BTreeMap<Digest, Vec<u8>>,
}

impl<S: CacheImport> CacheImport for RetainedAcousticSource<S> {
    fn read_verified_bounded_with_admission(
        &self,
        digest: Digest,
        maximum_bytes: u64,
        admit: &mut dyn FnMut(u64) -> ArtifactResult<()>,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        crate::bounded_object_read::read_overlay_admitted(
            &self.store,
            digest,
            self.objects.get(&digest).map(Vec::as_slice),
            maximum_bytes,
            admit,
        )
    }
    fn read_verified_bounded(
        &self,
        digest: Digest,
        maximum_bytes: u64,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        crate::bounded_object_read::read_overlay(
            &self.store,
            digest,
            self.objects.get(&digest).map(Vec::as_slice),
            maximum_bytes,
        )
    }
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        if let Some(bytes) = self.objects.get(&digest) {
            return Ok(Some(bytes.clone()));
        }
        self.store.read_verified(digest)
    }
}

impl<S: CacheImport + Send + Sync> PublishedObjectSource for RetainedAcousticSource<S> {
    fn read_verified(&self, reference: &ValueRef) -> Result<Vec<u8>, String> {
        self.artifact_access
            .validate_reference(
                reference,
                runmat_execution::value::ValueRefKind::ResultObject,
                &reference.media_type,
                &reference.value_schema,
            )
            .map_err(|error| error.to_string())?;
        let bytes = CacheImport::read_verified(self, reference.logical_digest)
            .map_err(|error| error.to_string())?
            .ok_or("acoustic source omitted a catalog object")?;
        if Digest::sha256(&bytes) != reference.logical_digest
            || bytes.len() as u64 != reference.encoded_length
        {
            return Err("acoustic source object digest or length differs".into());
        }
        Ok(bytes)
    }
}

impl<S: CacheImport + Send + Sync> AcousticStudySource for RetainedAcousticSource<S> {
    fn authenticate(&self, candidate: &AcousticStudyCompletion) -> Result<(), String> {
        let replay = import_acoustic_study_completion(
            self,
            self.completed.preparation().map_err(|e| e.to_string())?,
            self.completed.entry_closures(),
            self.completed.history_closures.clone(),
            self.completed.assembly_closure.clone(),
            self.completed.wave_wall_millis,
        )
        .map_err(|error| error.to_string())?;
        let expected = completion(&replay, self)?;
        if &expected != candidate {
            return Err("runtime acoustic completion differs from authenticated replay".into());
        }
        for reference in &expected.catalog.objects {
            PublishedObjectSource::read_verified(self, reference)?;
        }
        Ok(())
    }
}

fn completion(
    completed: &CompletedAcousticStudy,
    source: &impl CacheImport,
) -> Result<AcousticStudyCompletion, String> {
    let root = prepare_acoustic_study_root(completed).map_err(|error| error.to_string())?;
    let preparation = completed.preparation().map_err(|e| e.to_string())?;
    let access = &preparation.artifact_access;
    let artifact_root = |objects: &runmat_analysis_execution::PreparedAnalysisObjects| {
        AnalysisArtifactRoot::new(
            objects
                .root_reference(access, runmat_execution::value::ValueRefKind::ResultObject)
                .map_err(|e| e.to_string())?,
            objects.manifest.kind,
            objects.manifest.logical_identity,
            objects.manifest.payload_digest,
        )
    };
    let publication_root = artifact_root(&completed.assembly.publication_objects)?;
    let mut references = completed
        .assembly
        .publication_objects
        .unique_references(access, runmat_execution::value::ValueRefKind::ResultObject)
        .map_err(|e| e.to_string())?;
    let mut roots = vec![publication_root.reference().clone()];
    let mut samples = Vec::new();
    // Each entry's own recovery and observation tasks published its fields and outputs.
    for entry in &completed.entries {
        let fields_root = artifact_root(&entry.field_objects)?;
        roots.push(fields_root.reference().clone());
        references.extend(
            entry
                .field_objects
                .unique_references(access, runmat_execution::value::ValueRefKind::ResultObject)
                .map_err(|e| e.to_string())?,
        );
        let observations_root = entry
            .observation_objects
            .as_ref()
            .map(|objects| -> Result<_, String> {
                let root = artifact_root(objects)?;
                roots.push(root.reference().clone());
                references.extend(
                    objects
                        .unique_references(
                            access,
                            runmat_execution::value::ValueRefKind::ResultObject,
                        )
                        .map_err(|e| e.to_string())?,
                );
                Ok(root)
            })
            .transpose()?;
        samples.push(FrequencySampleArtifactRoots::new(
            fields_root,
            observations_root,
        ));
    }
    let mut unique = BTreeMap::new();
    for reference in references {
        if let Some(previous) = unique.insert(reference.id, reference.clone()) {
            if previous != reference {
                return Err(
                    "acoustic publication contains conflicting authority references".into(),
                );
            }
        }
    }
    let publication_catalog =
        runmat_execution::value::PublishedObjectCatalog::new(roots, unique.into_values().collect())
            .map_err(|e| e.to_string())?;
    let authority = completed
        .assembly
        .result
        .publication_authority()
        .map_err(|e| e.to_string())?;
    let publication_roots = FrequencySampleArtifactRootInventory::new(
        publication_root,
        samples,
        &FrequencySamplePublicationAccess {
            publication: &completed.assembly.publication,
            authority: &authority,
            object_catalog: &publication_catalog,
        },
    )?;
    Ok(AcousticStudyCompletion {
        compiled: completed.compiled.clone(),
        workload: completed.workload.clone(),
        result: completed.assembly.result.clone(),
        entries: completed
            .entries
            .iter()
            .map(|entry| {
                let solve = entry
                    .result_closures()
                    .iter()
                    .find(|closure| {
                        closure.phase == runmat_analysis_execution::AcousticStudyResultPhase::Solve
                    })
                    .ok_or("acoustic entry omits its exact solve phase")?;
                Ok(AcousticStudyEntryCompletion {
                    entry: entry.entry,
                    workload: entry.workload.clone(),
                    host_workload_digest: solve.workload.identity.digest(),
                    result: entry.result.result.clone(),
                    numerical_receipt: entry.result.numerical_receipt.clone(),
                    observation_catalog: entry.observation_catalog.clone(),
                    // The outputs this frequency schedules, in authored order.
                    observation_results: match &entry.observations {
                        Some(observations) => observations
                            .manifest
                            .results
                            .iter()
                            .map(|result| {
                                runmat_analysis_execution::import_observation_result(
                                    source,
                                    observations,
                                    &result.observation_id,
                                    preparation.output_policy.inventory_limits(),
                                )
                                .map_err(|error| error.to_string())
                            })
                            .collect::<Result<Vec<_>, String>>()?,
                        None => Vec::new(),
                    },
                    accuracy: entry.accuracy.artifact.clone(),
                    states: entry
                        .states
                        .iter()
                        .map(|state| state.state.clone())
                        .collect(),
                    baseline_usage: entry.baseline_usage.evidence.clone(),
                    terminal_usage: entry.terminal_usage.evidence.clone(),
                    publication: entry.publication.result.clone(),
                })
            })
            .collect::<Result<Vec<_>, String>>()?,
        history_catalog: completed
            .histories
            .as_ref()
            .map(|histories| histories.catalog.clone()),
        series_reductions: completed
            .histories
            .iter()
            .flat_map(|histories| &histories.histories)
            .filter_map(|history| history.series_reduction.clone())
            .collect(),
        resource_usage: completed.resource_usage.evidence.clone(),
        publication: completed.assembly.publication.clone(),
        publication_catalog,
        publication_roots,
        frozen_project_digest: preparation.frozen_project_digest,
        root: root.root,
        catalog: root.catalog,
    })
}

pub(crate) fn convert(
    completed: CompletedAcousticStudy,
    store: impl CacheImport + Send + Sync + 'static,
) -> Result<(AcousticStudyCompletion, AcousticStudyObjectSource), String> {
    let root = prepare_acoustic_study_root(&completed).map_err(|error| error.to_string())?;
    let source = RetainedAcousticSource {
        artifact_access: completed
            .preparation()
            .map_err(|e| e.to_string())?
            .artifact_access,
        objects: root
            .objects
            .into_iter()
            .map(|object| (object.descriptor.digest, object.bytes))
            .collect(),
        completed,
        store,
    };
    let dto = completion(&source.completed, &source)?;
    source.authenticate(&dto)?;
    Ok((dto, AcousticStudyObjectSource::new(source)))
}

/// Convert an imported completion using only its owned, complete published object closure.
/// Unlike local placement, no root or evidence bytes are synthesized to fill missing objects.
pub fn acoustic_study_completion_from_source(
    completed: CompletedAcousticStudy,
    source: impl CacheImport + Send + Sync + 'static,
) -> Result<(AcousticStudyCompletion, AcousticStudyObjectSource), String> {
    let source = RetainedAcousticSource {
        artifact_access: completed
            .preparation()
            .map_err(|e| e.to_string())?
            .artifact_access,
        completed,
        store: source,
        objects: BTreeMap::new(),
    };
    let dto = completion(&source.completed, &source)?;
    source.authenticate(&dto)?;
    Ok((dto, AcousticStudyObjectSource::new(source)))
}
