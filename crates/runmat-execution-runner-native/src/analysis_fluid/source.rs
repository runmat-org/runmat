//! Authenticated bridge-to-runtime boundary, without a runtime artifact dependency.
use std::collections::BTreeMap;
use std::sync::OnceLock;

use runmat_analysis_execution::{
    import_fluid_study_completion, prepare_fluid_study_root_for, CompletedFluidStudy,
    FluidStudyPreparation, VerifiedFluidStudy,
};
use runmat_execution::{
    value::{PublishedObjectSource, ValueRef},
    Digest,
};
use runmat_execution_artifact::{cache::CacheImport, ArtifactResult};
use runmat_runtime::analysis::{FluidStudyCompletion, FluidStudyObjectSource, FluidStudySource};

struct RetainedFluidSource<S> {
    artifact_access: runmat_execution_artifact::ArtifactAccess,
    completed: CompletedFluidStudy,
    /// `completed.preparation()`, validated and re-planned once: the retained study is
    /// immutable, so every replay, field materialization and DTO build shares it.
    preparation: FluidStudyPreparation,
    store: S,
    objects: BTreeMap<Digest, Vec<u8>>,
    /// The authenticated completion every candidate is compared against. A study this
    /// process's coordinator verified ([`VerifiedFluidStudy`]) sets it at conversion; a
    /// decoded study sets it from one cold replay of `completed` over this source's
    /// immutable store, after every catalog object was read back verified.
    authenticated: OnceLock<FluidStudyCompletion>,
}

impl<S: CacheImport> CacheImport for RetainedFluidSource<S> {
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

impl<S: CacheImport + Send + Sync> PublishedObjectSource for RetainedFluidSource<S> {
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
            .ok_or("fluid source omitted a catalog object")?;
        if Digest::sha256(&bytes) != reference.logical_digest
            || bytes.len() as u64 != reference.encoded_length
        {
            return Err("fluid source object digest or length differs".into());
        }
        Ok(bytes)
    }
}

impl<S: CacheImport + Send + Sync> FluidStudySource for RetainedFluidSource<S> {
    fn materialize_fields(
        &self,
        candidate: &FluidStudyCompletion,
    ) -> Result<runmat_analysis_fea::results::field::RecoveredFieldArtifact, String> {
        // Authenticate the retained artifact rather than generating fresh numeric copies
        // from U/p/q. Original phase admission is rederived from the saved immutable chain.
        self.authenticate(candidate)?;
        let plan = self.completed.field_recovery_resource_plan();
        let runmat_analysis_fea::planning::resource::FeaResourceDemand::FluidFieldRecovery(demand) =
            &plan.input.demand
        else {
            return Err(
                "fluid native field materialization lacks original recovery authority".into(),
            );
        };
        let preparation = &self.preparation;
        let mut limits = preparation.output_policy.inventory_limits();
        limits.max_total_bytes = limits
            .max_total_bytes
            .min(demand.output.transport_closure_bytes_upper_bound.0);
        let output = &self.completed.fields.output;
        let objects = runmat_analysis_execution::import_analysis_objects(
            self,
            &output.root,
            runmat_analysis_fea::execution::AnalysisNumericalObjectKind::Field,
            output.logical_identity,
            output.payload_digest,
            &self.artifact_access,
            limits,
        )
        .map_err(|e| e.to_string())?;
        if objects.manifest.encoded_length > demand.output.encoded_payload_bytes_upper_bound.0 {
            return Err(
                "fluid retained field bytes exceed original recovery output admission".into(),
            );
        }
        // Encoded/transport bounds precede the shared profiler's bounded partition joins.
        // The exact actual codec envelope reserves typed partitions/reconstruction, while
        // the separately retained transport remains live. These are requested heap bounds.
        let facts = runmat_analysis_execution::prepared_input_resource_facts(std::slice::from_ref(
            &objects,
        ))
        .map_err(|e| e.to_string())?;
        let required = facts
            .decoded_resident_bytes_admission_budget
            .0
            .checked_add(facts.transport_closure_bytes.0)
            .ok_or("fluid native field reconstruction requested heap overflow")?;
        // Recovery reserves exact catalog/identity/partition canonical encoding and
        // materialization separately from the transport-object builder. Reuse those
        // contemporaneous construction reservations, not unrelated validation headroom.
        let construction = demand
            .construction_workspace_bytes
            .0
            .checked_add(demand.output.construction_resident_bytes_admission_budget.0)
            .ok_or("fluid original field construction envelope overflow")?;
        if required > construction {
            return Err(format!(
                "fluid original field codec/reconstruction exceeds admitted construction heap: {required} > {construction}; original partition/materialization workspace={}, transport builder={}, phase={}",
                demand.construction_workspace_bytes.0,
                demand.output.construction_resident_bytes_admission_budget.0,
                plan.request.memory_bytes,
            ));
        }
        let coexisting = demand
            .source_validation_workspace_bytes
            .0
            .checked_add(demand.inputs.decoded_resident_bytes_admission_budget.0)
            .and_then(|bytes| bytes.checked_add(demand.inputs.transport_closure_bytes.0))
            .and_then(|bytes| bytes.checked_add(required))
            .ok_or("fluid original field coexisting reconstruction heap overflow")?;
        if coexisting > plan.request.memory_bytes {
            return Err("fluid original field source validation and codec reconstruction exceed phase admission".into());
        }
        let artifact = runmat_analysis_execution::decode_recovered_field_artifact(&objects)
            .map_err(|e| e.to_string())?;
        if artifact.manifest().map_err(|e| e.to_string())? != candidate.fields {
            return Err(
                "fluid original recovered-field artifact differs from authenticated DTO".into(),
            );
        }
        Ok(artifact)
    }

    fn authenticate(&self, candidate: &FluidStudyCompletion) -> Result<(), String> {
        let expected = self.authenticated()?;
        if expected.root.logical_digest != candidate.root.logical_digest || expected != candidate {
            return Err("runtime fluid completion differs from authenticated replay".into());
        }
        Ok(())
    }
}

impl<S: CacheImport + Send + Sync> RetainedFluidSource<S> {
    /// The authenticated completion. For a decoded study, replay the saved study once:
    /// import every phase from authored inputs, rebuild the completion DTO and read back
    /// every catalog object verified. The source is immutable, so a later authentication
    /// reuses this verified replay instead of importing again.
    fn authenticated(&self) -> Result<&FluidStudyCompletion, String> {
        if let Some(authenticated) = self.authenticated.get() {
            return Ok(authenticated);
        }
        let replay = import_fluid_study_completion(
            self,
            self.preparation.clone(),
            self.completed.result_closures().to_vec(),
        )
        .map_err(|error| error.to_string())?;
        // The replay was imported from exactly this preparation.
        let root = prepare_fluid_study_root_for(&replay, self.preparation.clone())
            .map_err(|error| error.to_string())?;
        let expected = completion(&replay, &self.preparation, self, root.root, root.catalog)?;
        for reference in &expected.catalog.objects {
            PublishedObjectSource::read_verified(self, reference)?;
        }
        Ok(self.authenticated.get_or_init(|| expected))
    }
}

fn completion(
    completed: &CompletedFluidStudy,
    preparation: &FluidStudyPreparation,
    source: &impl CacheImport,
    root: ValueRef,
    catalog: runmat_execution_artifact::PublishedObjectCatalog,
) -> Result<FluidStudyCompletion, String> {
    Ok(FluidStudyCompletion {
        host_workload_digest: completed
            .result_closures()
            .first()
            .ok_or("fluid completion omits solve")?
            .workload
            .identity
            .digest(),
        product: completed.product.clone(),
        result: completed.result.result.clone(),
        numerical_receipt: completed.result.numerical_receipt.clone(),
        fields: completed.fields.manifest.clone(),
        observation_catalog: completed.observations.catalog.clone(),
        observation_results: completed
            .observations
            .samples
            .iter()
            .flat_map(|sample| {
                sample.manifest.results.iter().map(move |entry| {
                    runmat_analysis_execution::import_observation_result(
                        source,
                        sample,
                        &entry.observation_id,
                        preparation.output_policy.inventory_limits(),
                    )
                    .map_err(|error| error.to_string())
                })
            })
            .collect::<Result<Vec<_>, _>>()?,
        observation_histories: completed
            .observations
            .histories
            .iter()
            .map(|finalized| {
                history(
                    source,
                    &finalized.history,
                    preparation.output_policy.inventory_limits(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?,
        series_reductions: completed
            .observations
            .histories
            .iter()
            .filter_map(|finalized| finalized.series_reduction.clone())
            .collect(),
        accuracy: completed.accuracy.artifact.clone(),
        states: completed
            .states
            .iter()
            .map(|state| state.state.clone())
            .collect(),
        baseline_usage: completed.baseline_usage.evidence.clone(),
        terminal_usage: completed.terminal_usage.evidence.clone(),
        publication: completed.publication.result.clone(),
        frozen_project_digest: preparation.frozen_project_digest,
        root,
        catalog,
    })
}

/// Convert the study this process's coordinator just verified. It is the same process's
/// own typed, already-validated result, so its DTO is the authenticated completion: no
/// replay of the phase import against it.
pub(crate) fn convert(
    verified: VerifiedFluidStudy,
    store: impl CacheImport + Send + Sync + 'static,
) -> Result<(FluidStudyCompletion, FluidStudyObjectSource), String> {
    let preparation = verified.preparation();
    let completed = verified.into_completed();
    let root = prepare_fluid_study_root_for(&completed, preparation.clone())
        .map_err(|error| error.to_string())?;
    let mut source = RetainedFluidSource {
        artifact_access: preparation.artifact_access.clone(),
        preparation,
        objects: root
            .objects
            .into_iter()
            .map(|object| (object.descriptor.digest, object.bytes))
            .collect(),
        completed,
        store,
        authenticated: OnceLock::new(),
    };
    let dto = completion(
        &source.completed,
        &source.preparation,
        &source,
        root.root,
        root.catalog,
    )?;
    source.authenticated = OnceLock::from(dto.clone());
    Ok((dto, FluidStudyObjectSource::new(source)))
}

/// Convert an imported completion using only its owned, complete published object closure.
/// Unlike local placement, no root or evidence bytes are synthesized to fill missing objects.
pub fn fluid_study_completion_from_source(
    completed: CompletedFluidStudy,
    source: impl CacheImport + Send + Sync + 'static,
) -> Result<(FluidStudyCompletion, FluidStudyObjectSource), String> {
    let preparation = completed.preparation().map_err(|e| e.to_string())?;
    let source = RetainedFluidSource {
        artifact_access: preparation.artifact_access.clone(),
        preparation,
        completed,
        store: source,
        objects: BTreeMap::new(),
        authenticated: OnceLock::new(),
    };
    let root = prepare_fluid_study_root_for(&source.completed, source.preparation.clone())
        .map_err(|error| error.to_string())?;
    let dto = completion(
        &source.completed,
        &source.preparation,
        &source,
        root.root,
        root.catalog,
    )?;
    source.authenticate(&dto)?;
    Ok((dto, FluidStudyObjectSource::new(source)))
}

/// The complete published history: whole histories as imported, partitioned ones sample by
/// sample from their retained chunks.
fn history(
    source: &impl CacheImport,
    imported: &runmat_analysis_execution::ImportedObservationHistory,
    limits: runmat_execution_artifact::object::ObjectInventoryLimits,
) -> Result<runmat_analysis_fea::results::observation::ObservationHistory, String> {
    if let runmat_analysis_execution::ImportedObservationHistoryStorage::Whole { history, .. } =
        &imported.storage
    {
        return Ok((**history).clone());
    }
    let samples = imported
        .manifest
        .samples
        .iter()
        .map(|entry| {
            runmat_analysis_execution::import_observation_history_sample(
                source,
                imported,
                entry.coordinate,
                limits,
            )
            .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    runmat_analysis_fea::results::observation::ObservationHistory::from_manifest_and_samples(
        imported.manifest.clone(),
        samples,
    )
    .map_err(|error| error.to_string())
}
