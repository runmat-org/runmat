//! Cold-root authentication at the runtime boundary; no retained decoded prefix inventory.
use crate::analysis_placement::source_failure as failure;
use failure::{bridge, fea, limit};
use runmat_analysis_execution::{
    import_nonlinear_solid_study_root, prepare_nonlinear_solid_study_root,
    CompletedNonlinearSolidStudy,
};
use runmat_execution::{
    value::{PublishedObjectSource, ValueRef, ValueRefKind},
    Digest,
};
use runmat_execution_artifact::{
    cache::CacheImport, object::ObjectInventoryLimits, ArtifactAccess, ArtifactResult,
};
use runmat_runtime::analysis::storage::AnalysisPublicationError;
use runmat_runtime::analysis::{
    NonlinearSolidStudyCompletion, NonlinearSolidStudyObjectSource, NonlinearSolidStudySource,
};

struct OriginalSource<S> {
    root: ValueRef,
    catalog: runmat_execution::value::PublishedObjectCatalog,
    access: ArtifactAccess,
    limits: ObjectInventoryLimits,
    maximum_resident_bytes: u64,
    store: S,
}
impl<S: CacheImport> CacheImport for OriginalSource<S> {
    fn read_verified_bounded_with_admission(
        &self,
        digest: Digest,
        maximum_bytes: u64,
        admit: &mut dyn FnMut(u64) -> ArtifactResult<()>,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        let reference = self
            .catalog
            .objects
            .iter()
            .find(|r| r.logical_digest == digest)
            .ok_or_else(|| {
                runmat_execution_artifact::ArtifactError::Identity(
                    "nonlinear read outside original catalog".into(),
                )
            })?;
        if reference.encoded_length > maximum_bytes {
            return Err(runmat_execution_artifact::ArtifactError::Limit(
                "original nonlinear object exceeds caller read allowance".into(),
            ));
        }
        self.store
            .read_verified_bounded_with_admission(digest, reference.encoded_length, admit)
    }
    fn read_verified_bounded(
        &self,
        digest: Digest,
        maximum_bytes: u64,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        let reference = self
            .catalog
            .objects
            .iter()
            .find(|r| r.logical_digest == digest)
            .ok_or_else(|| {
                runmat_execution_artifact::ArtifactError::Identity(
                    "nonlinear read outside original catalog".into(),
                )
            })?;
        if reference.encoded_length > maximum_bytes {
            return Err(runmat_execution_artifact::ArtifactError::Limit(
                "original nonlinear object exceeds caller read allowance".into(),
            ));
        }
        self.store
            .read_verified_bounded(digest, reference.encoded_length)
    }
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        self.read_verified_bounded(digest, u64::MAX)
    }
}

impl<S: CacheImport + Send + Sync> PublishedObjectSource for OriginalSource<S> {
    fn read_verified(&self, reference: &ValueRef) -> Result<Vec<u8>, String> {
        self.access
            .validate_reference(
                reference,
                ValueRefKind::ResultObject,
                &reference.media_type,
                &reference.value_schema,
            )
            .map_err(string)?;
        if !self.catalog.objects.contains(reference) {
            return Err("nonlinear source read outside original catalog".into());
        }
        let bytes = CacheImport::read_verified(self, reference.logical_digest)
            .map_err(string)?
            .ok_or("nonlinear original child missing")?;
        if bytes.len() as u64 != reference.encoded_length
            || Digest::sha256(&bytes) != reference.logical_digest
        {
            return Err("nonlinear original child differs from reference".into());
        }
        Ok(bytes)
    }
}
impl<S: CacheImport> OriginalSource<S> {
    fn import(
        &self,
        resident: u64,
    ) -> Result<CompletedNonlinearSolidStudy, AnalysisPublicationError> {
        import_nonlinear_solid_study_root(
            self,
            &self.root,
            self.root.logical_digest,
            &self.access,
            self.limits,
            resident,
        )
        .map_err(bridge)
    }
    fn candidate_budget(
        &self,
        candidate: &NonlinearSolidStudyCompletion,
        maximum_workspace_bytes: u64,
    ) -> Result<u64, AnalysisPublicationError> {
        let ceiling = self.maximum_resident_bytes.min(maximum_workspace_bytes);
        if ceiling == 0 {
            return Err(limit("nonlinear source requires admitted workspace"));
        }
        let profile = runmat_canonical_codec::profile_serialized(
            &(&self.root, &self.catalog, &self.access),
            runmat_canonical_codec::CanonicalLimits::content_bounded(
                usize::try_from(ceiling)
                    .map_err(|_| limit("nonlinear workspace exceeds address space"))?,
                65_536,
                64,
            ),
        )
        .map_err(AnalysisPublicationError::from)?;
        let retained = candidate
            .retained_storage_bytes(ceiling)?
            .checked_add(profile.materialization_resident_bytes_estimate)
            .ok_or_else(|| limit("nonlinear source handoff storage overflow"))?;
        ceiling
            .checked_sub(retained)
            .filter(|n| *n > 0)
            .ok_or_else(|| limit("nonlinear runtime source exceeds admitted workspace"))
    }
}
impl<S: CacheImport + Send + Sync> NonlinearSolidStudySource for OriginalSource<S> {
    fn authenticate(
        &self,
        candidate: &NonlinearSolidStudyCompletion,
        maximum_workspace_bytes: u64,
    ) -> Result<(), AnalysisPublicationError> {
        let expected = completion(
            self.import(self.candidate_budget(candidate, maximum_workspace_bytes)?)?,
            self.root.clone(),
            self.catalog.clone(),
        )?;
        if &expected != candidate {
            return Err("nonlinear runtime handoff differs from original cold root".into());
        }
        Ok(())
    }
    fn materialize_block(
        &self,
        candidate: &NonlinearSolidStudyCompletion,
        ordinal: u64,
        additional_retained_bytes: u64,
        maximum_workspace_bytes: u64,
    ) -> Result<runmat_analysis_fea::FeaNonlinearRunResult, AnalysisPublicationError> {
        let available = maximum_workspace_bytes
            .checked_sub(additional_retained_bytes)
            .filter(|n| *n > 0)
            .ok_or_else(|| {
                limit("nonlinear projection retained memory exceeds admitted workspace")
            })?;
        self.authenticate(candidate, available)?;
        let remaining = self.candidate_budget(candidate, available)?;
        let held = self.maximum_resident_bytes - remaining;
        let cold = self.import(remaining)?;
        let projection = cold
            .into_original_block_projection(self, ordinal, held)
            .map_err(bridge)?;
        let recovery_retained = held
            .checked_add(
                projection
                    .retained_storage_bytes(self.maximum_resident_bytes)
                    .map_err(bridge)?,
            )
            .ok_or_else(|| limit("nonlinear projection memory overflow"))?;
        let mut original =
            runmat_analysis_fea::pipeline::nonlinear::recover_original_nonlinear_block(
                &projection.workload,
                &projection.mesh,
                &projection.block.block,
                recovery_retained,
            )
            .map_err(fea)?;
        if projection.block.block.complete {
            if let Some(published) = &projection.observations {
                let mut observations = Vec::new();
                for declaration in &projection.workload.model.sequence.steps[0].observations {
                    let result = runmat_analysis_execution::import_observation_result(
                        self,
                        published,
                        declaration.observation_id(),
                        self.limits,
                    )
                    .map_err(bridge)?;
                    let recovered = original
                        .run
                        .observations
                        .iter()
                        .find(|o| o.observation_id == result.observation_id)
                        .ok_or("published original nonlinear observation absent from recovery")?;
                    if recovered.quantity != result.quantity
                        || recovered.location != result.location
                        || recovered.field.shape != result.field.shape
                        || recovered
                            .field
                            .as_host_f64()
                            .ok_or("original observation is not host FP64")?
                            != result
                                .field
                                .as_host_f64()
                                .ok_or("published observation is not host FP64")?
                    {
                        return Err("published original nonlinear observation differs from original physical fields".into());
                    }
                    observations.push(result);
                }
                original.run.observations = observations;
            }
        }
        Ok(original)
    }
}

fn completion(
    completed: CompletedNonlinearSolidStudy,
    root: ValueRef,
    catalog: runmat_execution::value::PublishedObjectCatalog,
) -> Result<NonlinearSolidStudyCompletion, String> {
    let frozen_project_digest = completed.frozen_project_digest();
    let host_workload_digest = completed
        .closures
        .iter()
        .rev()
        .find(|c| {
            matches!(
                c.phase,
                runmat_analysis_execution::NonlinearSolidStudyResultPhase::Block { .. }
            )
        })
        .ok_or("nonlinear completion omits blocks")?
        .workload
        .identity
        .digest();
    let block_count = completed
        .terminal_block
        .block
        .request
        .block_ordinal
        .checked_add(1)
        .ok_or("nonlinear block count overflow")?;
    Ok(NonlinearSolidStudyCompletion {
        compiled: completed.compiled,
        workload: completed.workload,
        terminal_block: completed.terminal_block.block,
        numerical_receipt: completed.terminal_block.numerical_receipt,
        host_workload_digest,
        block_count,
        observation_catalog: completed.observation_catalog,
        accuracy: completed.accuracy.artifact,
        states: completed.states.into_iter().map(|s| s.state).collect(),
        baseline_usage: completed.baseline_usage.evidence,
        terminal_usage: completed.terminal_usage.evidence,
        publication: completed.publication.result,
        frozen_project_digest,
        root,
        catalog,
    })
}

pub fn nonlinear_solid_study_completion_from_source(
    store: impl CacheImport + Send + Sync + 'static,
    root: ValueRef,
    expected_root: Digest,
    access: ArtifactAccess,
    limits: ObjectInventoryLimits,
    maximum_resident_bytes: u64,
) -> Result<
    (
        NonlinearSolidStudyCompletion,
        NonlinearSolidStudyObjectSource,
    ),
    String,
> {
    let completed = import_nonlinear_solid_study_root(
        &store,
        &root,
        expected_root,
        &access,
        limits,
        maximum_resident_bytes,
    )
    .map_err(string)?;
    let expected = prepare_nonlinear_solid_study_root(&completed).map_err(string)?;
    if expected.root != root {
        return Err("nonlinear imported root is noncanonical".into());
    }
    let catalog = expected.catalog;
    drop(expected.objects);
    let source = OriginalSource {
        root: root.clone(),
        catalog: catalog.clone(),
        access,
        limits,
        maximum_resident_bytes,
        store,
    };
    let dto = completion(completed, root, catalog)?;
    source
        .authenticate(&dto, maximum_resident_bytes)
        .map_err(string)?;
    Ok((dto, NonlinearSolidStudyObjectSource::new(source)))
}
fn string(error: impl std::fmt::Display) -> String {
    error.to_string()
}

/// Persist newly authored root objects into the session's original verified store.
/// The read-only source retains its lifetime after the executing pool is dropped.
pub(crate) fn convert(
    completed: CompletedNonlinearSolidStudy,
    source: crate::NativeProgramObjectSource,
    store: &mut impl runmat_execution_artifact::cache::CacheExport,
) -> Result<
    (
        NonlinearSolidStudyCompletion,
        NonlinearSolidStudyObjectSource,
    ),
    String,
> {
    let access = completed.artifact_access().clone();
    let limits = completed.output_policy().inventory_limits();
    let memory = completed.workload.model.sequence.steps[0]
        .accuracy
        .resources
        .resident_byte_budget();
    let prepared = prepare_nonlinear_solid_study_root(&completed).map_err(string)?;
    for object in &prepared.objects {
        store.write_verified(object).map_err(string)?;
    }
    let root = prepared.root;
    drop(prepared.objects);
    drop(prepared.catalog);
    drop(completed);
    nonlinear_solid_study_completion_from_source(
        source,
        root.clone(),
        root.logical_digest,
        access,
        limits,
        memory,
    )
}
