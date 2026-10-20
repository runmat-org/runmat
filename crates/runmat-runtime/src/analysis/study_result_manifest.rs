//! Physics-neutral whole-study evidence envelope shared by typed publication adapters.
use super::AnalysisStudyRunData;
use runmat_analysis_fea::study::CompiledAnalysisStudy;
use runmat_execution::Digest;
use runmat_meshing_core::{MeshingEvidence, SolverMeshArtifact, StableDigest};

/// The generic axes retain each physics owner's actual workload and numerical evidence schema.
/// Field names and ordering preserve the existing structural typed-result JSON contract.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct StudyResultManifest<Workloads, NumericalEvidence> {
    schema_version: u16,
    study_definition_digest: Digest,
    compiled: CompiledAnalysisStudy,
    workload_digest: Digest,
    workloads: Workloads,
    solver_mesh_digest: StableDigest,
    meshing_evidence: MeshingEvidence,
    numerical_evidence: NumericalEvidence,
    public_run: AnalysisStudyRunData,
}

pub(super) fn load_for_reassessment(
    definition: Digest,
    path: &std::path::Path,
    maximum_bytes: u64,
) -> Result<
    (CompiledAnalysisStudy, AnalysisStudyRunData),
    super::structural_study::StudyEvidenceReadError,
> {
    let store = super::storage::capture_artifact_store()?;
    let payload = store
        .load_publication_manifest_path(path, maximum_bytes)
        .map_err(|e| match e {
            super::storage::AssessmentStoreError::ResourceLimit => {
                super::structural_study::StudyEvidenceReadError::ResourceLimit
            }
            super::storage::AssessmentStoreError::Invalid(message) => {
                super::structural_study::StudyEvidenceReadError::Invalid(message)
            }
        })?;
    // Opaque family-owned numerical values are not duplicated into an assessment input. The
    // evidence envelope authenticates them; scalar lookup uses the individual run's store.
    let manifest: StudyResultManifest<serde::de::IgnoredAny, serde::de::IgnoredAny> =
        serde_json::from_value(payload).map_err(|error| error.to_string())?;
    if manifest.schema_version != 1
        || manifest.study_definition_digest != definition
        || manifest
            .compiled
            .definition_digest()
            .map_err(|error| error.to_string())?
            != definition
        || manifest.public_run.study_fingerprint != definition.to_string()
        || manifest.public_run.study_id != manifest.compiled.study_id
        || manifest.public_run.model_id != manifest.compiled.model.model_id.0
        || manifest.solver_mesh_digest != manifest.compiled.solver_mesh_digest
    {
        return Err("saved study publication differs from its compiled revision".into());
    }
    Ok((manifest.compiled, manifest.public_run))
}

impl<Workloads, NumericalEvidence> StudyResultManifest<Workloads, NumericalEvidence> {
    pub(super) fn compiled(&self) -> &CompiledAnalysisStudy {
        &self.compiled
    }
    pub(super) fn set_public_run(&mut self, public_run: AnalysisStudyRunData) {
        self.public_run = public_run;
    }
    pub(super) fn new(
        compiled: &CompiledAnalysisStudy,
        workload_digest: Digest,
        workloads: &Workloads,
        mesh: &SolverMeshArtifact,
        meshing_evidence: &MeshingEvidence,
        numerical_evidence: NumericalEvidence,
        publication: &mut super::structural_study::publication::DeferredStudyRun,
    ) -> Result<
        (Self, super::storage::publication::OwnedCopyReservation),
        super::storage::AnalysisPublicationError,
    >
    where
        Workloads: serde::Serialize + Clone,
        NumericalEvidence: serde::Serialize,
    {
        let public_run = &publication.published.data;
        let reservation = publication.staging.admit_compiled_manifest_copy(
            compiled,
            &(workloads, meshing_evidence, &numerical_evidence, public_run),
        )?;
        let definition = compiled
            .definition_digest()
            .map_err(|error| error.to_string())?;
        if public_run.study_fingerprint != definition.to_string() {
            return Err("study result manifest differs from published definition identity".into());
        }
        Ok((
            Self {
                schema_version: 1,
                study_definition_digest: definition,
                compiled: compiled.clone(),
                workload_digest,
                workloads: workloads.clone(),
                solver_mesh_digest: mesh.canonical_digest,
                meshing_evidence: meshing_evidence.clone(),
                numerical_evidence,
                public_run: public_run.clone(),
            },
            reservation,
        ))
    }
}

/// Read the original authenticated publication for saved-result presentation.
/// This does not execute, reassess, or qualify it under a newer implementation.
pub fn read_saved_study_publication(
    definition: Digest,
    path: &std::path::Path,
    maximum_bytes: u64,
) -> Result<AnalysisStudyRunData, String> {
    load_for_reassessment(definition, path, maximum_bytes)
        .map(|(_, publication)| publication)
        .map_err(|error| format!("{error:?}"))
}

/// A solver state held by a study manifest by reference. The state itself is persisted once as
/// canonical binary in the run's state store; the manifest names it by identity and digest.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PersistedStateEvidence {
    state_id: runmat_analysis_core::StateId,
    identity: runmat_analysis_fea::execution::StateIdentity,
    content_digest: Digest,
    topology_digest: Digest,
    artifact_path: String,
}

impl PersistedStateEvidence {
    pub(super) fn persist(
        state: &runmat_analysis_fea::execution::CanonicalSemanticState,
        study_digest: Digest,
        staging: &super::storage::AnalysisPublicationBuilder,
    ) -> Result<Self, super::storage::AnalysisPublicationError> {
        Ok(Self {
            artifact_path: staging.persist_state(&study_digest.to_string(), state)?,
            state_id: state.state_id.clone(),
            identity: state.identity,
            content_digest: state.content_digest,
            topology_digest: state.topology_digest,
        })
    }
}
