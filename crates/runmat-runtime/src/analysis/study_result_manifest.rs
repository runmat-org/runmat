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

impl<Workloads, NumericalEvidence> StudyResultManifest<Workloads, NumericalEvidence> {
    pub(super) fn new(
        compiled: &CompiledAnalysisStudy,
        workload_digest: Digest,
        workloads: Workloads,
        mesh: &SolverMeshArtifact,
        meshing_evidence: &MeshingEvidence,
        numerical_evidence: NumericalEvidence,
        public_run: AnalysisStudyRunData,
    ) -> Result<Self, String> {
        let definition = compiled
            .definition_digest()
            .map_err(|error| error.to_string())?;
        if public_run.study_fingerprint != definition.to_string() {
            return Err("study result manifest differs from published definition identity".into());
        }
        Ok(Self {
            schema_version: 1,
            study_definition_digest: definition,
            compiled: compiled.clone(),
            workload_digest,
            workloads,
            solver_mesh_digest: mesh.canonical_digest,
            meshing_evidence: meshing_evidence.clone(),
            numerical_evidence,
            public_run,
        })
    }
}

impl<Workloads: serde::Serialize, NumericalEvidence: serde::Serialize>
    StudyResultManifest<Workloads, NumericalEvidence>
{
    pub(super) fn persist(&self) -> Result<String, String> {
        let value = serde_json::to_value(self)
            .map_err(|error| format!("failed to encode typed study result manifest: {error}"))?;
        super::structural_study::persist_content_addressed_study_evidence(
            &self.study_definition_digest.to_string(),
            "study-result-manifest",
            value,
        )
    }
}
