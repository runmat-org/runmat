use runmat_analysis_fea::study::CompiledAnalysisStudy;
use runmat_execution::Digest;
use runmat_meshing_core::{MeshingEvidence, SolverMeshArtifact};

use super::StructuralStudyNumericalResult;

/// Placement-neutral completed structural study.
///
/// The compiled scientific contract, canonical mesh, governed meshing evidence, and numerical
/// result travel together. Local execution and a verified remote import must construct this same
/// type; paths, worker identities, retry state, and transport authority do not belong here.
#[derive(Debug, Clone)]
pub struct StructuralStudyResult {
    compiled: CompiledAnalysisStudy,
    mesh: SolverMeshArtifact,
    meshing_evidence: MeshingEvidence,
    numerical: StructuralStudyNumericalResult,
}

impl StructuralStudyResult {
    pub fn new(
        compiled: CompiledAnalysisStudy,
        mesh: SolverMeshArtifact,
        meshing_evidence: MeshingEvidence,
        numerical: StructuralStudyNumericalResult,
    ) -> Result<Self, String> {
        compiled
            .validate_against(&mesh)
            .map_err(|error| error.to_string())?;
        meshing_evidence
            .validate(&mesh)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            compiled,
            mesh,
            meshing_evidence,
            numerical,
        })
    }

    pub fn scientific_digest(&self) -> Result<Digest, String> {
        self.compiled
            .scientific_digest()
            .map_err(|error| error.to_string())
    }

    pub const fn compiled(&self) -> &CompiledAnalysisStudy {
        &self.compiled
    }

    pub const fn mesh(&self) -> &SolverMeshArtifact {
        &self.mesh
    }

    pub const fn meshing_evidence(&self) -> &MeshingEvidence {
        &self.meshing_evidence
    }

    pub const fn numerical(&self) -> &StructuralStudyNumericalResult {
        &self.numerical
    }

    pub fn into_parts(
        self,
    ) -> (
        CompiledAnalysisStudy,
        SolverMeshArtifact,
        MeshingEvidence,
        StructuralStudyNumericalResult,
    ) {
        (
            self.compiled,
            self.mesh,
            self.meshing_evidence,
            self.numerical,
        )
    }
}
