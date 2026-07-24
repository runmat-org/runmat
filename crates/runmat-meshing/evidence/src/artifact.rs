use serde::{Deserialize, Serialize};

use runmat_meshing_core::{
    contracts::{AnalysisMeshArtifact, MeshBackendSummary},
    validation::AnalysisMeshValidationOptions,
};

pub const MESH_EVIDENCE_SCHEMA_VERSION: &str = "mesh-evidence/v1";

use crate::cad::{cad_evidence, MeshCadEvidence};
use crate::summaries::{
    adaptive_evidence, quality_evidence, region_evidence, sizing_evidence, topology_evidence,
    MeshAdaptiveEvidence, MeshQualityEvidence, MeshRegionEvidence, MeshSizingEvidence,
    MeshTopologyEvidence,
};
use crate::tetrahedron_recovery::{tetrahedron_recovery_evidence, MeshTetrahedronRecoveryEvidence};
use crate::validation::{
    validation_evidence, validation_options_from_evidence, MeshValidationEvidence,
};
#[cfg(feature = "dev-evidence")]
use crate::MeshDebugEvidence;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshEvidenceArtifact {
    pub schema_version: String,
    pub mesh_id: String,
    pub backend: MeshBackendSummary,
    #[cfg(feature = "dev-evidence")]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub debug: Option<MeshDebugEvidence>,
    #[serde(default)]
    pub cad: MeshCadEvidence,
    pub topology: MeshTopologyEvidence,
    #[serde(default)]
    pub adaptive: MeshAdaptiveEvidence,
    pub sizing: MeshSizingEvidence,
    pub quality: MeshQualityEvidence,
    #[serde(default)]
    pub tetrahedron_recovery: MeshTetrahedronRecoveryEvidence,
    pub regions: MeshRegionEvidence,
    pub validation: MeshValidationEvidence,
}

pub fn build_mesh_evidence_artifact(
    mesh: &AnalysisMeshArtifact,
    validation: &AnalysisMeshValidationOptions,
) -> MeshEvidenceArtifact {
    build_mesh_evidence_artifact_with_validation_evidence(
        mesh,
        validation_evidence(mesh, validation),
    )
}

pub fn build_mesh_evidence_artifact_with_validation_evidence(
    mesh: &AnalysisMeshArtifact,
    validation: MeshValidationEvidence,
) -> MeshEvidenceArtifact {
    let validation = validation_evidence(mesh, &validation_options_from_evidence(&validation));
    MeshEvidenceArtifact {
        schema_version: MESH_EVIDENCE_SCHEMA_VERSION.to_string(),
        mesh_id: mesh.mesh_id.clone(),
        backend: mesh.backend.clone(),
        #[cfg(feature = "dev-evidence")]
        debug: None,
        cad: cad_evidence(mesh),
        topology: topology_evidence(mesh),
        adaptive: adaptive_evidence(mesh),
        sizing: sizing_evidence(mesh),
        quality: quality_evidence(mesh),
        tetrahedron_recovery: tetrahedron_recovery_evidence(mesh),
        regions: region_evidence(mesh),
        validation,
    }
}
