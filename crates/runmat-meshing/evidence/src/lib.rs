mod artifact;
mod authoring;
mod cad;
#[cfg(feature = "dev-evidence")]
pub mod dev_traces;
pub mod summaries;
mod tetrahedron_recovery;
mod validation;

pub use artifact::{
    build_mesh_evidence_artifact, build_mesh_evidence_artifact_with_validation_evidence,
    MeshEvidenceArtifact, MESH_EVIDENCE_SCHEMA_VERSION,
};
pub use authoring::{
    build_mesh_authoring_summary, MeshAuthoringBoundaryRegion, MeshAuthoringMaterialRegion,
    MeshAuthoringNestedTetrahedronShellSummary, MeshAuthoringQualitySummary,
    MeshAuthoringRecoverySummary, MeshAuthoringRegionSummary, MeshAuthoringSummary,
    MeshAuthoringTopologySummary, MESH_AUTHORING_SUMMARY_SCHEMA_VERSION,
};
pub use cad::MeshCadEvidence;

#[cfg(feature = "dev-evidence")]
pub use dev_traces::{build_mesh_evidence_artifact_with_debug, MeshDebugEvent, MeshDebugEvidence};

pub use summaries::{
    MeshAdaptiveEvidence, MeshQualityEvidence, MeshRegionEvidence, MeshSizingEvidence,
    MeshTopologyEvidence,
};
pub use tetrahedron_recovery::MeshTetrahedronRecoveryEvidence;
pub use validation::{MeshBoundaryRecoveryEvidence, MeshValidationEvidence};

#[cfg(test)]
mod tests;
