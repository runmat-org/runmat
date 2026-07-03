mod artifact;
#[cfg(feature = "dev-evidence")]
pub mod dev_traces;
pub mod summaries;

pub use artifact::{
    build_mesh_evidence_artifact, build_mesh_evidence_artifact_with_validation_evidence,
    MeshBoundaryRecoveryEvidence, MeshCadEvidence, MeshEvidenceArtifact,
    MeshTetrahedronRecoveryEvidence, MeshValidationEvidence, MESH_EVIDENCE_SCHEMA_VERSION,
};

#[cfg(feature = "dev-evidence")]
pub use dev_traces::{build_mesh_evidence_artifact_with_debug, MeshDebugEvent, MeshDebugEvidence};

pub use summaries::{
    MeshAdaptiveEvidence, MeshQualityEvidence, MeshRegionEvidence, MeshSizingEvidence,
    MeshTopologyEvidence,
};

#[cfg(test)]
mod tests;
