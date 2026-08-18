mod artifact;
mod artifact_classification;
mod artifact_identity;
mod artifact_order;
mod artifact_parameters;
mod artifact_types;
mod cancellation;
mod chunk;
mod codec;
mod error;
mod evidence;
mod failure;
mod identity;
mod manifest;
mod metric;
mod request;
mod workload;

pub use artifact::validate_solver_mesh_topology;
pub use artifact_identity::{
    solver_boundary_edge_identity, solver_boundary_face_identity, solver_midside_node_identity,
    solver_volume_element_identity,
};
pub use artifact_parameters::sort_solver_node_exact_parameters;
pub use artifact_types::{
    BoundaryEdgeOrder, BoundaryFaceRole, BoundaryTriangleOrder, ContactPair, FieldTopologyLocation,
    FieldTopologyMap, MaterialInterface, MeshNeighbor, MeshRegion, SolverBoundaryEdge,
    SolverBoundaryFace, SolverMeshArtifact, SolverMeshNode, SolverMeshTopology,
    SolverNodeExactParameter, SolverVolumeElement, ANALYSIS_MESH_ARTIFACT_SCHEMA_VERSION,
    TETRAHEDRON_MIDSIDE_EDGE_CORNERS,
};
pub use cancellation::{CancellationPolicy, MeshingCancellationSignal, NeverCancelled};
pub use chunk::{
    build_chunked_stage_payload, build_closed_stage_manifest, decode_stage_manifest_streams,
    verify_stage_manifest_closure, EncodedMeshingChunk, MeshingChunkPolicy, MeshingChunkStream,
    MeshingChunkedPayload,
};
pub use codec::{CanonicalMeshingContract, MeshingCanonicalLimits};
pub use error::MeshingContractError;
pub use evidence::{
    CacheAdmissionDecision, ErrorDistribution, InvariantEvidence, MeshingEvidence,
    MeshingResourceUsage, MeshingStageEvidence, PlatformBuildIdentity, SizingResolutionEvidence,
    MESHING_EVIDENCE_SCHEMA_VERSION,
};
pub use failure::{
    GeometricWitness, MeshingDiagnosticEntry, MeshingDiagnosticValue, MeshingFailure,
    MeshingFailureCategory, MeshingOperation, MeshingStageKind, MESHING_FAILURE_SCHEMA_VERSION,
};
pub use identity::{
    CanonicalEntityRange, GeometryRevisionRef, MeshingInputKind, MeshingInputRef,
    MeshingJoinIdentity, MeshingPartitionDescriptor, MeshingPartitionIdentity,
    MeshingPartitionKind, MeshingPartitionResultRef, MeshingStageIdentity,
    MeshingStageResultIdentity, MeshingValidationIdentity, StableDigest,
    MESHING_IDENTITY_SCHEMA_VERSION,
};
pub use manifest::{
    MeshingChunkDescriptor, MeshingChunkMediaType, MeshingManifestDisposition,
    MeshingStageManifest, MeshingStageResultKind, MESHING_STAGE_MANIFEST_SCHEMA_VERSION,
};
pub use metric::{
    MetricCombinationRule, MetricContribution, MetricContributionScope, MetricFieldRequest,
    MetricSourceKind, MetricTensor3,
};
pub use request::{
    AlgorithmVersionSet, CurveQualityTargets, ElementOrder, MeshingQualityTargets, MeshingRequest,
    MeshingResourceBudget, SurfaceQualityTargets, VolumeQualityTargets,
    MESHING_REQUEST_SCHEMA_VERSION,
};
pub use runmat_geometry_core::{GeometryTolerancePolicy, PersistentEntityId, PersistentEntityKind};
pub use workload::{
    MeshingCapabilityRequirement, MeshingProgress, MeshingWorkloadRequest, MeshingWorkloadResult,
    MESHING_PROGRESS_SCHEMA_VERSION, MESHING_WORKLOAD_SCHEMA_VERSION,
};

pub(super) fn validate_token(
    field: &str,
    value: &str,
    maximum_bytes: usize,
) -> Result<(), MeshingContractError> {
    if value.is_empty()
        || value.len() > maximum_bytes
        || !value.is_ascii()
        || value.chars().any(char::is_control)
        || value.trim() != value
    {
        return Err(MeshingContractError::invalid(
            field,
            format!("must be 1..={maximum_bytes} printable ASCII bytes without surrounding space"),
        ));
    }
    Ok(())
}

pub(super) fn validate_finite(field: &str, value: f64) -> Result<(), MeshingContractError> {
    if !value.is_finite() {
        return Err(MeshingContractError::invalid(field, "must be finite"));
    }
    Ok(())
}

#[cfg(test)]
mod artifact_tests;
#[cfg(test)]
mod execution_contract_tests;
#[cfg(test)]
mod tests;
