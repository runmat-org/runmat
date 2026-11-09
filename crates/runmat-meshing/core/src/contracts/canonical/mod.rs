mod artifact;
mod artifact_classification;
mod artifact_identity;
mod artifact_order;
mod artifact_parameters;
mod artifact_resources;
mod artifact_storage;
mod artifact_topology;
mod artifact_types;
mod topology_wire;
mod artifact_workspace;
mod authoring;
mod cancellation;
mod chunk;
mod codec;
mod error;
mod evidence;
mod failure;
mod identity;
mod lineage;
mod manifest;
mod metric;
mod request;
mod settings;
mod size_controls;
mod solver_projection;
mod transfer;
mod workload;

pub use artifact::{
    validate_solver_mesh_topology, validate_solver_mesh_topology_with_cancellation,
};
pub use artifact_identity::{
    solver_boundary_edge_identity, solver_boundary_face_identity, solver_midside_node_identity,
    solver_volume_element_identity,
};
pub use artifact_parameters::sort_solver_node_exact_parameters;
pub use artifact_topology::SharedSolverMeshTopology;
pub use artifact_types::{
    BoundaryEdgeOrder, BoundaryFaceRole, BoundaryTriangleOrder, ConformalInterface, ContactPair,
    FieldTopologyLocation, FieldTopologyMap, MeshNeighbor, MeshRegion, SolverBoundaryEdge,
    SolverBoundaryFace, SolverMeshArtifact, SolverMeshNode, SolverMeshTopology,
    SolverNodeExactParameter, SolverVolumeElement, SOLVER_MESH_ARTIFACT_SCHEMA_VERSION,
    TETRAHEDRON_MIDSIDE_EDGE_CORNERS,
};
pub use authoring::MeshDefinition;
pub use cancellation::{CancellationPolicy, MeshingCancellationSignal, NeverCancelled};
pub use chunk::{
    build_chunked_stage_payload, build_chunked_stage_payload_with_cancellation,
    build_closed_stage_manifest, close_stage_payload_with_admission, decode_stage_manifest_streams,
    decode_stage_manifest_streams_borrowed, decode_stage_manifest_streams_borrowed_with_admission,
    decode_stage_manifest_streams_with_cancellation, profile_chunked_stage_payload,
    verify_stage_manifest_closure, verify_stage_manifest_closure_borrowed,
    verify_stage_manifest_closure_borrowed_with_admission, ClosedMeshingStage, EncodedMeshingChunk,
    EncodedMeshingChunkRef, MeshingChunkPolicy, MeshingChunkStorageProfile, MeshingChunkStream,
    MeshingChunkedPayload, MeshingManifestMetadata, MeshingStreamStorageProfile,
};
pub use codec::{CanonicalMeshingContract, MeshingCanonicalCodecControl, MeshingCanonicalLimits};
pub use error::MeshingContractError;
pub use evidence::{
    AchievedEdgeLengths, CacheAdmissionDecision, ErrorDistribution, InvariantEvidence,
    MeshingEvidence, MeshingResourceUsage, MeshingStageEvidence, PlatformBuildIdentity,
    ReusedMeshingStageObservation, ReusedMeshingStageOrigin, SizingResolutionEvidence,
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
pub use lineage::{
    SolverMeshAdaptationCell, SolverMeshAdaptationKind, SolverMeshAdaptationLineage,
    SolverMeshAdaptationMark, SolverMeshAdaptationMutation,
    SOLVER_MESH_ADAPTATION_LINEAGE_SCHEMA_VERSION,
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
    AlgorithmVersionSet, CurveQualityTargets, ElementOrder, MeshDimension,
    MeshingExecutionCeilings, MeshingQualityTargets, MeshingRequest, MeshingResourceBudget,
    MeshingSizeLimits, SurfaceQualityTargets, VolumeQualityTargets, MESHING_REQUEST_SCHEMA_VERSION,
};
pub use runmat_geometry_core::{GeometryTolerancePolicy, PersistentEntityId, PersistentEntityKind};
pub use settings::{
    production_algorithm_versions, resolve_meshing_request, CrossSectionMeshSettings,
    MeshingRequestSettings,
};
pub use size_controls::{
    size_control_evidence, validate_size_controls, AppliedSizeControl, MeshSizeControl,
    MeshSizeControlTarget, NamedRegionEntities,
};
pub use solver_projection::{
    SolverElevationReport, SolverMeshProjection, SolverMeshValidation,
    ValidatedSolverMeshProjection, ValidatedSolverMeshTopology,
    SOLVER_MESH_PROJECTION_SCHEMA_VERSION, SOLVER_MESH_VALIDATION_SCHEMA_VERSION,
};
pub use transfer::{
    SolverEntityTransfer, SolverMeshTransferMap, SolverTransferMethod, SolverTransferSource,
    SOLVER_MESH_TRANSFER_SCHEMA_VERSION,
};
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
