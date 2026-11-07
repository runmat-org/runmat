/// Actual package build configuration used for internal artifact reuse.
#[doc(hidden)]
pub const BUILD_CONFIGURATION: &str = env!("RUNMAT_BUILD_CONFIGURATION");

extern crate self as runmat_meshing_core;

pub mod contracts;
pub mod fixtures;
pub mod interpolation;
pub mod quality;
pub use contracts::canonical::{
    build_chunked_stage_payload, build_chunked_stage_payload_with_cancellation,
    build_closed_stage_manifest, close_stage_payload_with_admission, decode_stage_manifest_streams,
    decode_stage_manifest_streams_borrowed, decode_stage_manifest_streams_borrowed_with_admission,
    decode_stage_manifest_streams_with_cancellation, production_algorithm_versions,
    profile_chunked_stage_payload, resolve_meshing_request, size_control_evidence,
    solver_boundary_edge_identity, solver_boundary_face_identity, solver_midside_node_identity,
    solver_volume_element_identity, sort_solver_node_exact_parameters, validate_size_controls,
    validate_solver_mesh_topology, validate_solver_mesh_topology_with_cancellation,
    verify_stage_manifest_closure, verify_stage_manifest_closure_borrowed,
    verify_stage_manifest_closure_borrowed_with_admission, AchievedEdgeLengths,
    AlgorithmVersionSet, AppliedSizeControl, BoundaryEdgeOrder, BoundaryFaceRole,
    BoundaryTriangleOrder, CacheAdmissionDecision, CancellationPolicy, CanonicalEntityRange,
    CanonicalMeshingContract, ClosedMeshingStage, ConformalInterface, ContactPair,
    CurveQualityTargets, ElementOrder, EncodedMeshingChunk, EncodedMeshingChunkRef,
    ErrorDistribution, FieldTopologyLocation, FieldTopologyMap, GeometricWitness,
    GeometryRevisionRef, GeometryTolerancePolicy, InvariantEvidence, MeshDefinition, MeshDimension,
    MeshNeighbor, MeshRegion, MeshSizeControl, MeshSizeControlTarget, MeshingCancellationSignal,
    MeshingCanonicalCodecControl, MeshingCanonicalLimits, MeshingCapabilityRequirement,
    MeshingChunkDescriptor, MeshingChunkMediaType, MeshingChunkPolicy, MeshingChunkStorageProfile,
    MeshingChunkStream, MeshingChunkedPayload, MeshingContractError, MeshingDiagnosticEntry,
    MeshingDiagnosticValue, MeshingEvidence, MeshingExecutionCeilings, MeshingFailure,
    MeshingFailureCategory, MeshingInputKind, MeshingInputRef, MeshingJoinIdentity,
    MeshingManifestDisposition, MeshingManifestMetadata, MeshingOperation,
    MeshingPartitionDescriptor, MeshingPartitionIdentity, MeshingPartitionKind,
    MeshingPartitionResultRef, MeshingProgress, MeshingQualityTargets, MeshingRequest,
    MeshingRequestSettings, MeshingResourceBudget, MeshingResourceUsage, MeshingSizeLimits,
    MeshingStageEvidence, MeshingStageIdentity, MeshingStageKind, MeshingStageManifest,
    MeshingStageResultIdentity, MeshingStageResultKind, MeshingStreamStorageProfile,
    MeshingValidationIdentity, MeshingWorkloadRequest, MeshingWorkloadResult,
    MetricCombinationRule, MetricContribution, MetricContributionScope, MetricFieldRequest,
    MetricSourceKind, MetricTensor3, NamedRegionEntities, NeverCancelled, PersistentEntityId,
    PersistentEntityKind, PlatformBuildIdentity, ReusedMeshingStageObservation,
    ReusedMeshingStageOrigin, SharedSolverMeshTopology, SizingResolutionEvidence,
    SolverBoundaryEdge, SolverBoundaryFace, SolverElevationReport, SolverEntityTransfer,
    SolverMeshAdaptationCell, SolverMeshAdaptationKind, SolverMeshAdaptationLineage,
    SolverMeshAdaptationMark, SolverMeshAdaptationMutation, SolverMeshArtifact, SolverMeshNode,
    SolverMeshProjection, SolverMeshTopology, SolverMeshTransferMap, SolverMeshValidation,
    SolverNodeExactParameter, SolverTransferMethod, SolverTransferSource, SolverVolumeElement,
    StableDigest, SurfaceQualityTargets, ValidatedSolverMeshProjection,
    ValidatedSolverMeshTopology, VolumeQualityTargets, MESHING_EVIDENCE_SCHEMA_VERSION,
    MESHING_FAILURE_SCHEMA_VERSION, MESHING_IDENTITY_SCHEMA_VERSION,
    MESHING_PROGRESS_SCHEMA_VERSION, MESHING_REQUEST_SCHEMA_VERSION,
    MESHING_STAGE_MANIFEST_SCHEMA_VERSION, MESHING_WORKLOAD_SCHEMA_VERSION,
    SOLVER_MESH_ADAPTATION_LINEAGE_SCHEMA_VERSION, SOLVER_MESH_ARTIFACT_SCHEMA_VERSION,
    SOLVER_MESH_PROJECTION_SCHEMA_VERSION, SOLVER_MESH_TRANSFER_SCHEMA_VERSION,
    SOLVER_MESH_VALIDATION_SCHEMA_VERSION, TETRAHEDRON_MIDSIDE_EDGE_CORNERS,
};
pub use contracts::{
    validate_meshing_stage_order, CadEdgeContract, CadEvaluatorCapabilities, CadFaceContract,
    CadModel, CadShellContract, CadVertexContract, CadVolumeContract, CurveMesh, CurveMeshElement,
    CurveMeshNode, MeshingStage, MeshingStageArtifacts, MeshingStageContractError, PlcFacet,
    PlcNode, PlcProtectedEdge, PlcValidationSummary, ProtectedBoundaryComplex, SizingFieldContract,
    SolveReadinessReport, StageEvidence, StageEvidenceStatus,
    SurfaceCadCurveBoundaryEdgeProvenance, SurfaceCadCurveBoundaryProvenance,
    SurfaceCurveBoundaryValidation, SurfaceLoopCoverage, SurfaceMesh, SurfaceMeshNode,
    SurfaceMeshTriangle, Tetrahedron4Element, TetrahedronBoundaryFace, TetrahedronMesh,
    TetrahedronMeshNode, TopologyEntityId, MESHING_CONTRACT_SCHEMA_VERSION,
};
pub use predicate::{
    closest_point_on_triangle, distance, distance_squared, point_in_closed_triangle_surface,
    point_triangle_distance, ray_triangle_intersection, tetrahedron_centroid,
    tetrahedron_circumsphere, tetrahedron_circumsphere_contains_point,
    tetrahedron_edge_aspect_ratio, tetrahedron_signed_volume, tetrahedron_volume, triangle_area,
    triangle_centroid, PlanarPredicateError, PlanarPredicatePoint, Point3, PointInClosedSurface,
    PredicateSign, RayTriangleHit, Tetrahedron3, Triangle3,
};
pub use quality::boundary::{
    evaluate_boundary_quality_candidate, BoundaryQualityCandidateConstraints,
    BoundaryQualityCandidateError, BoundaryQualityCandidateEvaluation,
    BoundaryQualityCandidateOptions, BoundaryQualityCandidateRejectionReason,
};
pub use quality::boundary_face::{
    boundary_face_rule, boundary_face_validity, is_valid_boundary_face, BoundaryFaceRule,
    BoundaryFaceValidity, BOUNDARY_FACE_MINIMUM_NORMAL_JACOBIAN_RATIO, GAUSS_LEGENDRE_EIGHT_UNIT,
    GAUSS_LEGENDRE_SIXTEEN_UNIT, GAUSS_LEGENDRE_TWELVE_UNIT,
};
pub use quality::tet10_element::{
    is_valid_tetrahedron10_element, tensor_tetrahedron_rule, tetrahedron10_is_affine,
    tetrahedron10_jacobian_margin, tetrahedron10_jacobian_margin_for_category,
    tetrahedron10_jacobian_margin_with_capacity, tetrahedron10_reference_gradients,
    tetrahedron10_stiffness_moments, tetrahedron10_stiffness_moments_for_category,
    tetrahedron10_stiffness_moments_with_capacity, Tet10CapacityError, Tet10ElementInvalidity,
    Tet10JacobianMargin, Tet10StiffnessMoments, Tet10StiffnessRule,
    STIFFNESS_MOMENT_RELATIVE_TOLERANCE, TET10_GRADIENT_MOMENT_COUNT, TET10_MINIMUM_JACOBIAN_RATIO,
    TETRAHEDRON_FOUR_POINT_RULE,
};
pub use quality::ElementQuality;
pub use quality::{predicate, spatial_index, tolerance};
pub use runmat_meshing_size::adaptive::{
    build_refinement_markers_from_samples, evaluate_adaptive_convergence,
    plan_refinement_indicators, AdaptiveConvergenceMetrics, AdaptiveConvergenceStatus,
    AdaptiveIterationSummary, RefinementIndicatorAvailability, RefinementIndicatorKey,
    RefinementIndicatorSample, RefinementIndicatorStatus, RefinementIndicatorSummary,
    RefinementMarker, RefinementMarkerError, RefinementMarkerOptions, SizingFieldUpdate,
};
pub use runmat_meshing_size::field::{
    AnisotropicSizingSample, MeshSizingField, PointSizingQuery, SegmentSizingQuery,
    SizingFieldService, SizingQueryResult, SizingQuerySource, SizingSample,
    SizingSampleApplication, SizingSampleRejection,
};
pub use spatial_index::{
    Aabb3, BoundsSpatialIndex, LinearSpatialIndex, PointSpatialIndex, SpatialEntry, SpatialPoint,
    UniformGridSpatialIndex,
};
pub use tolerance::MeshingTolerance;

/// Build settings of the shared metric and canonical-codec dependencies.
#[doc(hidden)]
pub fn numerical_dependency_build_configuration() -> [&'static str; 2] {
    [
        runmat_meshing_size::BUILD_CONFIGURATION,
        runmat_canonical_codec::BUILD_CONFIGURATION,
    ]
}
