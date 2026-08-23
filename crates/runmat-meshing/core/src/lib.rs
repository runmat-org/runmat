extern crate self as runmat_meshing_core;

pub mod contracts;
pub mod fixtures;
pub mod quality;
pub use quality::{predicate, spatial_index, tolerance};
pub mod validation;

pub use contracts::artifact::{
    analysis_mesh_field_topology, AnalysisBoundaryEdge, AnalysisBoundaryFace,
    AnalysisFieldTopologyDescriptor, AnalysisFieldTopologyLocation, AnalysisMeshArtifact,
    AnalysisMeshNode, AnalysisVolumeElement, MeshBackendSummary,
    ANALYSIS_MESH_BOUNDARY_EDGE_TOPOLOGY_ID, ANALYSIS_MESH_BOUNDARY_FACE_TOPOLOGY_ID,
    ANALYSIS_MESH_FIELD_TOPOLOGY_ID, TETRAHEDRON4_FIELD_ELEMENT_KIND, TRI3_FIELD_ELEMENT_KIND,
};
pub use contracts::backend::{select_volume_backend, MeshBackendKind, MeshBackendSelection};
pub use contracts::boundary::{BoundaryMeshInput, BoundaryMeshTriangle};
pub use contracts::canonical::{
    build_chunked_stage_payload, build_closed_stage_manifest, decode_stage_manifest_streams,
    solver_boundary_edge_identity, solver_boundary_face_identity, solver_midside_node_identity,
    solver_volume_element_identity, sort_solver_node_exact_parameters,
    validate_solver_mesh_topology, verify_stage_manifest_closure, AlgorithmVersionSet,
    BoundaryEdgeOrder, BoundaryFaceRole, BoundaryTriangleOrder, CacheAdmissionDecision,
    CancellationPolicy, CanonicalEntityRange, CanonicalMeshingContract, ContactPair,
    CurveQualityTargets, ElementOrder, EncodedMeshingChunk, ErrorDistribution,
    FieldTopologyLocation, FieldTopologyMap, GeometricWitness, GeometryRevisionRef,
    GeometryTolerancePolicy, InvariantEvidence, MaterialInterface, MeshNeighbor, MeshRegion,
    MeshingCancellationSignal, MeshingCanonicalLimits, MeshingCapabilityRequirement,
    MeshingChunkDescriptor, MeshingChunkMediaType, MeshingChunkPolicy, MeshingChunkStream,
    MeshingChunkedPayload, MeshingContractError, MeshingDiagnosticEntry, MeshingDiagnosticValue,
    MeshingDomainModel, MeshingEvidence, MeshingFailure, MeshingFailureCategory, MeshingInputKind,
    MeshingInputRef, MeshingJoinIdentity, MeshingManifestDisposition, MeshingOperation,
    MeshingPartitionDescriptor, MeshingPartitionIdentity, MeshingPartitionKind,
    MeshingPartitionResultRef, MeshingProgress, MeshingQualityTargets, MeshingRequest,
    MeshingResourceBudget, MeshingResourceUsage, MeshingStageEvidence, MeshingStageIdentity,
    MeshingStageKind, MeshingStageManifest, MeshingStageResultIdentity, MeshingStageResultKind,
    MeshingValidationIdentity, MeshingWorkloadRequest, MeshingWorkloadResult,
    MetricCombinationRule, MetricContribution, MetricContributionScope, MetricFieldRequest,
    MetricSourceKind, MetricTensor3, NeverCancelled, PersistentEntityId, PersistentEntityKind,
    PlatformBuildIdentity, RegionMaterialAssignment, SizingResolutionEvidence, SolverBoundaryEdge,
    SolverBoundaryFace, SolverEntityTransfer, SolverMeshAdaptationCell, SolverMeshAdaptationKind,
    SolverMeshAdaptationLineage, SolverMeshAdaptationMark, SolverMeshAdaptationMutation,
    SolverMeshArtifact, SolverMeshNode, SolverMeshProjection, SolverMeshTopology,
    SolverMeshTransferMap, SolverMeshValidation, SolverNodeExactParameter, SolverTransferMethod,
    SolverTransferSource, SolverVolumeElement, StableDigest, SurfaceQualityTargets,
    VolumeQualityTargets, ANALYSIS_MESH_ARTIFACT_SCHEMA_VERSION, MESHING_DOMAIN_MODEL_MEDIA_TYPE,
    MESHING_DOMAIN_MODEL_SCHEMA_VERSION, MESHING_EVIDENCE_SCHEMA_VERSION,
    MESHING_FAILURE_SCHEMA_VERSION, MESHING_IDENTITY_SCHEMA_VERSION,
    MESHING_PROGRESS_SCHEMA_VERSION, MESHING_REQUEST_SCHEMA_VERSION,
    MESHING_STAGE_MANIFEST_SCHEMA_VERSION, MESHING_WORKLOAD_SCHEMA_VERSION,
    SOLVER_MESH_ADAPTATION_LINEAGE_SCHEMA_VERSION, SOLVER_MESH_PROJECTION_SCHEMA_VERSION,
    SOLVER_MESH_TRANSFER_SCHEMA_VERSION, SOLVER_MESH_VALIDATION_SCHEMA_VERSION,
    TETRAHEDRON_MIDSIDE_EDGE_CORNERS,
};
pub use contracts::options::{
    AdaptiveMeshingOptions, MeshElementOrder, MeshKindRequest, MeshProfile, MeshRefinementOptions,
    MeshTargetSize, MeshValidationPolicyOptions, RefinementConvergenceOptions,
    RefinementFocusLevel, RefinementFocusOptions, RefinementIndicatorMode,
    RefinementIndicatorOverrides, RefinementStrategy, VolumeMeshingOptions,
};
pub use contracts::provenance::{AnalysisMeshProvenance, MeshEntityProvenance, SourceEntityKind};
pub use contracts::topology::{BoundaryElementKind, VolumeElementKind};
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
pub use quality::{AnalysisMeshQualityReport, ElementQuality, QualityThresholds};
pub use runmat_meshing_size::adaptive::{
    build_refinement_markers_from_samples, default_refinement_indicators_for_analysis,
    evaluate_adaptive_convergence, plan_refinement_indicators,
    structural_static_default_refinement_indicators, AdaptiveConvergenceMetrics,
    AdaptiveConvergenceStatus, AdaptiveIterationSummary, RefinementIndicatorAvailability,
    RefinementIndicatorKey, RefinementIndicatorSample, RefinementIndicatorStatus,
    RefinementIndicatorSummary, RefinementMarker, RefinementMarkerError, RefinementMarkerOptions,
    SizingFieldUpdate,
};
pub use runmat_meshing_size::field::{
    AnisotropicSizingSample, MeshSizingField, PointSizingQuery, SegmentSizingQuery,
    SizingFieldService, SizingQueryResult, SizingQuerySource, SizingSample,
    SizingSampleApplication, SizingSampleRejection,
};
pub use spatial_index::{Aabb3, LinearSpatialIndex, SpatialEntry, UniformGridSpatialIndex};
pub use tolerance::MeshingTolerance;
pub use validation::{
    analysis_mesh_validation_error_code, validate_analysis_mesh,
    validate_analysis_mesh_with_options, volume_component_count, AnalysisMeshValidationError,
    AnalysisMeshValidationOptions,
};
