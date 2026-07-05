extern crate self as runmat_meshing_core;

pub use runmat_meshing_cad as cad;
pub mod contracts;
pub use runmat_meshing_curve as curve;
pub mod fixtures;
pub use runmat_meshing_opt as opt;
pub mod quality;
pub use quality::{predicate, spatial_index, tolerance};
pub use runmat_meshing_size as size;
pub mod source_topology {
    pub use runmat_meshing_cad::topology::{
        extract_source_topology, SourceTopologyEdge, SourceTopologyError, SourceTopologyFace,
        SourceTopologyModel, SourceTopologyVertex,
    };
}
pub use runmat_meshing_surface as surface;
pub mod validation;

pub use cad::eval::{
    build_cad_evaluation_model, build_cad_evaluation_model_with_provider, project_to_face,
    summarize_cad_evaluation, CadEvaluationError, CadEvaluationModel, CadEvaluationReport,
    CadEvaluationSource, CadFaceEvaluationFrame, CadFaceEvaluationRequest,
    CadFaceEvaluatorProvider, CadFaceProjection, NoopCadFaceEvaluatorProvider,
};
pub use cad::topology::{
    build_cad_topology, validate_cad_topology_model, CadEdge, CadEntityId, CadEntityKind, CadFace,
    CadLoop, CadShell, CadTopologyError, CadTopologyModel, CadTopologyReport, CadTopologySource,
    CadVertex, CadVolume,
};
pub use cad::{CadCurveEvaluationSample, CadCurveEvaluationSampleSource};
pub use contracts::artifact::{
    analysis_mesh_field_topology, AnalysisBoundaryEdge, AnalysisBoundaryFace,
    AnalysisFieldTopologyDescriptor, AnalysisFieldTopologyLocation, AnalysisMeshArtifact,
    AnalysisMeshNode, AnalysisVolumeElement, MeshBackendSummary,
    ANALYSIS_MESH_BOUNDARY_EDGE_TOPOLOGY_ID, ANALYSIS_MESH_BOUNDARY_FACE_TOPOLOGY_ID,
    ANALYSIS_MESH_FIELD_TOPOLOGY_ID, TETRAHEDRON4_FIELD_ELEMENT_KIND, TRI3_FIELD_ELEMENT_KIND,
};
pub use contracts::backend::{select_volume_backend, MeshBackendKind, MeshBackendSelection};
pub use contracts::boundary::{BoundaryMeshInput, BoundaryMeshTriangle};
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
pub use curve::{
    discretize_cad_topology_curves_with_sizing,
    discretize_cad_topology_curves_with_sizing_and_provider, discretize_topology_curves,
    CadCurveDiscretization, CadCurveEdgeProvenance, CadCurveEvaluationRequest,
    CadCurveEvaluatorProvider, CurveDiscretization, CurveDiscretizationError,
    CurveDiscretizationOptions, CurveElement, CurveNode, NoopCadCurveEvaluatorProvider,
};
pub use opt::sliver::recovery::{
    classify_sliver_tetrahedra, evaluate_sliver_removal, SliverClassification,
    SliverClassificationReason, SliverRecoveryError, SliverRecoveryOptions,
    SliverRemovalEvaluation, SliverRemovalRejectionReason, SliverTetrahedronQuality,
};
pub use predicate::{
    closest_point_on_triangle, distance, distance_squared, point_in_closed_triangle_surface,
    point_triangle_distance, ray_triangle_intersection, tetrahedron_centroid,
    tetrahedron_circumsphere, tetrahedron_circumsphere_contains_point,
    tetrahedron_edge_aspect_ratio, tetrahedron_signed_volume, tetrahedron_volume, triangle_area,
    triangle_centroid, Point3, PointInClosedSurface, RayTriangleHit, Tetrahedron3, Triangle3,
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
pub use size::field::{
    AnisotropicSizingSample, MeshSizingField, PointSizingQuery, SegmentSizingQuery,
    SizingFieldService, SizingQueryResult, SizingQuerySource, SizingSample,
    SizingSampleApplication, SizingSampleRejection,
};
pub use source_topology::{
    extract_source_topology, SourceTopologyEdge, SourceTopologyError, SourceTopologyFace,
    SourceTopologyModel, SourceTopologyVertex,
};
pub use spatial_index::{Aabb3, LinearSpatialIndex, SpatialEntry, UniformGridSpatialIndex};
pub use surface::recovery::{
    validate_surface_recovery, SurfaceRecoveryError, SurfaceRecoveryOptions, SurfaceRecoveryReport,
};
pub use surface::validate::{
    validate_surface_discretization, SurfaceValidationError, SurfaceValidationOptions,
    SurfaceValidationReport,
};
pub use surface::{
    discretize_cad_surfaces, discretize_cad_surfaces_with_curves,
    discretize_cad_topology_surfaces_with_cad_curves, discretize_topology_surfaces,
    SurfaceCadCurveBoundaryProvenanceReport, SurfaceDiscretization, SurfaceDiscretizationError,
    SurfaceDiscretizationOptions, SurfaceElement, SurfaceNode, INTERNAL_SOURCE_EDGE_ID,
};
pub use tolerance::MeshingTolerance;
pub use validation::{
    analysis_mesh_validation_error_code, validate_analysis_mesh,
    validate_analysis_mesh_with_options, volume_component_count, AnalysisMeshValidationError,
    AnalysisMeshValidationOptions,
};
