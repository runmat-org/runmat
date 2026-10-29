//! Surface parameter triangulation, recovery, and validation stages.

/// Actual package build configuration used for internal artifact reuse.
#[doc(hidden)]
pub const BUILD_CONFIGURATION: &str = env!("RUNMAT_BUILD_CONFIGURATION");

pub const CRATE_PURPOSE: &str = "face-domain triangulation, loop recovery, and surface validation";

mod acceptance;
mod chart;
mod exact_metric;
mod face_geometry;
mod face_mesh;
mod face_pipeline;
mod refinement;
mod solver_topology;
mod surface_mesh;

#[cfg(test)]
mod face_pipeline_tests;

pub mod exact_boundary;
pub mod exact_cdt;

pub use acceptance::{
    accept_exact_face_chart_mesh, accept_exact_face_mesh, validate_exact_face_acceptance,
    validate_exact_face_chart_acceptance, ExactFaceAcceptanceError, ExactFaceAcceptanceErrorKind,
    ExactFaceAcceptanceOptions, ExactFaceAcceptanceReport, ExactFaceChartAcceptanceReport,
    ExactFaceTriangleAcceptance,
};
pub use chart::{
    build_exact_face_charts, recover_exact_face_chart_domains, triangulate_exact_face_charts,
    validate_exact_face_chart_delaunay, validate_exact_face_chart_domains,
    validate_exact_face_charts, ExactFaceChart, ExactFaceChartConstrainedDomain,
    ExactFaceChartDelaunay, ExactFaceChartDelaunayContext, ExactFaceChartError,
    ExactFaceChartErrorKind, ExactFaceChartOptions, ExactFaceChartParameterization,
    ExactFaceCharts,
};
pub use exact_boundary::{
    build_exact_surface_boundary, build_exact_surface_boundary_with_cancellation,
    validate_exact_surface_boundary, validate_exact_surface_boundary_with_cancellation,
    ExactFaceBoundary, ExactFaceBoundaryLoop, ExactFaceBoundarySegment, ExactSurfaceBoundary,
    ExactSurfaceBoundaryConflict, ExactSurfaceBoundaryError, ExactSurfaceBoundaryErrorKind,
    EXACT_SURFACE_BOUNDARY_SCHEMA_VERSION,
};
pub use exact_cdt::{
    build_exact_face_pslg, carve_exact_face_domain, exact_face_chart_cut_node_id,
    exact_face_interior_node_id, recover_exact_face_segments, triangulate_exact_face_pslg,
    validate_exact_face_constrained_delaunay, validate_exact_face_delaunay,
    validate_exact_face_pslg, validate_exact_face_trimmed_delaunay, ExactFaceConstrainedDelaunay,
    ExactFaceDelaunay, ExactFaceDelaunayError, ExactFaceDelaunayErrorKind,
    ExactFaceDelaunayOptions, ExactFaceDelaunayTriangle, ExactFacePslg, ExactFacePslgError,
    ExactFacePslgErrorKind, ExactFacePslgLoop, ExactFacePslgLoopSource, ExactFacePslgSegment,
    ExactFacePslgSegmentSource, ExactFacePslgVertex, ExactFaceRecoveredSegment,
    ExactFaceTrimmedDelaunay,
};
pub use exact_metric::{
    validate_exact_face_metric_evaluation,
    validate_exact_face_metric_evaluation_in_parameterization, ExactFaceMetricError,
    ExactFaceMetricErrorKind, ExactFaceMetricEvaluation, ExactFaceMetricProvenance,
    ParametricMetricTensor, ResolvedFaceMetricField,
};
pub use face_geometry::{
    evaluate_exact_face_geometry, evaluate_exact_face_geometry_in_parameterization,
    validate_exact_face_geometry, validate_exact_face_geometry_in_parameterization,
    ExactBoundaryNodeCoordinates, ExactFaceGeometry, ExactFaceGeometryContext,
    ExactFaceGeometryError, ExactFaceGeometryErrorKind, ExactFaceGeometryVertex,
    ExactFaceTriangleGeometry,
};
pub use face_mesh::{
    join_exact_face_charts, validate_exact_face_mesh, ExactFaceJoinContext, ExactFaceJoinError,
    ExactFaceJoinErrorKind, ExactFaceJoinOptions, ExactFaceMesh, ExactFaceMeshBoundarySegment,
    ExactFaceMeshEdgeParameter, ExactFaceMeshJoinedCut, ExactFaceMeshNode, ExactFaceMeshNodeUse,
    ExactFaceMeshTriangle, EXACT_FACE_MESH_SCHEMA_VERSION,
};
pub use face_pipeline::{
    mesh_exact_face_partition, mesh_exact_face_partition_with_reuse, visit_exact_face_partition,
    visit_exact_face_partition_with_reuse, AcceptedExactFace, ExactFaceMeshReuse,
    ExactFacePartitionCompletion, ExactFacePartitionContext, ExactFacePartitionError,
    ExactFacePartitionErrorKind, ExactFacePartitionOptions,
};
pub use refinement::{
    classify_exact_face_refinement_candidate, derive_exact_face_feature_collars,
    insert_exact_face_refinement_candidate, refine_exact_face_chart_until_blocked,
    refine_exact_face_until_blocked, select_exact_face_refinement_candidate,
    split_exact_face_chart_cut, validate_exact_face_chart_cut_split_result,
    validate_exact_face_feature_collars, ExactChartCutSplit, ExactChartCutSplitImage,
    ExactFaceCandidateDisposition, ExactFaceChartRefinedMesh, ExactFaceChartRefinementOptions,
    ExactFaceChartRefinementOutcome, ExactFaceFeatureCollar, ExactFaceFeatureCollars,
    ExactFaceRefinedMesh, ExactFaceRefinedTopology, ExactFaceRefinementCandidate,
    ExactFaceRefinementContext, ExactFaceRefinementError, ExactFaceRefinementErrorKind,
    ExactFaceRefinementOptions, ExactFaceRefinementOutcome, ExactFaceRefinementPolicy,
    ExactFaceRefinementReason, ExactProtectedSegmentSplit,
};
pub use solver_topology::build_surface_solver_topology;
pub use surface_mesh::{
    build_exact_face_mesh_batch, build_exact_face_partition_result, decide_exact_surface_pass,
    decide_exact_surface_pass_owned, decide_exact_surface_pass_owned_admitted,
    decide_exact_surface_pass_with_cancellation, decode_exact_face_mesh_batch,
    decode_exact_face_partition_result, decode_exact_face_partition_result_admitted,
    decode_exact_face_partition_result_with_cancellation,
    decode_exact_face_reuse_record_with_cancellation, decode_exact_surface_mesh,
    decode_exact_surface_mesh_from_pass, decode_exact_surface_pass_result,
    decode_exact_surface_pass_result_with_cancellation, decode_published_exact_surface_mesh,
    decode_published_exact_surface_mesh_admitted,
    decode_published_exact_surface_mesh_with_cancellation,
    encode_decided_exact_surface_pass_result,
    encode_decided_exact_surface_pass_result_with_cancellation, encode_exact_face_mesh_batch,
    encode_exact_face_partition_result, encode_exact_face_partition_result_with_cancellation,
    encode_exact_surface_mesh, encode_exact_surface_mesh_from_pass,
    encode_exact_surface_mesh_from_pass_with_cancellation, encode_exact_surface_pass_result,
    face_partition_descriptors, join_exact_face_mesh_batches,
    profile_exact_face_partition_result_encoding, resolve_exact_surface_pass,
    validate_exact_face_mesh_batch, validate_exact_face_partition_result,
    validate_exact_face_partition_result_with_cancellation, validate_exact_surface_mesh,
    validate_exact_surface_pass_result, validate_published_exact_surface_mesh,
    AcceptedExactSurfacePass, ExactFaceMeshBatch, ExactFacePartitionOutcome,
    ExactFacePartitionResult, ExactSurfaceConvergenceError, ExactSurfaceConvergenceErrorKind,
    ExactSurfaceConvergenceOutcome, ExactSurfaceJoinOptions, ExactSurfaceMesh,
    ExactSurfaceMeshError, ExactSurfaceMeshErrorKind, ExactSurfacePassOutcome,
    ExactSurfacePassResult, ExactSurfaceShellEvidence, PreparedExactSurfacePassEncoding,
    EXACT_FACE_MESH_BATCH_SCHEMA_VERSION, EXACT_FACE_PARTITION_RESULT_SCHEMA_VERSION,
    EXACT_SURFACE_MESH_SCHEMA_VERSION, EXACT_SURFACE_PASS_RESULT_SCHEMA_VERSION,
    MAX_EXACT_FACE_PARTITIONS,
};
pub use surface_mesh::{
    decode_exact_face_partition_records, decode_exact_face_partition_records_admitted,
    decode_exact_face_partition_records_with_cancellation, encode_exact_face_partition_completion,
    profile_exact_face_partition_completion, EXACT_FACE_PARTITION_RECORD_SET_SCHEMA_VERSION,
};

pub use surface_mesh::ExactFacePartitionEncoding;
