//! Surface parameter triangulation, recovery, and validation stages.

pub const CRATE_PURPOSE: &str = "face-domain triangulation, loop recovery, and surface validation";

mod acceptance;
mod chart;
mod exact_metric;
mod face_geometry;
mod math;
mod refinement;

pub mod contract;
pub mod exact_boundary;
pub mod exact_cdt;
pub mod param_tri;
pub mod recovery;
pub mod validate;

pub use acceptance::{
    accept_exact_face_mesh, validate_exact_face_acceptance, ExactFaceAcceptanceError,
    ExactFaceAcceptanceErrorKind, ExactFaceAcceptanceOptions, ExactFaceAcceptanceReport,
    ExactFaceTriangleAcceptance,
};
pub use chart::{
    build_exact_face_charts, recover_exact_face_chart_domains, triangulate_exact_face_charts,
    validate_exact_face_chart_delaunay, validate_exact_face_chart_domains,
    validate_exact_face_charts, ExactFaceChart, ExactFaceChartConstrainedDomain,
    ExactFaceChartDelaunay, ExactFaceChartDelaunayContext, ExactFaceChartError,
    ExactFaceChartErrorKind, ExactFaceChartOptions, ExactFaceCharts,
};
pub use contract::build_surface_mesh_contract;
pub use exact_boundary::{
    build_exact_surface_boundary, validate_exact_surface_boundary, ExactFaceBoundary,
    ExactFaceBoundaryLoop, ExactFaceBoundarySegment, ExactSurfaceBoundary,
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
    validate_exact_face_metric_evaluation, ExactFaceMetricError, ExactFaceMetricErrorKind,
    ExactFaceMetricEvaluation, ParametricMetricTensor, ResolvedFaceMetricField,
};
pub use face_geometry::{
    evaluate_exact_face_geometry, validate_exact_face_geometry, ExactFaceGeometry,
    ExactFaceGeometryError, ExactFaceGeometryErrorKind, ExactFaceGeometryVertex,
    ExactFaceTriangleGeometry,
};
pub use param_tri::{
    discretize_cad_surfaces, discretize_cad_surfaces_with_curves,
    discretize_cad_topology_surfaces_with_cad_curves,
    discretize_cad_topology_surfaces_with_cad_curves_and_sizing,
    discretize_cad_topology_surfaces_with_curves, discretize_topology_surfaces,
    SurfaceCadCurveBoundaryEdgeProvenance, SurfaceCadCurveBoundaryProvenanceReport,
    SurfaceDiscretization, SurfaceDiscretizationError, SurfaceDiscretizationOptions,
    SurfaceElement, SurfaceLoopCoverageReport, SurfaceNode, INTERNAL_SOURCE_EDGE_ID,
};
pub use recovery::{
    validate_surface_recovery, SurfaceRecoveryError, SurfaceRecoveryOptions, SurfaceRecoveryReport,
};
pub use refinement::{
    classify_exact_face_refinement_candidate, derive_exact_face_feature_collars,
    insert_exact_face_refinement_candidate, refine_exact_face_until_blocked,
    select_exact_face_refinement_candidate, split_exact_face_chart_cut,
    validate_exact_face_chart_cut_split_result, validate_exact_face_feature_collars,
    ExactChartCutSplit, ExactChartCutSplitImage, ExactFaceCandidateDisposition,
    ExactFaceFeatureCollar, ExactFaceFeatureCollars, ExactFaceRefinedMesh,
    ExactFaceRefinedTopology, ExactFaceRefinementCandidate, ExactFaceRefinementContext,
    ExactFaceRefinementError, ExactFaceRefinementErrorKind, ExactFaceRefinementOptions,
    ExactFaceRefinementOutcome, ExactFaceRefinementPolicy, ExactFaceRefinementReason,
    ExactProtectedSegmentSplit,
};
pub use validate::{
    validate_cad_topology_surface_discretization, validate_surface_discretization,
    SurfaceValidationError, SurfaceValidationOptions, SurfaceValidationReport,
};
