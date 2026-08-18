//! Surface parameter triangulation, recovery, and validation stages.

pub const CRATE_PURPOSE: &str = "face-domain triangulation, loop recovery, and surface validation";

mod math;

pub mod contract;
pub mod exact_boundary;
pub mod exact_cdt;
pub mod param_tri;
pub mod recovery;
pub mod validate;

pub use contract::build_surface_mesh_contract;
pub use exact_boundary::{
    build_exact_surface_boundary, validate_exact_surface_boundary, ExactFaceBoundary,
    ExactFaceBoundaryLoop, ExactFaceBoundarySegment, ExactSurfaceBoundary,
    ExactSurfaceBoundaryConflict, ExactSurfaceBoundaryError, ExactSurfaceBoundaryErrorKind,
    EXACT_SURFACE_BOUNDARY_SCHEMA_VERSION,
};
pub use exact_cdt::{
    build_exact_face_pslg, recover_exact_face_segments, triangulate_exact_face_pslg,
    validate_exact_face_constrained_delaunay, validate_exact_face_delaunay,
    validate_exact_face_pslg, ExactFaceConstrainedDelaunay, ExactFaceDelaunay,
    ExactFaceDelaunayError, ExactFaceDelaunayErrorKind, ExactFaceDelaunayOptions,
    ExactFaceDelaunayTriangle, ExactFacePslg, ExactFacePslgError, ExactFacePslgErrorKind,
    ExactFacePslgLoop, ExactFacePslgSegment, ExactFacePslgVertex, ExactFaceRecoveredSegment,
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
pub use validate::{
    validate_cad_topology_surface_discretization, validate_surface_discretization,
    SurfaceValidationError, SurfaceValidationOptions, SurfaceValidationReport,
};
