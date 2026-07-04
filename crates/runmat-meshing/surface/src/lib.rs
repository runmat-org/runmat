//! Surface parameter triangulation, recovery, and validation stages.

pub const CRATE_PURPOSE: &str = "face-domain triangulation, loop recovery, and surface validation";

mod math;

pub mod param_tri;
pub mod recovery;
pub mod validate;

pub use param_tri::{
    discretize_cad_surfaces, discretize_cad_surfaces_with_curves,
    discretize_cad_topology_surfaces_with_curves, discretize_topology_surfaces,
    SurfaceDiscretization, SurfaceDiscretizationError, SurfaceDiscretizationOptions,
    SurfaceElement, SurfaceLoopCoverageReport, SurfaceNode, INTERNAL_SOURCE_EDGE_ID,
};
pub use recovery::{
    validate_surface_recovery, SurfaceRecoveryError, SurfaceRecoveryOptions, SurfaceRecoveryReport,
};
pub use validate::{
    validate_surface_discretization, SurfaceValidationError, SurfaceValidationOptions,
    SurfaceValidationReport,
};
