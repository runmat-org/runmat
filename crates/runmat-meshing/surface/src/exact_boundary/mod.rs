mod build;
mod control;
mod crossings;
mod types;
mod validate;

pub use build::{build_exact_surface_boundary, build_exact_surface_boundary_with_cancellation};
pub use types::{
    ExactFaceBoundary, ExactFaceBoundaryLoop, ExactFaceBoundarySegment, ExactSurfaceBoundary,
    ExactSurfaceBoundaryConflict, ExactSurfaceBoundaryError, ExactSurfaceBoundaryErrorKind,
    EXACT_SURFACE_BOUNDARY_SCHEMA_VERSION,
};
pub use validate::{
    validate_exact_surface_boundary, validate_exact_surface_boundary_with_cancellation,
};

#[cfg(test)]
mod tests;
