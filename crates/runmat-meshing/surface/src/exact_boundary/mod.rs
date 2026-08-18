mod build;
mod types;
mod validate;

pub use build::build_exact_surface_boundary;
pub use types::{
    ExactFaceBoundary, ExactFaceBoundaryLoop, ExactFaceBoundarySegment, ExactSurfaceBoundary,
    ExactSurfaceBoundaryError, ExactSurfaceBoundaryErrorKind,
    EXACT_SURFACE_BOUNDARY_SCHEMA_VERSION,
};
pub use validate::validate_exact_surface_boundary;

#[cfg(test)]
mod tests;
