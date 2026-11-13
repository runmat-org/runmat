//! Independent analytic qualification data shared by core and scheduled native witnesses.
//! These frozen centroid boundary samples are validation fixtures, not CAD profile support
//! or spatial-accuracy certificates for arbitrary authored studies.
pub mod fixture;
pub mod mass_flow;
mod metrics;
pub mod reference;
pub use metrics::{cell_centroid_velocity_error, wall_shear_error};
