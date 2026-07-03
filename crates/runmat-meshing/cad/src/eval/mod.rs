pub mod frames;
mod samples;
pub mod types;

pub use frames::*;
pub use types::*;

pub const MODULE_PURPOSE: &str =
    "CAD projection, parameters, normals, derivatives, curvature, and evaluator capabilities";
