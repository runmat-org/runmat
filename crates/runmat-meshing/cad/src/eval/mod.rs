pub mod frames;
pub mod projection;
mod samples;
pub mod types;

pub use frames::*;
pub use projection::*;
pub use types::*;

pub const MODULE_PURPOSE: &str =
    "CAD projection, parameters, normals, derivatives, curvature, and evaluator capabilities";
