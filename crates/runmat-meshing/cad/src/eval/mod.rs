pub mod frames;
pub mod projection;
pub mod report;
mod samples;
pub mod types;

pub use frames::*;
pub use projection::*;
pub use report::*;
pub use types::*;

pub const MODULE_PURPOSE: &str =
    "CAD projection, parameters, normals, derivatives, curvature, and evaluator capabilities";
