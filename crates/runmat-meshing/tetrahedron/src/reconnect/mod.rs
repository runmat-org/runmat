mod boundary_links;
pub(crate) use boundary_links::BoundaryEdgeStarCandidates;
pub mod local_flip;
pub use local_flip::*;

pub const MODULE_PURPOSE: &str = "local flips and reconnections with rollback gates";
