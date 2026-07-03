pub mod checks;
mod geometry;
mod source_edges;
#[cfg(test)]
mod tests;
mod types;

pub use checks::*;
pub use types::*;

pub const MODULE_PURPOSE: &str = "surface projection, normal, orientation, and provenance checks";
