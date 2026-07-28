pub mod checks;
mod edges;
mod geometry;
#[cfg(test)]
mod tests;
mod types;

pub use checks::*;
pub use types::*;

pub const MODULE_PURPOSE: &str = "loop, hole, and source-edge recovery before PLC construction";
