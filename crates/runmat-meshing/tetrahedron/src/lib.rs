//! Tetrahedron4 generation, cavity, recovery, and reconnect stages.

pub const CRATE_PURPOSE: &str =
    "Tetrahedron4 generation from validated PLCs and constraint recovery";

pub mod cavity;
pub mod generate;
mod protected_edges;
pub mod reconnect;
pub mod recover;
pub mod structured_grid;
