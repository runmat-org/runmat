//! Tetrahedron4 generation, cavity, recovery, reconnect, and optimization stages.

pub const CRATE_PURPOSE: &str =
    "Tetrahedron4 generation from validated PLCs and constraint recovery";

pub mod cavity;
pub mod cdt;
pub mod generate;
pub mod optimize;
mod protected_edges;
pub mod reconnect;
pub mod recover;
pub mod structured_grid;
