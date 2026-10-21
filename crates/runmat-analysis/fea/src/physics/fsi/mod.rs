//! Partitioned fluid-structure coupling.

mod diagnostics;
mod interface;
mod recovery;

pub use recovery::{couple_fluid_structure, FsiCouplingOptions, FsiCouplingResult};
