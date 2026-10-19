//! Constrained Delaunay tetrahedralization and solver-topology projection.

/// Actual package build configuration used for internal artifact reuse.
#[doc(hidden)]
pub const BUILD_CONFIGURATION: &str = env!("RUNMAT_BUILD_CONFIGURATION");

pub const CRATE_PURPOSE: &str = "constrained Delaunay tetrahedralization from exact PLCs";

pub mod cavity;
pub mod cdt;
pub mod reconnect;
