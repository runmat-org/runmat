//! Curve discretization and validation stages.

pub const CRATE_PURPOSE: &str = "CAD edge discretization and curve mesh validation";

pub mod discretize;
pub mod validate;

pub use discretize::{
    discretize_topology_curves, CurveDiscretization, CurveDiscretizationError,
    CurveDiscretizationOptions, CurveElement, CurveNode,
};
