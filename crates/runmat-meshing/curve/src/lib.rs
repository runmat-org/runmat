//! Curve discretization and validation stages.

pub const CRATE_PURPOSE: &str = "CAD edge discretization and curve mesh validation";

pub mod discretize;
pub mod validate;

pub use discretize::{
    discretize_cad_topology_curves_with_sizing, discretize_topology_curves,
    discretize_topology_curves_with_sizing, CadCurveDiscretization, CadCurveEdgeProvenance,
    CurveDiscretization, CurveDiscretizationError, CurveDiscretizationOptions, CurveElement,
    CurveNode,
};
pub use validate::{
    validate_curve_discretization, CurveValidationError, CurveValidationOptions,
    CurveValidationReport,
};
