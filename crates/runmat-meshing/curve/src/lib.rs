//! Curve discretization and validation stages.

pub const CRATE_PURPOSE: &str = "CAD edge discretization and curve mesh validation";

pub mod contract;
pub mod discretize;
mod shared;
pub mod validate;

pub use contract::build_curve_mesh_contract;
pub use discretize::{
    discretize_cad_topology_curves_with_sizing,
    discretize_cad_topology_curves_with_sizing_and_provider, discretize_topology_curves,
    discretize_topology_curves_with_sizing, CadCurveDiscretization, CadCurveEdgeProvenance,
    CadCurveEvaluationRequest, CadCurveEvaluatorProvider, CurveDiscretization,
    CurveDiscretizationError, CurveDiscretizationOptions, CurveElement, CurveNode,
    NoopCadCurveEvaluatorProvider,
};
pub use shared::{
    decode_shared_curve_mesh, encode_shared_curve_mesh, shared_curve_node_id,
    CurveResolutionEvidence, CurveResolutionPolicy, SharedCurve, SharedCurveFaceUse,
    SharedCurveMesh, SharedCurveNode, SharedCurveValidationError, SHARED_CURVE_MESH_SCHEMA_VERSION,
};
pub use validate::{
    validate_curve_discretization, CurveValidationError, CurveValidationOptions,
    CurveValidationReport,
};
