mod codec;
mod discretize;
mod error;
mod geometric_validation;
mod identity;
mod resolved_metric;
mod types;
mod validation;

pub use codec::{decode_shared_curve_mesh, encode_shared_curve_mesh};
pub use discretize::{
    discretize_shared_curves, CurveMetricEvaluation, CurveMetricField, CurveMetricQuery,
    SharedCurveDiscretizationOptions, UniformCurveMetric,
};
pub use error::{SharedCurveError, SharedCurveErrorKind};
pub use geometric_validation::{
    validate_shared_curve_geometry, SharedCurveGeometryValidationReport,
};
pub use identity::{shared_curve_node_id, shared_degenerate_curve_node_id};
pub use resolved_metric::ResolvedCurveMetricField;
pub use types::{
    CurveMetricResolutionEvidence, CurveResolutionEvidence, CurveResolutionPolicy, SharedCurve,
    SharedCurveFaceUse, SharedCurveMesh, SharedCurveNode, SHARED_CURVE_MESH_SCHEMA_VERSION,
};

#[cfg(test)]
mod tests;
