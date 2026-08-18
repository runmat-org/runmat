mod batch;
mod batch_codec;
mod codec;
mod discretize;
mod error;
mod geometric_validation;
mod geometry_metric;
mod identity;
mod resolved_metric;
mod types;
mod validation;

pub use batch::{
    curve_partition_descriptors, discretize_shared_curve_partition, join_shared_curve_batches,
};
pub use batch_codec::{decode_shared_curve_batch, encode_shared_curve_batch};
pub use codec::{decode_shared_curve_mesh, encode_shared_curve_mesh};
pub use discretize::{
    discretize_shared_curves, CurveMetricEvaluation, CurveMetricField, CurveMetricQuery,
    SharedCurveDiscretizationOptions, UniformCurveMetric,
};
pub use error::{SharedCurveError, SharedCurveErrorKind};
pub use geometric_validation::{
    validate_shared_curve_geometry, SharedCurveGeometryValidationReport,
};
pub use geometry_metric::derive_curve_geometry_metric;
pub use identity::{shared_curve_interior_node_id, shared_curve_vertex_node_id};
pub use resolved_metric::ResolvedCurveMetricField;
pub use types::{
    CurveMetricResolutionEvidence, CurveResolutionEvidence, CurveResolutionPolicy, SharedCurve,
    SharedCurveBatch, SharedCurveFaceUse, SharedCurveMesh, SharedCurveNode,
    SHARED_CURVE_BATCH_SCHEMA_VERSION, SHARED_CURVE_MESH_SCHEMA_VERSION,
};

#[cfg(test)]
mod tests;
