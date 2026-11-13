//! Curve discretization and validation stages.

/// Actual package build configuration used for internal artifact reuse.
#[doc(hidden)]
pub const BUILD_CONFIGURATION: &str = env!("RUNMAT_BUILD_CONFIGURATION");

pub const CRATE_PURPOSE: &str = "CAD edge discretization and curve mesh validation";

mod shared;
pub use shared::{
    apply_shared_curve_splits, canonicalize_shared_curve_splits, curve_partition_descriptors,
    decode_shared_curve_batch, decode_shared_curve_mesh,
    decode_shared_curve_mesh_with_cancellation, derive_curve_geometry_metric,
    discretize_shared_curve_partition, discretize_shared_curves, encode_shared_curve_batch,
    encode_shared_curve_batch_admitted, encode_shared_curve_mesh,
    encode_shared_curve_mesh_admitted, join_shared_curve_batches, shared_curve_interior_node_id,
    shared_curve_vertex_node_id, validate_shared_curve_geometry, validate_shared_curve_split_set,
    validate_shared_curve_split_set_with_cancellation, CurveMetricEvaluation, CurveMetricField,
    CurveMetricQuery, CurveMetricResolutionEvidence, CurveResolutionEvidence,
    CurveResolutionPolicy, ExactFacePcurveEvaluator, ResolvedCurveMetricField, SharedCurve,
    SharedCurveBatch, SharedCurveDiscretizationOptions, SharedCurveError, SharedCurveErrorKind,
    SharedCurveEvaluationContext, SharedCurveFaceUse, SharedCurveGeometryValidationReport,
    SharedCurveMesh, SharedCurveNode, SharedCurveSegmentSplit, UniformCurveMetric,
    SHARED_CURVE_BATCH_SCHEMA_VERSION, SHARED_CURVE_MESH_SCHEMA_VERSION,
};
