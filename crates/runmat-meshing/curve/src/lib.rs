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
    curve_partition_descriptors, decode_shared_curve_batch, decode_shared_curve_mesh,
    derive_curve_geometry_metric, discretize_shared_curve_partition, discretize_shared_curves,
    encode_shared_curve_batch, encode_shared_curve_mesh, join_shared_curve_batches,
    shared_curve_node_id, shared_degenerate_curve_node_id, validate_shared_curve_geometry,
    CurveMetricEvaluation, CurveMetricField, CurveMetricQuery, CurveMetricResolutionEvidence,
    CurveResolutionEvidence, CurveResolutionPolicy, ResolvedCurveMetricField, SharedCurve,
    SharedCurveBatch, SharedCurveDiscretizationOptions, SharedCurveError, SharedCurveErrorKind,
    SharedCurveFaceUse, SharedCurveGeometryValidationReport, SharedCurveMesh, SharedCurveNode,
    UniformCurveMetric, SHARED_CURVE_BATCH_SCHEMA_VERSION, SHARED_CURVE_MESH_SCHEMA_VERSION,
};
pub use validate::{
    validate_curve_discretization, CurveValidationError, CurveValidationOptions,
    CurveValidationReport,
};
