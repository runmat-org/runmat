use crate::shared::{SharedCurveError, SharedCurveErrorKind};
use runmat_geometry_core::{ExactEdge, GeometryEvaluationError, ParameterRange};

use super::types::SharedCurveDiscretizationOptions;

pub(crate) fn require_parameter_range(
    edge: &ExactEdge,
    range: ParameterRange,
) -> Result<(), SharedCurveError> {
    if range.start.is_finite() && range.end.is_finite() && range.start < range.end {
        Ok(())
    } else {
        Err(edge_error(
            edge,
            SharedCurveErrorKind::GeometricMismatch,
            "curve parameter range",
            "exact evaluator range must be finite and increasing",
        ))
    }
}

pub(crate) fn validate_options(
    options: SharedCurveDiscretizationOptions,
) -> Result<(), SharedCurveError> {
    let resolution = options.resolution;
    if !resolution.maximum_chordal_deviation_m.is_finite()
        || resolution.maximum_chordal_deviation_m <= 0.0
        || !resolution.maximum_tangent_change_rad.is_finite()
        || resolution.maximum_tangent_change_rad <= 0.0
        || !resolution.minimum_metric_edge_length.is_finite()
        || resolution.minimum_metric_edge_length <= 0.0
        || !resolution.maximum_metric_edge_length.is_finite()
        || resolution.minimum_metric_edge_length > resolution.maximum_metric_edge_length
        || options.maximum_nodes_per_edge < 2
        || options.maximum_subdivision_depth == 0
        || !options.geometry_absolute_error_m.is_finite()
        || options.geometry_absolute_error_m <= 0.0
        || !options.pcurve_absolute_error.is_finite()
        || options.pcurve_absolute_error <= 0.0
        || !options.arc_length_absolute_error_m.is_finite()
        || options.arc_length_absolute_error_m <= 0.0
    {
        return Err(SharedCurveError::invalid_request(
            "shared curve discretization options",
            "resolution, node, depth, geometry, pcurve, and arc-length bounds must be finite, positive, and ordered",
        ));
    }
    Ok(())
}

pub(crate) fn geometry_error(edge: &ExactEdge, error: GeometryEvaluationError) -> SharedCurveError {
    edge_error(
        edge,
        SharedCurveErrorKind::GeometryEvaluation(error.kind),
        "exact evaluator",
        error.reason,
    )
}

pub(crate) fn edge_error(
    edge: &ExactEdge,
    kind: SharedCurveErrorKind,
    field: impl Into<String>,
    reason: impl Into<String>,
) -> SharedCurveError {
    SharedCurveError {
        edge_id: Some(edge.id.clone()),
        kind,
        field: field.into(),
        reason: reason.into(),
    }
}
