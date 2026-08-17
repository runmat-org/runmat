use runmat_geometry_core::{ExactEdge, GeometryEvaluationError, ParameterRange};

use super::types::{
    SharedCurveDiscretizationError, SharedCurveDiscretizationErrorKind,
    SharedCurveDiscretizationOptions,
};

pub(super) fn require_parameter_range(
    edge: &ExactEdge,
    range: ParameterRange,
) -> Result<(), SharedCurveDiscretizationError> {
    if range.start.is_finite() && range.end.is_finite() && range.start < range.end {
        Ok(())
    } else {
        Err(edge_error(
            edge,
            SharedCurveDiscretizationErrorKind::InvalidResult,
            "curve parameter range",
            "exact evaluator range must be finite and increasing",
        ))
    }
}

pub(super) fn validate_options(
    options: SharedCurveDiscretizationOptions,
) -> Result<(), SharedCurveDiscretizationError> {
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
        || !options.arc_length_absolute_error_m.is_finite()
        || options.arc_length_absolute_error_m <= 0.0
    {
        return Err(SharedCurveDiscretizationError::invalid(
            "shared curve discretization options",
            "resolution, node, depth, and arc-length bounds must be finite, positive, and ordered",
        ));
    }
    Ok(())
}

pub(super) fn geometry_error(
    edge: &ExactEdge,
    error: GeometryEvaluationError,
) -> SharedCurveDiscretizationError {
    edge_error(
        edge,
        SharedCurveDiscretizationErrorKind::GeometryEvaluation(error.kind),
        "exact evaluator",
        error.reason,
    )
}

pub(super) fn edge_error(
    edge: &ExactEdge,
    kind: SharedCurveDiscretizationErrorKind,
    field: impl Into<String>,
    reason: impl Into<String>,
) -> SharedCurveDiscretizationError {
    SharedCurveDiscretizationError {
        edge_id: Some(edge.id.clone()),
        kind,
        field: field.into(),
        reason: reason.into(),
    }
}
