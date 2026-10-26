use runmat_geometry_core::{
    ExactCurveEvaluator, ExactEdge, GeometryEvaluationControl, GeometryTransform,
};

use super::super::{
    discretize::{geometry_error, sub},
    SharedCurve, SharedCurveError,
};
use super::mismatch;

pub(super) fn validate_degenerate_geometry(
    curve: &SharedCurve,
    edge: &ExactEdge,
    curves: &dyn ExactCurveEvaluator,
    control: &dyn GeometryEvaluationControl,
    transform: GeometryTransform,
    absolute_error_m: f64,
) -> Result<(), SharedCurveError> {
    let anchor = curve.nodes[0].coordinates_m.canonical_value();
    for index in 0..=8 {
        control
            .checkpoint()
            .map_err(|error| geometry_error(edge, error))?;
        let parameter = curve
            .parameter_range
            .parameter_at_fraction(index as f64 / 8.0)
            .map_err(|error| geometry_error(edge, error))?;
        let point = curves
            .point(&edge.curve_evaluator_id, parameter, control)
            .map(|point| transform.transform_point(point.canonical_value()))
            .map_err(|error| geometry_error(edge, error))?;
        if length(sub(point, anchor)) > absolute_error_m {
            return Err(mismatch(
                edge,
                "degenerate exact edge",
                "independent exact samples do not collapse to the stored 3D node",
            ));
        }
    }
    Ok(())
}

fn length(vector: [f64; 3]) -> f64 {
    (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt()
}
