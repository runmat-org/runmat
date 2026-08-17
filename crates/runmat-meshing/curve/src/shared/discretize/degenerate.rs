use runmat_geometry_core::{
    ExactBRepTopology, ExactCurveEvaluator, ExactEdge, ExactPcurveEvaluator,
    GeometryEvaluationControl,
};

use crate::shared::{
    shared_degenerate_curve_node_id, CurveMetricResolutionEvidence, CurveResolutionEvidence,
    SharedCurve, SharedCurveError, SharedCurveErrorKind, SharedCurveNode,
};

use super::{
    error::{edge_error, geometry_error, require_parameter_range},
    math::{norm, sub},
    pcurves::face_uses_for_parameters,
    sampling::interval_parameters,
    types::SharedCurveDiscretizationOptions,
};

pub(super) fn discretize_degenerate_edge(
    topology: &ExactBRepTopology,
    edge: &ExactEdge,
    curves: &dyn ExactCurveEvaluator,
    pcurves: &dyn ExactPcurveEvaluator,
    control: &dyn GeometryEvaluationControl,
    options: SharedCurveDiscretizationOptions,
) -> Result<SharedCurve, SharedCurveError> {
    let parameter_range = curves
        .parameter_range(&edge.curve_evaluator_id)
        .map_err(|error| geometry_error(edge, error))?;
    require_parameter_range(edge, parameter_range)?;
    let transform = topology.world_transform_for(&edge.id).map_err(|error| {
        edge_error(
            edge,
            SharedCurveErrorKind::GeometricMismatch,
            "edge occurrence transform",
            error.to_string(),
        )
    })?;
    let witness_parameters = interval_parameters(parameter_range.start, parameter_range.end);
    let anchor = curves
        .point(&edge.curve_evaluator_id, parameter_range.start, control)
        .map(|point| transform.transform_point(point))
        .map_err(|error| geometry_error(edge, error))?;
    for parameter in witness_parameters {
        control
            .checkpoint()
            .map_err(|error| geometry_error(edge, error))?;
        let point = curves
            .point(&edge.curve_evaluator_id, parameter, control)
            .map(|point| transform.transform_point(point))
            .map_err(|error| geometry_error(edge, error))?;
        if norm(sub(point, anchor)) > options.geometry_absolute_error_m {
            return Err(edge_error(
                edge,
                SharedCurveErrorKind::GeometricMismatch,
                "degenerate exact edge",
                "declared degenerate curve does not collapse to one 3D point",
            ));
        }
    }
    let parameters = [parameter_range.start, parameter_range.end];
    let node_id = shared_degenerate_curve_node_id(&edge.id);
    let nodes = parameters
        .into_iter()
        .map(|parameter| SharedCurveNode {
            node_id,
            source_vertex_id: edge.start_vertex_id.clone(),
            parameter,
            arc_length_m: 0.0,
            coordinates_m: anchor,
        })
        .collect();
    let face_uses = face_uses_for_parameters(
        topology,
        edge,
        pcurves,
        control,
        parameter_range,
        &parameters,
    )?;
    Ok(SharedCurve {
        source_edge_id: edge.id.clone(),
        parameter_range,
        nodes,
        face_uses,
        requested: options.resolution,
        achieved: CurveResolutionEvidence {
            maximum_chordal_deviation_m: 0.0,
            maximum_tangent_change_rad: 0.0,
            minimum_metric_edge_length: 0.0,
            maximum_metric_edge_length: 0.0,
        },
        metric_resolution: CurveMetricResolutionEvidence::DegenerateTopologicalCollapse,
    })
}
