mod degenerate;
mod sampler;

use std::collections::BTreeMap;

use runmat_geometry_core::{
    ExactBRepTopology, ExactCurveEvaluator, ExactEdge, ExactPcurveEvaluator,
    GeometryEvaluationControl, GeometryTransform, ParameterRange,
};

use super::{
    discretize::{
        edge_error, geometry_error, sub, validate_options, world_arc_length, CurveMetricField,
        SharedCurveDiscretizationOptions, SharedCurveEvaluationContext,
    },
    SharedCurve, SharedCurveError, SharedCurveErrorKind, SharedCurveMesh,
};
use degenerate::validate_degenerate_geometry;
use sampler::{validate_interval, ValidationSampler};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedCurveGeometryValidationReport {
    pub edge_count: u64,
    pub node_count: u64,
    pub metric_evaluation_count: u64,
}

pub fn validate_shared_curve_geometry(
    mesh: &SharedCurveMesh,
    topology: &ExactBRepTopology,
    curves: &dyn ExactCurveEvaluator,
    pcurves: &dyn ExactPcurveEvaluator,
    metric_field: &dyn CurveMetricField,
    control: &dyn GeometryEvaluationControl,
    options: SharedCurveDiscretizationOptions,
) -> Result<SharedCurveGeometryValidationReport, SharedCurveError> {
    validate_options(options)?;
    let context =
        SharedCurveEvaluationContext::new(topology, curves, pcurves, metric_field, control);
    mesh.validate_against(context.topology)?;
    let edge_by_id = context
        .topology
        .edges
        .iter()
        .map(|edge| (&edge.id, edge))
        .collect::<BTreeMap<_, _>>();
    let mut node_count = 0u64;
    let mut metric_evaluation_count = 0u64;
    for curve in &mesh.edges {
        let edge = edge_by_id[&curve.source_edge_id];
        metric_evaluation_count =
            metric_evaluation_count.saturating_add(validate_curve(curve, edge, context, options)?);
        node_count = node_count.saturating_add(curve.nodes.len() as u64);
    }
    Ok(SharedCurveGeometryValidationReport {
        edge_count: mesh.edges.len() as u64,
        node_count,
        metric_evaluation_count,
    })
}

fn validate_curve(
    curve: &SharedCurve,
    edge: &ExactEdge,
    context: SharedCurveEvaluationContext<'_>,
    options: SharedCurveDiscretizationOptions,
) -> Result<u64, SharedCurveError> {
    if curve.requested != options.resolution {
        return Err(mismatch(
            edge,
            "curve validation request",
            "artifact resolution policy differs from the independent validator request",
        ));
    }
    let exact_range = context
        .curves
        .parameter_range(&edge.curve_evaluator_id)
        .map_err(|error| geometry_error(edge, error))?;
    if exact_range != curve.parameter_range {
        return Err(mismatch(
            edge,
            "curve parameter range",
            "artifact range differs from the exact evaluator",
        ));
    }
    let transform = context
        .topology
        .world_transform_for(&edge.id)
        .map_err(|error| mismatch(edge, "edge occurrence transform", error.to_string()))?;
    validate_nodes(
        curve,
        edge,
        context.curves,
        context.control,
        transform,
        options,
    )?;
    validate_pcurves(
        curve,
        edge,
        context.topology,
        context.pcurves,
        context.control,
        options,
    )?;
    if edge.is_degenerate {
        validate_degenerate_geometry(
            curve,
            edge,
            context.curves,
            context.control,
            transform,
            options.geometry_absolute_error_m,
        )?;
        return Ok(0);
    }

    let mut sampler = ValidationSampler::new(
        edge,
        context.curves,
        context.metric_field,
        context.control,
        transform,
    );
    let mut maximum_chordal_deviation_m: f64 = 0.0;
    let mut maximum_tangent_change_rad: f64 = 0.0;
    let mut minimum_metric_edge_length = f64::INFINITY;
    let mut maximum_metric_edge_length: f64 = 0.0;
    for nodes in curve.nodes.windows(2) {
        let evidence = validate_interval(&mut sampler, nodes[0].parameter, nodes[1].parameter)?;
        maximum_chordal_deviation_m = maximum_chordal_deviation_m.max(evidence.0);
        maximum_tangent_change_rad = maximum_tangent_change_rad.max(evidence.1);
        minimum_metric_edge_length = minimum_metric_edge_length.min(evidence.2);
        maximum_metric_edge_length = maximum_metric_edge_length.max(evidence.2);
    }
    let requested = curve.requested;
    if maximum_chordal_deviation_m
        > requested.maximum_chordal_deviation_m + options.geometry_absolute_error_m
        || maximum_tangent_change_rad > requested.maximum_tangent_change_rad + 1.0e-12
        || minimum_metric_edge_length + 1.0e-12 < requested.minimum_metric_edge_length
        || maximum_metric_edge_length > requested.maximum_metric_edge_length + 1.0e-12
        || curve.achieved.maximum_chordal_deviation_m + options.geometry_absolute_error_m
            < maximum_chordal_deviation_m
        || curve.achieved.maximum_tangent_change_rad + 1.0e-12 < maximum_tangent_change_rad
        || curve.achieved.minimum_metric_edge_length > minimum_metric_edge_length + 1.0e-12
        || curve.achieved.maximum_metric_edge_length + 1.0e-12 < maximum_metric_edge_length
    {
        return Err(mismatch(
            edge,
            "curve achieved resolution",
            "independent exact sampling violates or exceeds the recorded resolution evidence",
        ));
    }
    sampler.validate_metric_evidence(curve)?;
    Ok(sampler.sample_count())
}

fn validate_nodes(
    curve: &SharedCurve,
    edge: &ExactEdge,
    curves: &dyn ExactCurveEvaluator,
    control: &dyn GeometryEvaluationControl,
    transform: GeometryTransform,
    options: SharedCurveDiscretizationOptions,
) -> Result<(), SharedCurveError> {
    let segment_error = options.arc_length_absolute_error_m / (curve.nodes.len() - 1) as f64;
    let mut arc_length_m = 0.0;
    for (index, node) in curve.nodes.iter().enumerate() {
        let point_m = curves
            .point(&edge.curve_evaluator_id, node.parameter, control)
            .map(|point| transform.transform_point(point))
            .map_err(|error| geometry_error(edge, error))?;
        if length(sub(point_m, node.coordinates_m)) > options.geometry_absolute_error_m {
            return Err(mismatch(
                edge,
                "curve node coordinates",
                "stored coordinates differ from exact evaluation",
            ));
        }
        if index > 0 {
            arc_length_m += world_arc_length(
                edge,
                curves,
                control,
                transform,
                ParameterRange {
                    start: curve.nodes[index - 1].parameter,
                    end: node.parameter,
                },
                segment_error,
                options.maximum_subdivision_depth,
            )?;
        }
        if (arc_length_m - node.arc_length_m).abs() > options.arc_length_absolute_error_m {
            return Err(mismatch(
                edge,
                "curve node arc length",
                "stored cumulative arc length differs from exact integration",
            ));
        }
    }
    Ok(())
}

fn validate_pcurves(
    curve: &SharedCurve,
    edge: &ExactEdge,
    topology: &ExactBRepTopology,
    pcurves: &dyn ExactPcurveEvaluator,
    control: &dyn GeometryEvaluationControl,
    options: SharedCurveDiscretizationOptions,
) -> Result<(), SharedCurveError> {
    let coedge_by_id = topology
        .coedges
        .iter()
        .map(|coedge| (&coedge.id, coedge))
        .collect::<BTreeMap<_, _>>();
    for face_use in &curve.face_uses {
        let coedge = coedge_by_id[&face_use.coedge_id];
        let range = pcurves
            .parameter_range(&coedge.pcurve_evaluator_id)
            .map_err(|error| geometry_error(edge, error))?;
        if range != curve.parameter_range {
            return Err(mismatch(
                edge,
                "pcurve parameter range",
                "pcurve and shared edge ranges differ",
            ));
        }
        for (node, stored_uv) in curve.nodes.iter().zip(&face_use.node_uv) {
            let exact_uv = pcurves
                .point(&coedge.pcurve_evaluator_id, node.parameter, control)
                .map_err(|error| geometry_error(edge, error))?;
            if ((exact_uv[0] - stored_uv[0]).powi(2) + (exact_uv[1] - stored_uv[1]).powi(2)).sqrt()
                > options.pcurve_absolute_error
            {
                return Err(mismatch(
                    edge,
                    "curve face-use UV",
                    "stored pcurve image differs from exact evaluation",
                ));
            }
        }
    }
    Ok(())
}

fn length(vector: [f64; 3]) -> f64 {
    (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt()
}

pub(super) fn mismatch(
    edge: &ExactEdge,
    field: impl Into<String>,
    reason: impl Into<String>,
) -> SharedCurveError {
    edge_error(edge, SharedCurveErrorKind::GeometricMismatch, field, reason)
}
