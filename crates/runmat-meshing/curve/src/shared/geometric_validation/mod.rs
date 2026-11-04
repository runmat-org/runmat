mod degenerate;
mod sampler;

use std::collections::BTreeMap;

use runmat_geometry_core::{
    ExactBRepTopology, ExactCurveEvaluator, ExactEdge, ExactPcurveEvaluator, ExactSurfaceEvaluator,
    GeometryEvaluationControl, GeometryTransform, ParameterRange,
};

use super::{
    discretize::{
        edge_error, face_uses_for_parameters, geometry_error, sub, validate_options,
        world_arc_length, CurveMetricField, SharedCurveDiscretizationOptions,
        SharedCurveEvaluationContext,
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
    surfaces: &dyn ExactSurfaceEvaluator,
    metric_field: &dyn CurveMetricField,
    control: &dyn GeometryEvaluationControl,
    options: SharedCurveDiscretizationOptions,
) -> Result<SharedCurveGeometryValidationReport, SharedCurveError> {
    validate_options(options)?;
    let context = SharedCurveEvaluationContext::new(
        topology,
        curves,
        pcurves,
        surfaces,
        metric_field,
        control,
    );
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
        context.topology,
        context.curves,
        context.control,
        transform,
        options,
    )?;
    validate_pcurves(curve, edge, context, options)?;
    if edge.is_degenerate {
        validate_degenerate_geometry(
            curve,
            edge,
            context.curves,
            context.control,
            transform,
            options.geometry_absolute_error_m.canonical_value(),
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
        > requested.maximum_chordal_deviation_m.canonical_value()
            + options.geometry_absolute_error_m.canonical_value()
        || maximum_tangent_change_rad > requested.maximum_tangent_change_rad + 1.0e-12
        || minimum_metric_edge_length + 1.0e-12 < requested.minimum_metric_edge_length
        || maximum_metric_edge_length > requested.maximum_metric_edge_length + 1.0e-12
        || curve.achieved.maximum_chordal_deviation_m.canonical_value()
            + options.geometry_absolute_error_m.canonical_value()
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
    topology: &ExactBRepTopology,
    curves: &dyn ExactCurveEvaluator,
    control: &dyn GeometryEvaluationControl,
    transform: GeometryTransform,
    options: SharedCurveDiscretizationOptions,
) -> Result<(), SharedCurveError> {
    let segment_error =
        options.arc_length_absolute_error_m.canonical_value() / (curve.nodes.len() - 1) as f64;
    let mut arc_length_m = 0.0;
    for (index, node) in curve.nodes.iter().enumerate() {
        let point_m = curves
            .point(&edge.curve_evaluator_id, node.parameter, control)
            .map(|point| transform.transform_point(point.canonical_value()))
            .map_err(|error| geometry_error(edge, error))?;
        let coordinate_tolerance_m = node
            .source_vertex_id
            .as_ref()
            .and_then(|vertex_id| {
                topology
                    .vertices
                    .binary_search_by(|vertex| vertex.id.cmp(vertex_id))
                    .ok()
                    .map(|index| topology.vertices[index].tolerance_m.canonical_value())
            })
            .unwrap_or(0.0)
            .max(options.geometry_absolute_error_m.canonical_value());
        if length(sub(point_m, node.coordinates_m.canonical_value())) > coordinate_tolerance_m {
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
        if (arc_length_m - node.arc_length_m.canonical_value()).abs()
            > options.arc_length_absolute_error_m.canonical_value()
        {
            return Err(mismatch(
                edge,
                "curve node arc length",
                "stored cumulative arc length differs from exact integration",
            ));
        }
    }
    Ok(())
}

/// Stored face-use images equal an independent derivation: the exact pcurve image at every
/// node, with source-vertex nodes on their admitted canonical face coordinate.
fn validate_pcurves(
    curve: &SharedCurve,
    edge: &ExactEdge,
    context: SharedCurveEvaluationContext<'_>,
    options: SharedCurveDiscretizationOptions,
) -> Result<(), SharedCurveError> {
    let parameters = curve
        .nodes
        .iter()
        .map(|node| node.parameter)
        .collect::<Vec<_>>();
    let derived = face_uses_for_parameters(
        context.topology,
        edge,
        context.pcurves,
        context.surfaces,
        context.control,
        curve.parameter_range,
        &parameters,
        options,
    )?;
    for face_use in &curve.face_uses {
        let expected = derived
            .iter()
            .find(|derived| derived.coedge_id == face_use.coedge_id)
            .ok_or_else(|| mismatch(edge, "curve face-use UV", "face use has no exact coedge"))?;
        if expected.node_uv.len() != face_use.node_uv.len() {
            return Err(mismatch(
                edge,
                "curve face-use UV",
                "stored pcurve image count differs from the curve nodes",
            ));
        }
        for (exact_uv, stored_uv) in expected.node_uv.iter().zip(&face_use.node_uv) {
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
