use runmat_geometry_core::{
    ExactBRepTopology, ExactCurveEvaluator, ExactEdge, ExactPcurveEvaluator,
    GeometryEvaluationControl, ParameterRange,
};

use crate::shared::{
    shared_curve_node_id, CurveResolutionEvidence, SharedCurve, SharedCurveError,
    SharedCurveErrorKind, SharedCurveFaceUse, SharedCurveMesh, SharedCurveNode,
    SHARED_CURVE_MESH_SCHEMA_VERSION,
};

use super::{
    arc_length::world_arc_length,
    error::{edge_error, geometry_error, require_parameter_range, validate_options},
    sampling::{interval_evidence, EvaluatedPoint, EvaluationCache},
    types::{CurveMetricField, SharedCurveDiscretizationOptions},
};

pub fn discretize_shared_curves(
    topology: &ExactBRepTopology,
    curves: &dyn ExactCurveEvaluator,
    pcurves: &dyn ExactPcurveEvaluator,
    metric_field: &dyn CurveMetricField,
    control: &dyn GeometryEvaluationControl,
    options: SharedCurveDiscretizationOptions,
) -> Result<SharedCurveMesh, SharedCurveError> {
    validate_options(options)?;
    let mut edges = Vec::with_capacity(topology.edges.len());
    for edge in &topology.edges {
        control
            .checkpoint()
            .map_err(|error| geometry_error(edge, error))?;
        edges.push(discretize_edge(
            topology,
            edge,
            curves,
            pcurves,
            metric_field,
            control,
            options,
        )?);
    }
    let mesh = SharedCurveMesh {
        schema_version: SHARED_CURVE_MESH_SCHEMA_VERSION,
        edges,
    };
    mesh.validate_against(topology)
        .map_err(|error| SharedCurveError {
            edge_id: None,
            kind: SharedCurveErrorKind::GeometricMismatch,
            field: error.field,
            reason: error.reason,
        })?;
    Ok(mesh)
}

#[allow(clippy::too_many_arguments)]
fn discretize_edge(
    topology: &ExactBRepTopology,
    edge: &ExactEdge,
    curves: &dyn ExactCurveEvaluator,
    pcurves: &dyn ExactPcurveEvaluator,
    metric_field: &dyn CurveMetricField,
    control: &dyn GeometryEvaluationControl,
    options: SharedCurveDiscretizationOptions,
) -> Result<SharedCurve, SharedCurveError> {
    if edge.is_degenerate {
        return Err(edge_error(
            edge,
            SharedCurveErrorKind::UnsatisfiedConstraint,
            "degenerate exact edge",
            "zero-length exact edges require singular-boundary discretization",
        ));
    }
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
    let mut cache = EvaluationCache::new(edge, curves, metric_field, control, transform);
    let left = cache.sample(parameter_range.start)?;
    let right = cache.sample(parameter_range.end)?;
    let mut samples = vec![left];
    subdivide(&mut cache, left, right, 0, &mut samples, options)?;
    let achieved = resolution_evidence(&mut cache, &samples)?;
    if achieved.minimum_metric_edge_length + 1.0e-12 < options.resolution.minimum_metric_edge_length
    {
        return Err(edge_error(
            edge,
            SharedCurveErrorKind::UnsatisfiedConstraint,
            "minimum metric edge length",
            "upper geometric bounds require an edge shorter than the requested minimum",
        ));
    }

    let arc_error = options.arc_length_absolute_error_m / (samples.len() - 1) as f64;
    let mut arc_length_m = 0.0;
    let mut nodes = Vec::with_capacity(samples.len());
    for (index, sample) in samples.iter().enumerate() {
        if index > 0 {
            arc_length_m += world_arc_length(
                edge,
                curves,
                control,
                transform,
                ParameterRange {
                    start: samples[index - 1].parameter,
                    end: sample.parameter,
                },
                arc_error,
                options.maximum_subdivision_depth,
            )?;
        }
        nodes.push(SharedCurveNode {
            node_id: shared_curve_node_id(&edge.id, sample.parameter),
            source_vertex_id: if index == 0 {
                edge.start_vertex_id.clone()
            } else if index + 1 == samples.len() {
                edge.end_vertex_id.clone()
            } else {
                None
            },
            parameter: sample.parameter,
            arc_length_m,
            coordinates_m: sample.point_m,
        });
    }

    let mut coedges = topology
        .coedges
        .iter()
        .filter(|coedge| coedge.edge_id == edge.id)
        .collect::<Vec<_>>();
    coedges.sort_by(|left, right| left.id.cmp(&right.id));
    let mut face_uses = Vec::with_capacity(coedges.len());
    for coedge in coedges {
        let pcurve_range = pcurves
            .parameter_range(&coedge.pcurve_evaluator_id)
            .map_err(|error| geometry_error(edge, error))?;
        if pcurve_range != parameter_range {
            return Err(edge_error(
                edge,
                SharedCurveErrorKind::GeometricMismatch,
                "pcurve parameter range",
                "edge and coedge evaluator ranges differ",
            ));
        }
        let node_uv = samples
            .iter()
            .map(|sample| {
                pcurves
                    .point(&coedge.pcurve_evaluator_id, sample.parameter, control)
                    .map_err(|error| geometry_error(edge, error))
            })
            .collect::<Result<Vec<_>, _>>()?;
        face_uses.push(SharedCurveFaceUse {
            coedge_id: coedge.id.clone(),
            face_id: coedge.face_id.clone(),
            orientation: coedge.orientation,
            seam_image: coedge.seam_image,
            node_uv,
        });
    }

    Ok(SharedCurve {
        source_edge_id: edge.id.clone(),
        parameter_range,
        nodes,
        face_uses,
        requested: options.resolution,
        achieved,
        metric_resolution: cache.metric_evidence_for(&samples)?,
    })
}

fn subdivide(
    cache: &mut EvaluationCache<'_>,
    left: EvaluatedPoint,
    right: EvaluatedPoint,
    depth: u16,
    output: &mut Vec<EvaluatedPoint>,
    options: SharedCurveDiscretizationOptions,
) -> Result<(), SharedCurveError> {
    let evidence = interval_evidence(cache, left, right)?;
    let violates = evidence.chordal_deviation_m > options.resolution.maximum_chordal_deviation_m
        || evidence.tangent_change_rad > options.resolution.maximum_tangent_change_rad
        || evidence.metric_length > options.resolution.maximum_metric_edge_length;
    if violates {
        if depth >= options.maximum_subdivision_depth
            || output.len() + 1 >= options.maximum_nodes_per_edge as usize
        {
            return Err(edge_error(
                cache.edge,
                SharedCurveErrorKind::ResourceLimit,
                "curve subdivision",
                "required refinement exceeds the per-edge node or depth limit",
            ));
        }
        subdivide(cache, left, evidence.midpoint, depth + 1, output, options)?;
        subdivide(cache, evidence.midpoint, right, depth + 1, output, options)?;
    } else {
        output.push(right);
    }
    Ok(())
}

fn resolution_evidence(
    cache: &mut EvaluationCache<'_>,
    samples: &[EvaluatedPoint],
) -> Result<CurveResolutionEvidence, SharedCurveError> {
    let mut maximum_chordal_deviation_m: f64 = 0.0;
    let mut maximum_tangent_change_rad: f64 = 0.0;
    let mut minimum_metric_edge_length = f64::INFINITY;
    let mut maximum_metric_edge_length: f64 = 0.0;
    for pair in samples.windows(2) {
        let evidence = interval_evidence(cache, pair[0], pair[1])?;
        maximum_chordal_deviation_m = maximum_chordal_deviation_m.max(evidence.chordal_deviation_m);
        maximum_tangent_change_rad = maximum_tangent_change_rad.max(evidence.tangent_change_rad);
        minimum_metric_edge_length = minimum_metric_edge_length.min(evidence.metric_length);
        maximum_metric_edge_length = maximum_metric_edge_length.max(evidence.metric_length);
    }
    Ok(CurveResolutionEvidence {
        maximum_chordal_deviation_m,
        maximum_tangent_change_rad,
        minimum_metric_edge_length,
        maximum_metric_edge_length,
    })
}
