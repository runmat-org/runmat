use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{
    ExactBRepTopology, ExactCoedge, ExactEdge, PersistentEntityId, PersistentEntityKind,
};
use runmat_meshing_core::MetricSourceKind;

use super::{
    shared_curve_node_id, shared_degenerate_curve_node_id, CurveMetricResolutionEvidence,
    CurveResolutionEvidence, CurveResolutionPolicy, SharedCurve, SharedCurveError, SharedCurveMesh,
    SHARED_CURVE_MESH_SCHEMA_VERSION,
};

const MAX_CURVE_NODES: usize = 20_000_000;
const MAX_CURVE_FACE_USES: usize = 20_000_000;
const MAX_CURVE_UV_SAMPLES: usize = 100_000_000;

pub(super) fn validate_shared_curve_mesh(
    mesh: &SharedCurveMesh,
    topology: &ExactBRepTopology,
) -> Result<(), SharedCurveError> {
    if mesh.schema_version != SHARED_CURVE_MESH_SCHEMA_VERSION {
        return Err(invalid("shared curve schema", "unsupported version"));
    }
    let edge_by_id = topology
        .edges
        .iter()
        .map(|edge| (&edge.id, edge))
        .collect::<BTreeMap<_, _>>();
    let coedges_by_edge = topology.coedges.iter().fold(
        BTreeMap::<&PersistentEntityId, Vec<&ExactCoedge>>::new(),
        |mut index, coedge| {
            index.entry(&coedge.edge_id).or_default().push(coedge);
            index
        },
    );
    let mut previous_edge = None;
    let mut node_count = 0usize;
    let mut face_use_count = 0usize;
    let mut uv_sample_count = 0usize;
    for curve in &mesh.edges {
        if previous_edge.is_some_and(|previous| previous >= &curve.source_edge_id) {
            return Err(invalid(
                "shared curve edges",
                "edges must be canonical and unique",
            ));
        }
        previous_edge = Some(&curve.source_edge_id);
        let edge = edge_by_id.get(&curve.source_edge_id).ok_or_else(|| {
            invalid(
                "shared curve edge",
                "source edge does not exist in exact topology",
            )
        })?;
        validate_curve(curve, edge, coedges_by_edge.get(&curve.source_edge_id))?;
        node_count = node_count.saturating_add(curve.nodes.len());
        face_use_count = face_use_count.saturating_add(curve.face_uses.len());
        uv_sample_count = curve
            .face_uses
            .iter()
            .fold(uv_sample_count, |count, face_use| {
                count.saturating_add(face_use.node_uv.len())
            });
    }
    if mesh.edges.len() != topology.edges.len()
        || node_count > MAX_CURVE_NODES
        || face_use_count > MAX_CURVE_FACE_USES
        || uv_sample_count > MAX_CURVE_UV_SAMPLES
    {
        return Err(invalid(
            "shared curve inventory",
            "every exact edge must appear once within hard aggregate bounds",
        ));
    }
    Ok(())
}

fn validate_curve(
    curve: &SharedCurve,
    edge: &ExactEdge,
    expected_coedges: Option<&Vec<&ExactCoedge>>,
) -> Result<(), SharedCurveError> {
    if curve.source_edge_id.kind != PersistentEntityKind::Edge
        || !finite_range(curve.parameter_range.start, curve.parameter_range.end)
        || curve.nodes.len() < 2
    {
        return Err(invalid(
            "shared curve edge",
            "edge kind, parameter range, or node inventory is invalid",
        ));
    }
    validate_resolution(curve.requested, curve.achieved, edge.is_degenerate)?;
    validate_metric_resolution(curve, edge.is_degenerate)?;
    let mut node_ids = BTreeSet::new();
    for (index, node) in curve.nodes.iter().enumerate() {
        let expected_node_id = if edge.is_degenerate {
            shared_degenerate_curve_node_id(&curve.source_edge_id)
        } else {
            shared_curve_node_id(&curve.source_edge_id, node.parameter)
        };
        if (!edge.is_degenerate && !node_ids.insert(node.node_id))
            || node.node_id != expected_node_id
            || !node.parameter.is_finite()
            || !node.arc_length_m.is_finite()
            || node.arc_length_m < 0.0
            || node.coordinates_m.iter().any(|value| !value.is_finite())
        {
            return Err(invalid(
                "shared curve node",
                "node identity and finite geometry must be canonical",
            ));
        }
        if index > 0
            && (curve.nodes[index - 1].parameter >= node.parameter
                || (!edge.is_degenerate
                    && curve.nodes[index - 1].arc_length_m >= node.arc_length_m))
        {
            return Err(invalid(
                "shared curve node order",
                "parameters and arc length must increase strictly",
            ));
        }
    }
    if edge.is_degenerate
        && curve.nodes.iter().any(|node| {
            node.arc_length_m.to_bits() != 0
                || node
                    .coordinates_m
                    .iter()
                    .zip(curve.nodes[0].coordinates_m)
                    .any(|(actual, expected)| actual.to_bits() != expected.to_bits())
        })
    {
        return Err(invalid(
            "degenerate shared curve",
            "all parameter samples must identify one coordinate with zero arc length",
        ));
    }
    let first = &curve.nodes[0];
    let last = curve.nodes.last().expect("node count checked");
    if first.parameter != curve.parameter_range.start
        || last.parameter != curve.parameter_range.end
        || first.arc_length_m != 0.0
        || first.source_vertex_id != edge.start_vertex_id
        || last.source_vertex_id != edge.end_vertex_id
        || curve.nodes[1..curve.nodes.len() - 1]
            .iter()
            .any(|node| node.source_vertex_id.is_some())
    {
        return Err(invalid(
            "shared curve endpoints",
            "endpoints must bind the exact range and source vertices",
        ));
    }

    let mut expected = expected_coedges.cloned().unwrap_or_default();
    expected.sort_by(|left, right| left.id.cmp(&right.id));
    if curve.face_uses.len() != expected.len() {
        return Err(invalid(
            "shared curve face uses",
            "every exact coedge must consume the shared curve once",
        ));
    }
    for (actual, expected) in curve.face_uses.iter().zip(expected) {
        if actual.coedge_id != expected.id
            || actual.face_id != expected.face_id
            || actual.orientation != expected.orientation
            || actual.seam_image != expected.seam_image
            || actual.node_uv.len() != curve.nodes.len()
            || actual
                .node_uv
                .iter()
                .flatten()
                .any(|value| !value.is_finite())
        {
            return Err(invalid(
                "shared curve face use",
                "coedge provenance and per-node UV images must be complete and canonical",
            ));
        }
    }
    Ok(())
}

fn validate_metric_resolution(
    curve: &SharedCurve,
    is_degenerate: bool,
) -> Result<(), SharedCurveError> {
    match &curve.metric_resolution {
        CurveMetricResolutionEvidence::DegenerateTopologicalCollapse if is_degenerate => Ok(()),
        CurveMetricResolutionEvidence::Evaluated {
            active_sources,
            evaluation_count,
            applied_contribution_count,
            minimum_tangent_target_size_m,
            maximum_tangent_target_size_m,
            clipped_contribution_count,
            rejected_contribution_count,
            ..
        } if !is_degenerate
            && !active_sources.is_empty()
            && active_sources.first() == Some(&MetricSourceKind::Global)
            && *evaluation_count >= curve.nodes.len() as u64
            && u64::from(*applied_contribution_count)
                <= evaluation_count.saturating_mul(65_536)
            && u64::from(*clipped_contribution_count)
                <= evaluation_count.saturating_mul(65_536)
            && u64::from(*rejected_contribution_count)
                <= evaluation_count.saturating_mul(65_536)
            && minimum_tangent_target_size_m.is_finite()
            && *minimum_tangent_target_size_m > 0.0
            && maximum_tangent_target_size_m.is_finite()
            && minimum_tangent_target_size_m <= maximum_tangent_target_size_m
            && !active_sources
                .windows(2)
                .any(|pair| metric_source_rank(pair[0]) >= metric_source_rank(pair[1])) =>
        {
            Ok(())
        }
        _ => Err(invalid(
            "shared curve metric resolution",
            "evidence kind, sources, count, and tangent target range must match edge topology",
        )),
    }
}

pub(super) const fn metric_source_rank(source: MetricSourceKind) -> u8 {
    source.canonical_rank()
}

fn validate_resolution(
    requested: CurveResolutionPolicy,
    achieved: CurveResolutionEvidence,
    is_degenerate: bool,
) -> Result<(), SharedCurveError> {
    let requested_values = [
        requested.maximum_chordal_deviation_m,
        requested.maximum_tangent_change_rad,
        requested.minimum_metric_edge_length,
        requested.maximum_metric_edge_length,
    ];
    let achieved_values = [
        achieved.maximum_chordal_deviation_m,
        achieved.maximum_tangent_change_rad,
        achieved.minimum_metric_edge_length,
        achieved.maximum_metric_edge_length,
    ];
    if requested_values
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
        || requested.minimum_metric_edge_length > requested.maximum_metric_edge_length
    {
        return Err(invalid(
            "shared curve resolution",
            "requested bounds must be finite, positive, and ordered",
        ));
    }
    if is_degenerate {
        return if achieved_values.iter().all(|value| value.to_bits() == 0) {
            Ok(())
        } else {
            Err(invalid(
                "shared curve resolution",
                "degenerate edges must record zero achieved geometric and metric extent",
            ))
        };
    }
    if achieved_values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
        || achieved.minimum_metric_edge_length <= 0.0
        || achieved.minimum_metric_edge_length > achieved.maximum_metric_edge_length
        || achieved.minimum_metric_edge_length < requested.minimum_metric_edge_length
        || achieved.maximum_chordal_deviation_m > requested.maximum_chordal_deviation_m
        || achieved.maximum_tangent_change_rad > requested.maximum_tangent_change_rad
        || achieved.maximum_metric_edge_length > requested.maximum_metric_edge_length
    {
        return Err(invalid(
            "shared curve resolution",
            "requested bounds must be positive and achieved maxima must satisfy them",
        ));
    }
    Ok(())
}

fn finite_range(start: f64, end: f64) -> bool {
    start.is_finite() && end.is_finite() && start < end
}

pub(super) fn invalid(field: impl Into<String>, reason: impl Into<String>) -> SharedCurveError {
    SharedCurveError::invalid_contract(field, reason)
}
