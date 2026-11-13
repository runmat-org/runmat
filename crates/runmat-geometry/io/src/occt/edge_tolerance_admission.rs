//! Projects the kernel's edge tolerance admissions into import evidence and names each edge by
//! its persistent identity.

use std::collections::BTreeMap;

use runmat_geometry_core::{
    ExactBRepTopology, ExactCurveEvaluator, ExactEdge, ExactVertex, GeometryEvaluationControl,
    GeometryEvaluationError, PersistentEntityId,
};

use super::ffi::bridge::OcctEdgeToleranceAdmissionPayload;
use crate::exact::EdgeToleranceAdmission;

pub(super) fn project(
    payloads: &[OcctEdgeToleranceAdmissionPayload],
    meters_per_source_unit: f64,
    bound_m: f64,
    topology: &ExactBRepTopology,
    evaluator: &dyn ExactCurveEvaluator,
    control: &dyn GeometryEvaluationControl,
) -> Result<Vec<EdgeToleranceAdmission>, GeometryEvaluationError> {
    let scale = |x: f64, y: f64, z: f64| {
        [
            x * meters_per_source_unit,
            y * meters_per_source_unit,
            z * meters_per_source_unit,
        ]
    };
    let vertices: BTreeMap<&PersistentEntityId, &ExactVertex> = topology
        .vertices
        .iter()
        .map(|vertex| (&vertex.id, vertex))
        .collect();
    payloads
        .iter()
        .map(|payload| {
            control.checkpoint()?;
            let mut admission = EdgeToleranceAdmission {
                edge_id: None,
                start_m: scale(payload.start_x, payload.start_y, payload.start_z),
                middle_m: scale(payload.middle_x, payload.middle_y, payload.middle_z),
                end_m: scale(payload.end_x, payload.end_y, payload.end_z),
                written_tolerance_m: payload.written_tolerance * meters_per_source_unit,
                gap_before_m: payload.gap_before * meters_per_source_unit,
                gap_after_m: payload.gap_after * meters_per_source_unit,
                admitted_tolerance_m: payload.admitted_tolerance * meters_per_source_unit,
                bound_m,
                repaired: payload.repaired,
                admitted: payload.within_bound,
            };
            admission.edge_id = identify(&admission, topology, &vertices, evaluator, control)?;
            Ok(admission)
        })
        .collect()
}

/// The one imported edge whose endpoints match the admission's curve endpoints, using the
/// curve midpoint to choose among edges that share both endpoints (closed or parallel edges).
fn identify(
    admission: &EdgeToleranceAdmission,
    topology: &ExactBRepTopology,
    vertices: &BTreeMap<&PersistentEntityId, &ExactVertex>,
    evaluator: &dyn ExactCurveEvaluator,
    control: &dyn GeometryEvaluationControl,
) -> Result<Option<PersistentEntityId>, GeometryEvaluationError> {
    let vertex =
        |id: &Option<PersistentEntityId>| id.as_ref().and_then(|id| vertices.get(id).copied());
    let near = |left: [f64; 3], right: [f64; 3], tolerance: f64| {
        (0..3)
            .map(|axis| (left[axis] - right[axis]).powi(2))
            .sum::<f64>()
            <= tolerance * tolerance
    };
    let candidates: Vec<&ExactEdge> = topology
        .edges
        .iter()
        .filter(|edge| {
            let (Some(start), Some(end)) =
                (vertex(&edge.start_vertex_id), vertex(&edge.end_vertex_id))
            else {
                return false;
            };
            let tolerance = 2.0
                * admission
                    .admitted_tolerance_m
                    .max(start.tolerance_m.canonical_value())
                    .max(end.tolerance_m.canonical_value());
            let start = start.point_m.canonical_value();
            let end = end.point_m.canonical_value();
            (near(start, admission.start_m, tolerance) && near(end, admission.end_m, tolerance))
                || (near(start, admission.end_m, tolerance)
                    && near(end, admission.start_m, tolerance))
        })
        .collect();
    if candidates.len() <= 1 {
        return Ok(candidates.first().map(|edge| edge.id.clone()));
    }
    let mut best: Option<(f64, &ExactEdge)> = None;
    for edge in candidates {
        let range = evaluator.parameter_range(&edge.curve_evaluator_id)?;
        let middle = evaluator
            .point(
                &edge.curve_evaluator_id,
                0.5 * (range.start + range.end),
                control,
            )?
            .canonical_value();
        let distance = (0..3)
            .map(|axis| (middle[axis] - admission.middle_m[axis]).powi(2))
            .sum::<f64>();
        if best.is_none_or(|(closest, _)| distance < closest) {
            best = Some((distance, edge));
        }
    }
    Ok(best.map(|(_, edge)| edge.id.clone()))
}
