use super::super::{ExactBRepTopology, ExactCoedge, ExactEdge, ExactFace, PersistentEntityId};
use super::{
    ExactCurveEvaluator, ExactPcurveEvaluator, ExactSurfaceEvaluator, GeometryEvaluationControl,
    GeometryEvaluationError, GeometryEvaluationErrorKind, ParameterRange,
};

const INITIAL_INTERVALS: usize = 32;
const MAX_REFINEMENT_DEPTH: u8 = 32;
const MAX_CONSISTENCY_INTERVALS: usize = 1_000_000;
/// An edge tolerance written by a CAD kernel bounds the gap between the 3D curve and each pcurve
/// at the writer's control points (OCCT checks 23 evenly spaced parameters); between them the
/// true gap of two smooth curves can exceed it slightly. This check bisects adaptively and finds
/// that larger maximum, so it admits a small fixed allowance over the written edge tolerance.
/// Exported CAD in the meshing qualification corpus exceeded its edge tolerance by under 1%.
const EDGE_TOLERANCE_SAMPLING_ALLOWANCE: f64 = 1.1;

#[derive(Clone, Copy)]
struct Sample {
    curve_point: [f64; 3],
    mapped_point: [f64; 3],
}

#[derive(Clone, Copy)]
struct Interval {
    start: f64,
    end: f64,
    start_sample: Sample,
    end_sample: Sample,
    depth: u8,
}

struct EdgeUse<'a> {
    edge: &'a ExactEdge,
    coedge: &'a ExactCoedge,
    face: &'a ExactFace,
}

/// Independently checks exact curve endpoints and every curve/pcurve/surface incidence.
///
/// The evaluator owns representation-specific queries; geometry core owns sampling, adaptive
/// refinement, budgets, and the consistency decision.
pub fn validate_exact_incidence<E>(
    topology: &ExactBRepTopology,
    evaluator: &E,
    tolerance_m: f64,
    control: &dyn GeometryEvaluationControl,
) -> Result<(), GeometryEvaluationError>
where
    E: ExactCurveEvaluator + ExactPcurveEvaluator + ExactSurfaceEvaluator + ?Sized,
{
    validate_exact_incidence_with_parameters(
        topology,
        evaluator,
        tolerance_m,
        control,
        |_, _, _| Ok(Vec::new()),
    )
}

pub(crate) fn validate_exact_incidence_with_parameters<E, F>(
    topology: &ExactBRepTopology,
    evaluator: &E,
    tolerance_m: f64,
    control: &dyn GeometryEvaluationControl,
    additional_parameters: F,
) -> Result<(), GeometryEvaluationError>
where
    E: ExactCurveEvaluator + ExactPcurveEvaluator + ExactSurfaceEvaluator + ?Sized,
    F: Fn(&ExactEdge, &ExactCoedge, ParameterRange) -> Result<Vec<f64>, GeometryEvaluationError>,
{
    if !tolerance_m.is_finite() || tolerance_m <= 0.0 {
        return Err(invalid(
            "geometry consistency tolerance must be finite and positive",
        ));
    }
    for edge in &topology.edges {
        control.checkpoint()?;
        let range = ExactCurveEvaluator::parameter_range(evaluator, &edge.curve_evaluator_id)?;
        if let Some(vertex_id) = &edge.start_vertex_id {
            validate_vertex(
                topology,
                evaluator,
                vertex_id,
                edge,
                range.start,
                tolerance_m,
                control,
            )?;
        }
        if let Some(vertex_id) = &edge.end_vertex_id {
            validate_vertex(
                topology,
                evaluator,
                vertex_id,
                edge,
                range.end,
                tolerance_m,
                control,
            )?;
        }
    }
    for coedge in &topology.coedges {
        control.checkpoint()?;
        let edge = find_by_id(&topology.edges, &coedge.edge_id, |edge| &edge.id, "edge")?;
        let face = find_by_id(&topology.faces, &coedge.face_id, |face| &face.id, "face")?;
        validate_edge_use(
            evaluator,
            edge,
            coedge,
            face,
            tolerance_m.max(edge.tolerance_m.canonical_value() * EDGE_TOLERANCE_SAMPLING_ALLOWANCE),
            control,
            &additional_parameters,
        )
        .map_err(|error| with_edge_use_context(error, edge, face))?;
    }
    Ok(())
}

fn validate_vertex<E>(
    topology: &ExactBRepTopology,
    evaluator: &E,
    vertex_id: &PersistentEntityId,
    edge: &ExactEdge,
    parameter: f64,
    tolerance_m: f64,
    control: &dyn GeometryEvaluationControl,
) -> Result<(), GeometryEvaluationError>
where
    E: ExactCurveEvaluator + ?Sized,
{
    let vertex = find_by_id(&topology.vertices, vertex_id, |vertex| &vertex.id, "vertex")?;
    let point =
        ExactCurveEvaluator::point(evaluator, &edge.curve_evaluator_id, parameter, control)?;
    require_consistent(
        distance(point.canonical_value(), vertex.point_m.canonical_value()),
        tolerance_m.max(vertex.tolerance_m.canonical_value()),
        format!(
            "curve endpoint {:?} at parameter {parameter:e} disagrees with topological vertex {:?}",
            edge.id, vertex.id
        ),
    )
}

fn validate_edge_use<E, F>(
    evaluator: &E,
    edge: &ExactEdge,
    coedge: &ExactCoedge,
    face: &ExactFace,
    tolerance_m: f64,
    control: &dyn GeometryEvaluationControl,
    additional_parameters: &F,
) -> Result<(), GeometryEvaluationError>
where
    E: ExactCurveEvaluator + ExactPcurveEvaluator + ExactSurfaceEvaluator + ?Sized,
    F: Fn(&ExactEdge, &ExactCoedge, ParameterRange) -> Result<Vec<f64>, GeometryEvaluationError>,
{
    let range = ExactCurveEvaluator::parameter_range(evaluator, &edge.curve_evaluator_id)?;
    let pcurve_range =
        ExactPcurveEvaluator::parameter_range(evaluator, &coedge.pcurve_evaluator_id)?;
    if range != pcurve_range {
        return Err(inconsistent(
            "3D curve and face-use pcurve parameter domains disagree",
        ));
    }
    let mut parameters = uniform_parameters(range)?;
    parameters.extend(additional_parameters(edge, coedge, range)?);
    parameters.retain(|value| value.is_finite() && *value >= range.start && *value <= range.end);
    parameters.sort_by(f64::total_cmp);
    parameters.dedup_by(|left, right| left.to_bits() == right.to_bits());
    if parameters.first() != Some(&range.start) || parameters.last() != Some(&range.end) {
        return Err(invalid(
            "geometry consistency parameters do not cover the evaluator domain",
        ));
    }
    charge_allocation(parameters.len(), std::mem::size_of::<f64>(), control)?;
    // Depth-first bisection holds at most one pending sibling per level, so one stack of
    // `MAX_REFINEMENT_DEPTH + 1` intervals serves every interval of this edge use.
    let stack_capacity = usize::from(MAX_REFINEMENT_DEPTH) + 1;
    charge_allocation(stack_capacity, std::mem::size_of::<Interval>(), control)?;
    let mut pending = Vec::with_capacity(stack_capacity);
    for pair in parameters.windows(2) {
        let start_sample = sample_edge_use(evaluator, edge, coedge, face, pair[0], control)?;
        let end_sample = sample_edge_use(evaluator, edge, coedge, face, pair[1], control)?;
        refine_interval(
            evaluator,
            EdgeUse { edge, coedge, face },
            Interval {
                start: pair[0],
                end: pair[1],
                start_sample,
                end_sample,
                depth: 0,
            },
            tolerance_m,
            control,
            &mut pending,
        )?;
    }
    Ok(())
}

fn refine_interval<E>(
    evaluator: &E,
    edge_use: EdgeUse<'_>,
    initial: Interval,
    tolerance_m: f64,
    control: &dyn GeometryEvaluationControl,
    pending: &mut Vec<Interval>,
) -> Result<(), GeometryEvaluationError>
where
    E: ExactCurveEvaluator + ExactPcurveEvaluator + ExactSurfaceEvaluator + ?Sized,
{
    let EdgeUse { edge, coedge, face } = edge_use;
    pending.clear();
    pending.push(initial);
    let mut visited = 0usize;
    while let Some(interval) = pending.pop() {
        control.checkpoint()?;
        control.consume_search_work(1)?;
        visited = visited.saturating_add(1);
        if visited > MAX_CONSISTENCY_INTERVALS {
            return Err(budget(
                "geometry consistency exceeded its hard interval bound",
            ));
        }
        validate_sample(interval.start_sample, tolerance_m)?;
        validate_sample(interval.end_sample, tolerance_m)?;
        let midpoint = interval.start.midpoint(interval.end);
        if midpoint == interval.start || midpoint == interval.end {
            return Err(budget(
                "geometry consistency exhausted parameter resolution",
            ));
        }
        let midpoint_sample = sample_edge_use(evaluator, edge, coedge, face, midpoint, control)?;
        validate_sample(midpoint_sample, tolerance_m)?;
        // The validated quantity is the gap g(t) = curve(t) - surface(pcurve(t)). Stop refining
        // once g is linear between the interval ends to a quarter of the tolerance: the samples
        // then bound g on the interval. Requiring each curve itself to be that flat would bisect
        // a curved edge with a nanometre tolerance into micrometre intervals.
        let gap_curvature = midpoint_deviation(
            gap(interval.start_sample),
            gap(interval.end_sample),
            gap(midpoint_sample),
        );
        if gap_curvature <= tolerance_m * 0.25 {
            continue;
        }
        if interval.depth >= MAX_REFINEMENT_DEPTH {
            return Err(budget(
                "geometry consistency could not resolve curvature within its hard depth bound",
            ));
        }
        let depth = interval.depth + 1;
        pending.push(Interval {
            start: midpoint,
            end: interval.end,
            start_sample: midpoint_sample,
            end_sample: interval.end_sample,
            depth,
        });
        pending.push(Interval {
            start: interval.start,
            end: midpoint,
            start_sample: interval.start_sample,
            end_sample: midpoint_sample,
            depth,
        });
    }
    Ok(())
}

fn sample_edge_use<E>(
    evaluator: &E,
    edge: &ExactEdge,
    coedge: &ExactCoedge,
    face: &ExactFace,
    parameter: f64,
    control: &dyn GeometryEvaluationControl,
) -> Result<Sample, GeometryEvaluationError>
where
    E: ExactCurveEvaluator + ExactPcurveEvaluator + ExactSurfaceEvaluator + ?Sized,
{
    let curve_point =
        ExactCurveEvaluator::point(evaluator, &edge.curve_evaluator_id, parameter, control)?;
    let uv =
        ExactPcurveEvaluator::point(evaluator, &coedge.pcurve_evaluator_id, parameter, control)?
            .map(|coordinate| coordinate.with_native_coordinate(|value, _axis| value));
    let mapped_point =
        ExactSurfaceEvaluator::point(evaluator, &face.surface_evaluator_id, uv, control)?;
    Ok(Sample {
        curve_point: curve_point.canonical_value(),
        mapped_point: mapped_point.canonical_value(),
    })
}

fn uniform_parameters(range: ParameterRange) -> Result<Vec<f64>, GeometryEvaluationError> {
    Ok(range
        .uniform_samples::<{ INITIAL_INTERVALS + 1 }>()?
        .to_vec())
}

fn validate_sample(sample: Sample, tolerance_m: f64) -> Result<(), GeometryEvaluationError> {
    require_consistent(
        distance(sample.curve_point, sample.mapped_point),
        tolerance_m,
        "3D curve and surface-evaluated pcurve disagree",
    )
}

fn require_consistent(
    discrepancy_m: f64,
    tolerance_m: f64,
    reason: impl Into<String>,
) -> Result<(), GeometryEvaluationError> {
    if !discrepancy_m.is_finite() || discrepancy_m > tolerance_m {
        let reason = reason.into();
        return Err(GeometryEvaluationError::new(
            GeometryEvaluationErrorKind::InconsistentGeometry,
            format!("{reason}: discrepancy {discrepancy_m:e} m exceeds {tolerance_m:e} m"),
        ));
    }
    Ok(())
}

fn with_edge_use_context(
    error: GeometryEvaluationError,
    edge: &ExactEdge,
    face: &ExactFace,
) -> GeometryEvaluationError {
    if error.kind != GeometryEvaluationErrorKind::InconsistentGeometry {
        return error;
    }
    GeometryEvaluationError::new(
        error.kind,
        format!(
            "{} (edge {} on face {}, written edge tolerance {:e} m)",
            error.reason,
            edge.id.source_topology_id,
            face.id.source_topology_id,
            edge.tolerance_m.canonical_value()
        ),
    )
}

fn gap(sample: Sample) -> [f64; 3] {
    std::array::from_fn(|axis| sample.curve_point[axis] - sample.mapped_point[axis])
}

fn midpoint_deviation(start: [f64; 3], end: [f64; 3], midpoint: [f64; 3]) -> f64 {
    let chord_midpoint = std::array::from_fn(|axis| start[axis] + (end[axis] - start[axis]) * 0.5);
    distance(chord_midpoint, midpoint)
}

fn distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter()
        .zip(right)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt()
}

fn find_by_id<'a, T>(
    values: &'a [T],
    id: &PersistentEntityId,
    key: impl Fn(&T) -> &PersistentEntityId,
    kind: &str,
) -> Result<&'a T, GeometryEvaluationError> {
    values
        .binary_search_by(|value| key(value).cmp(id))
        .map(|index| &values[index])
        .map_err(|_| invalid(format!("admitted {kind} index is incomplete")))
}

fn charge_allocation(
    count: usize,
    item_bytes: usize,
    control: &dyn GeometryEvaluationControl,
) -> Result<(), GeometryEvaluationError> {
    let bytes = count
        .checked_mul(item_bytes)
        .ok_or_else(|| invalid("geometry consistency allocation-byte count overflow"))?;
    control.consume_allocation_bytes(
        u64::try_from(bytes)
            .map_err(|_| invalid("geometry consistency allocation-byte count does not fit u64"))?,
    )
}

fn budget(reason: &str) -> GeometryEvaluationError {
    GeometryEvaluationError::new(GeometryEvaluationErrorKind::BudgetExceeded, reason)
}

fn inconsistent(reason: impl Into<String>) -> GeometryEvaluationError {
    GeometryEvaluationError::new(GeometryEvaluationErrorKind::InconsistentGeometry, reason)
}

fn invalid(reason: impl Into<String>) -> GeometryEvaluationError {
    GeometryEvaluationError::new(GeometryEvaluationErrorKind::InvalidResult, reason)
}

#[cfg(test)]
mod sampling_tests {
    use super::*;
    #[test]
    fn sampling_preserves_non_representable_domain_endpoints() {
        for range in [
            ParameterRange {
                start: -3.0,
                end: 0.1,
            },
            ParameterRange {
                start: -1.0,
                end: 0.123456789,
            },
        ] {
            let parameters = uniform_parameters(range).unwrap();
            assert_eq!(parameters[0].to_bits(), range.start.to_bits());
            assert_eq!(parameters[INITIAL_INTERVALS].to_bits(), range.end.to_bits());
            assert!(parameters
                .iter()
                .all(|value| *value >= range.start && *value <= range.end));
        }
    }
}
