use super::super::{ExactBRepTopology, ExactCoedge, ExactEdge, ExactFace, PersistentEntityId};
use super::{
    ExactCurveEvaluator, ExactPcurveEvaluator, ExactSurfaceEvaluator, GeometryEvaluationControl,
    GeometryEvaluationError, GeometryEvaluationErrorKind, ParameterRange,
};

const INITIAL_INTERVALS: usize = 32;
const MAX_REFINEMENT_DEPTH: u8 = 32;
const MAX_CONSISTENCY_INTERVALS: usize = 1_000_000;

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
            tolerance_m,
            control,
            &additional_parameters,
        )?;
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
        distance(point, vertex.point_m),
        tolerance_m.max(vertex.tolerance_m),
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
    let mut parameters = uniform_parameters(range);
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
    for pair in parameters.windows(2) {
        let start_sample = sample_edge_use(evaluator, edge, coedge, face, pair[0], control)?;
        let end_sample = sample_edge_use(evaluator, edge, coedge, face, pair[1], control)?;
        refine_interval(
            evaluator,
            edge,
            coedge,
            face,
            Interval {
                start: pair[0],
                end: pair[1],
                start_sample,
                end_sample,
                depth: 0,
            },
            tolerance_m,
            control,
        )?;
    }
    Ok(())
}

fn refine_interval<E>(
    evaluator: &E,
    edge: &ExactEdge,
    coedge: &ExactCoedge,
    face: &ExactFace,
    initial: Interval,
    tolerance_m: f64,
    control: &dyn GeometryEvaluationControl,
) -> Result<(), GeometryEvaluationError>
where
    E: ExactCurveEvaluator + ExactPcurveEvaluator + ExactSurfaceEvaluator + ?Sized,
{
    charge_allocation(1, std::mem::size_of::<Interval>(), control)?;
    let mut pending = vec![initial];
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
        let midpoint = interval.start + (interval.end - interval.start) * 0.5;
        if midpoint == interval.start || midpoint == interval.end {
            return Err(budget(
                "geometry consistency exhausted parameter resolution",
            ));
        }
        let midpoint_sample = sample_edge_use(evaluator, edge, coedge, face, midpoint, control)?;
        validate_sample(midpoint_sample, tolerance_m)?;
        let curve_flatness = midpoint_deviation(
            interval.start_sample.curve_point,
            interval.end_sample.curve_point,
            midpoint_sample.curve_point,
        );
        let mapped_flatness = midpoint_deviation(
            interval.start_sample.mapped_point,
            interval.end_sample.mapped_point,
            midpoint_sample.mapped_point,
        );
        if curve_flatness.max(mapped_flatness) <= tolerance_m * 0.25 {
            continue;
        }
        if interval.depth >= MAX_REFINEMENT_DEPTH {
            return Err(budget(
                "geometry consistency could not resolve curvature within its hard depth bound",
            ));
        }
        charge_allocation(2, std::mem::size_of::<Interval>(), control)?;
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
        ExactPcurveEvaluator::point(evaluator, &coedge.pcurve_evaluator_id, parameter, control)?;
    let mapped_point =
        ExactSurfaceEvaluator::point(evaluator, &face.surface_evaluator_id, uv, control)?;
    Ok(Sample {
        curve_point,
        mapped_point,
    })
}

fn uniform_parameters(range: ParameterRange) -> Vec<f64> {
    (0..=INITIAL_INTERVALS)
        .map(|index| {
            range.start + (range.end - range.start) * index as f64 / INITIAL_INTERVALS as f64
        })
        .collect()
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
