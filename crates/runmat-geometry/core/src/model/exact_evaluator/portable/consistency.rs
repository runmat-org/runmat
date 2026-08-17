use super::super::{
    ExactCurveDefinition, ExactCurveEvaluator, ExactCurveImplementation, ExactPcurveDefinition,
    ExactPcurveEvaluator, ExactPcurveImplementation, ExactSurfaceEvaluator,
    GeometryEvaluationControl, GeometryEvaluationError, GeometryEvaluationErrorKind,
    ParameterRange,
};
use super::vector::distance;
use super::{find_by_id, invalid_result, PortableExactEvaluator};

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

impl PortableExactEvaluator<'_> {
    /// Independently checks every portable edge use against its face pcurve and
    /// surface, and checks curve endpoints against admitted topological vertices.
    /// Kernel-owned records must use their ABI-owned consistency validator.
    pub fn validate_incidence_consistency(
        &self,
        tolerance_m: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<(), GeometryEvaluationError> {
        if !tolerance_m.is_finite() || tolerance_m <= 0.0 {
            return Err(invalid_result(
                "geometry consistency tolerance must be finite and positive",
            ));
        }
        for edge in &self.topology.edges {
            control.checkpoint()?;
            let range = ExactCurveEvaluator::parameter_range(self, &edge.curve_evaluator_id)?;
            if let Some(vertex_id) = &edge.start_vertex_id {
                self.validate_vertex(
                    vertex_id,
                    &edge.curve_evaluator_id,
                    range.start,
                    tolerance_m,
                    control,
                )?;
            }
            if let Some(vertex_id) = &edge.end_vertex_id {
                self.validate_vertex(
                    vertex_id,
                    &edge.curve_evaluator_id,
                    range.end,
                    tolerance_m,
                    control,
                )?;
            }
        }
        for coedge in &self.topology.coedges {
            control.checkpoint()?;
            let edge = find_by_id(
                &self.topology.edges,
                &coedge.edge_id,
                |edge| &edge.id,
                "edge",
            )?;
            let face = find_by_id(
                &self.topology.faces,
                &coedge.face_id,
                |face| &face.id,
                "face",
            )?;
            self.validate_edge_use(edge, coedge, face, tolerance_m, control)?;
        }
        Ok(())
    }

    fn validate_vertex(
        &self,
        vertex_id: &super::super::super::PersistentEntityId,
        curve_id: &super::super::CurveEvaluatorId,
        parameter: f64,
        tolerance_m: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<(), GeometryEvaluationError> {
        let vertex = find_by_id(
            &self.topology.vertices,
            vertex_id,
            |vertex| &vertex.id,
            "vertex",
        )?;
        let point = ExactCurveEvaluator::point(self, curve_id, parameter, control)?;
        require_consistent(
            distance(&point, &vertex.point_m),
            tolerance_m.max(vertex.tolerance_m),
            "curve endpoint disagrees with its topological vertex",
        )
    }

    fn validate_edge_use(
        &self,
        edge: &super::super::super::ExactEdge,
        coedge: &super::super::super::ExactCoedge,
        face: &super::super::super::ExactFace,
        tolerance_m: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<(), GeometryEvaluationError> {
        let range = ExactCurveEvaluator::parameter_range(self, &edge.curve_evaluator_id)?;
        let curve_record = self.curve_record(&edge.curve_evaluator_id)?;
        let pcurve_record = self.pcurve_record(&coedge.pcurve_evaluator_id)?;
        let parameters = consistency_parameters(
            range,
            &curve_record.implementation,
            &pcurve_record.implementation,
        );
        charge_allocation(parameters.len(), std::mem::size_of::<f64>(), control)?;
        for pair in parameters.windows(2) {
            let start_sample = self.sample_edge_use(edge, coedge, face, pair[0], control)?;
            let end_sample = self.sample_edge_use(edge, coedge, face, pair[1], control)?;
            self.refine_interval(
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

    fn refine_interval(
        &self,
        edge: &super::super::super::ExactEdge,
        coedge: &super::super::super::ExactCoedge,
        face: &super::super::super::ExactFace,
        initial: Interval,
        tolerance_m: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<(), GeometryEvaluationError> {
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
            let midpoint_sample = self.sample_edge_use(edge, coedge, face, midpoint, control)?;
            validate_sample(midpoint_sample, tolerance_m)?;
            let curve_flatness = midpoint_deviation(
                &interval.start_sample.curve_point,
                &interval.end_sample.curve_point,
                &midpoint_sample.curve_point,
            );
            let mapped_flatness = midpoint_deviation(
                &interval.start_sample.mapped_point,
                &interval.end_sample.mapped_point,
                &midpoint_sample.mapped_point,
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

    fn sample_edge_use(
        &self,
        edge: &super::super::super::ExactEdge,
        coedge: &super::super::super::ExactCoedge,
        face: &super::super::super::ExactFace,
        parameter: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<Sample, GeometryEvaluationError> {
        let curve_point =
            ExactCurveEvaluator::point(self, &edge.curve_evaluator_id, parameter, control)?;
        let uv =
            ExactPcurveEvaluator::point(self, &coedge.pcurve_evaluator_id, parameter, control)?;
        let mapped_point =
            ExactSurfaceEvaluator::point(self, &face.surface_evaluator_id, uv, control)?;
        Ok(Sample {
            curve_point,
            mapped_point,
        })
    }
}

fn consistency_parameters(
    range: ParameterRange,
    curve: &ExactCurveImplementation,
    pcurve: &ExactPcurveImplementation,
) -> Vec<f64> {
    let mut parameters = (0..=INITIAL_INTERVALS)
        .map(|index| {
            range.start + (range.end - range.start) * index as f64 / INITIAL_INTERVALS as f64
        })
        .collect::<Vec<_>>();
    if let ExactCurveImplementation::Portable {
        definition: ExactCurveDefinition::Nurbs { definition },
    } = curve
    {
        parameters.extend(
            definition
                .knots
                .iter()
                .copied()
                .filter(|value| *value > range.start && *value < range.end),
        );
    }
    if let ExactPcurveImplementation::Portable {
        definition: ExactPcurveDefinition::Nurbs { definition },
    } = pcurve
    {
        parameters.extend(
            definition
                .knots
                .iter()
                .copied()
                .filter(|value| *value > range.start && *value < range.end),
        );
    }
    parameters.sort_by(f64::total_cmp);
    parameters.dedup_by(|left, right| left.to_bits() == right.to_bits());
    parameters
}

fn validate_sample(sample: Sample, tolerance_m: f64) -> Result<(), GeometryEvaluationError> {
    require_consistent(
        distance(&sample.curve_point, &sample.mapped_point),
        tolerance_m,
        "3D curve and surface-evaluated pcurve disagree",
    )
}

fn require_consistent(
    discrepancy_m: f64,
    tolerance_m: f64,
    reason: &str,
) -> Result<(), GeometryEvaluationError> {
    if !discrepancy_m.is_finite() || discrepancy_m > tolerance_m {
        return Err(GeometryEvaluationError::new(
            GeometryEvaluationErrorKind::InconsistentGeometry,
            format!("{reason}: discrepancy {discrepancy_m:e} m exceeds {tolerance_m:e} m"),
        ));
    }
    Ok(())
}

fn midpoint_deviation(start: &[f64; 3], end: &[f64; 3], midpoint: &[f64; 3]) -> f64 {
    let chord_midpoint = std::array::from_fn(|axis| start[axis] + (end[axis] - start[axis]) * 0.5);
    distance(&chord_midpoint, midpoint)
}

fn charge_allocation(
    count: usize,
    item_bytes: usize,
    control: &dyn GeometryEvaluationControl,
) -> Result<(), GeometryEvaluationError> {
    let bytes = count
        .checked_mul(item_bytes)
        .ok_or_else(|| invalid_result("geometry consistency allocation-byte count overflow"))?;
    control.consume_allocation_bytes(u64::try_from(bytes).map_err(|_| {
        invalid_result("geometry consistency allocation-byte count does not fit u64")
    })?)
}

fn budget(reason: &str) -> GeometryEvaluationError {
    GeometryEvaluationError::new(GeometryEvaluationErrorKind::BudgetExceeded, reason)
}
