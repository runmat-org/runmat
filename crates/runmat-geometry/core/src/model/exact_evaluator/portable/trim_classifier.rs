use super::super::super::{PersistentEntityId, TopologicalOrientation};
use super::super::definition_validation::pcurve_domain;
use super::super::{
    ExactPcurveDefinition, ExactPcurveImplementation, ExactTrimClassifier,
    ExactTrimClassifierImplementation, GeometryEvaluationControl, GeometryEvaluationError,
    GeometryEvaluationErrorKind, ParameterRange, TrimClassifierId, TrimDomainLocation,
};
use super::integration::adaptive_scalar_integral;
use super::pcurve::evaluate_pcurve;
use super::projection::{
    charge_seed_allocation, project_parametric, uniform_seeds, ParametricDerivatives,
};
use super::{invalid_result, kernel_owned, PortableExactEvaluator};

const WINDING_ABSOLUTE_ERROR_RAD: f64 = 1.0e-10;
const MAX_PCURVE_PROJECTION_SEEDS: usize = 1_000_000;

impl ExactTrimClassifier for PortableExactEvaluator<'_> {
    fn classify(
        &self,
        id: &TrimClassifierId,
        uv: [f64; 2],
        boundary_tolerance_uv: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<TrimDomainLocation, GeometryEvaluationError> {
        if uv.iter().any(|value| !value.is_finite())
            || !boundary_tolerance_uv.is_finite()
            || boundary_tolerance_uv <= 0.0
        {
            return Err(invalid_result(
                "trim query and boundary tolerance must be finite and valid",
            ));
        }
        control.checkpoint()?;
        let record = self.trim_classifier_record(id)?;
        if matches!(
            record.implementation,
            ExactTrimClassifierImplementation::Kernel { .. }
        ) {
            return Err(kernel_owned("trim classifier"));
        }
        let face = self
            .topology
            .faces
            .iter()
            .find(|face| &face.trim_classifier_id == id)
            .ok_or_else(|| invalid_result("trim classifier has no admitted face owner"))?;

        let outer = self.classify_wire(&face.outer_wire_id, uv, boundary_tolerance_uv, control)?;
        if outer == WireLocation::Boundary {
            return Ok(TrimDomainLocation::OnBoundary);
        }
        if outer == WireLocation::Outside {
            return Ok(TrimDomainLocation::Outside);
        }
        for wire_id in &face.inner_wire_ids {
            match self.classify_wire(wire_id, uv, boundary_tolerance_uv, control)? {
                WireLocation::Boundary => return Ok(TrimDomainLocation::OnBoundary),
                WireLocation::Inside => return Ok(TrimDomainLocation::Outside),
                WireLocation::Outside => {}
            }
        }
        Ok(TrimDomainLocation::Inside)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum WireLocation {
    Inside,
    Boundary,
    Outside,
}

impl PortableExactEvaluator<'_> {
    fn classify_wire(
        &self,
        wire_id: &PersistentEntityId,
        uv: [f64; 2],
        boundary_tolerance_uv: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<WireLocation, GeometryEvaluationError> {
        let wire = find_by_id(&self.topology.wires, wire_id, |wire| &wire.id, "wire")?;
        let integration_error = WINDING_ABSOLUTE_ERROR_RAD / wire.coedge_ids.len() as f64;
        let mut winding_radians = 0.0;
        for coedge_id in &wire.coedge_ids {
            control.checkpoint()?;
            let coedge = find_by_id(
                &self.topology.coedges,
                coedge_id,
                |coedge| &coedge.id,
                "coedge",
            )?;
            let record = self.pcurve_record(&coedge.pcurve_evaluator_id)?;
            let ExactPcurveImplementation::Portable { definition } = &record.implementation else {
                return Err(kernel_owned("pcurve"));
            };
            let range =
                pcurve_domain(&record.implementation).ok_or_else(|| kernel_owned("pcurve"))?;
            if self.pcurve_distance(definition, range, uv, boundary_tolerance_uv, control)?
                <= boundary_tolerance_uv
            {
                return Ok(WireLocation::Boundary);
            }
            let direction =
                orientation_sign(wire.orientation) * orientation_sign(coedge.orientation);
            winding_radians += direction
                * adaptive_scalar_integral(
                    range,
                    integration_error,
                    control,
                    "trim winding integration",
                    |parameter| {
                        let value = evaluate_pcurve(definition, parameter, control)?;
                        let offset = [value.point_uv[0] - uv[0], value.point_uv[1] - uv[1]];
                        let denominator = offset[0] * offset[0] + offset[1] * offset[1];
                        if !denominator.is_finite() || denominator <= 0.0 {
                            return Err(invalid_result(
                                "trim winding encountered an unresolved boundary singularity",
                            ));
                        }
                        Ok(
                            (offset[0] * value.first_uv[1] - offset[1] * value.first_uv[0])
                                / denominator,
                        )
                    },
                )?;
        }
        let turns = winding_radians / std::f64::consts::TAU;
        let nearest = turns.round();
        if !turns.is_finite() || (turns - nearest).abs() > WINDING_ABSOLUTE_ERROR_RAD * 4.0 {
            return Err(GeometryEvaluationError::new(
                GeometryEvaluationErrorKind::ClassificationDidNotConverge,
                "trim winding did not converge to an integer",
            ));
        }
        if nearest.abs() > 1.0 {
            return Err(invalid_result("trim wire has a non-simple winding number"));
        }
        Ok(if nearest == 0.0 {
            WireLocation::Outside
        } else {
            WireLocation::Inside
        })
    }

    fn pcurve_distance(
        &self,
        definition: &ExactPcurveDefinition,
        range: ParameterRange,
        uv: [f64; 2],
        tolerance: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<f64, GeometryEvaluationError> {
        let seeds = pcurve_projection_seeds(definition, range)?;
        charge_seed_allocation(seeds.len(), std::mem::size_of::<f64>(), control)?;
        let projection = project_parametric(range, uv, tolerance, seeds, control, |parameter| {
            let value = evaluate_pcurve(definition, parameter, control)?;
            Ok(ParametricDerivatives {
                point: value.point_uv,
                first: value.first_uv,
                second: value.second_uv,
            })
        })?;
        Ok(projection.distance)
    }
}

fn pcurve_projection_seeds(
    definition: &ExactPcurveDefinition,
    range: ParameterRange,
) -> Result<Vec<f64>, GeometryEvaluationError> {
    let ExactPcurveDefinition::Nurbs { definition } = definition else {
        return Ok(uniform_seeds(range, 64));
    };
    let subdivisions = usize::from(definition.degree).saturating_mul(8).max(8);
    let mut seeds = Vec::new();
    for knots in definition.knots.windows(2) {
        let start = knots[0].max(range.start);
        let end = knots[1].min(range.end);
        if start >= end {
            continue;
        }
        for index in 0..=subdivisions {
            if seeds.len() >= MAX_PCURVE_PROJECTION_SEEDS {
                return Err(GeometryEvaluationError::new(
                    GeometryEvaluationErrorKind::BudgetExceeded,
                    "pcurve projection seed count exceeds its hard bound",
                ));
            }
            seeds.push(start + (end - start) * index as f64 / subdivisions as f64);
        }
    }
    Ok(seeds)
}

fn orientation_sign(orientation: TopologicalOrientation) -> f64 {
    match orientation {
        TopologicalOrientation::Forward => 1.0,
        TopologicalOrientation::Reversed => -1.0,
    }
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
        .map_err(|_| invalid_result(format!("admitted {kind} index is incomplete")))
}
