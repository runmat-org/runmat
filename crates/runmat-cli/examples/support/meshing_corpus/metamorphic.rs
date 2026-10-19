use super::contract::{MeshScenarioReference, MetamorphicRelation};
use super::report::{CaseReport, MeshQualityReport, MeshReport, MetamorphicReport};

pub fn evaluate(relations: &[MetamorphicRelation], cases: &[CaseReport]) -> Vec<MetamorphicReport> {
    relations
        .iter()
        .map(|relation| evaluate_one(relation, cases))
        .collect()
}

fn evaluate_one(relation: &MetamorphicRelation, cases: &[CaseReport]) -> MetamorphicReport {
    let compared_fixture_ids = relation
        .comparisons
        .iter()
        .map(|reference| reference.fixture_id.clone())
        .collect();
    let detail = match scenario(cases, &relation.reference).and_then(successful_observation) {
        Err(error) => Some(error),
        Ok(baseline) => relation.comparisons.iter().find_map(|comparison| {
            let observed = scenario(cases, comparison)
                .and_then(successful_observation)
                .map_err(|error| {
                    format!(
                        "{}/{}: {error}",
                        comparison.fixture_id, comparison.scenario_id
                    )
                });
            match observed {
                Ok(observed) => compare(relation, baseline, observed).err().map(|error| {
                    format!(
                        "{}/{}: {error}",
                        comparison.fixture_id, comparison.scenario_id
                    )
                }),
                Err(error) => Some(error),
            }
        }),
    };

    MetamorphicReport {
        id: relation.id.clone(),
        reference_fixture_id: relation.reference.fixture_id.clone(),
        reference_scenario_id: relation.reference.scenario_id.clone(),
        compared_fixture_ids,
        passed: detail.is_none(),
        detail,
    }
}

fn scenario<'a>(
    cases: &'a [CaseReport],
    reference: &MeshScenarioReference,
) -> Result<&'a MeshReport, String> {
    cases
        .iter()
        .find(|case| case.fixture_id == reference.fixture_id)
        .and_then(|case| {
            case.mesh_scenarios
                .iter()
                .find(|scenario| scenario.id == reference.scenario_id)
        })
        .ok_or_else(|| {
            format!(
                "required scenario {}/{} was not executed",
                reference.fixture_id, reference.scenario_id
            )
        })
}

fn successful_observation(report: &MeshReport) -> Result<&MeshReport, String> {
    if report.passed && report.outcome == "success" && report.quality.is_some() {
        Ok(report)
    } else {
        Err("required successful observation is unavailable".into())
    }
}

fn compare(
    relation: &MetamorphicRelation,
    baseline: &MeshReport,
    observed: &MeshReport,
) -> Result<(), String> {
    if relation.require_equal_entity_counts
        && (baseline.node_count != observed.node_count
            || baseline.element_count != observed.element_count
            || baseline.boundary_face_count != observed.boundary_face_count
            || baseline.region_count != observed.region_count)
    {
        return Err(format!(
            "entity counts changed from nodes/elements/faces/regions={:?}/{:?}/{:?}/{:?} to {:?}/{:?}/{:?}/{:?}",
            baseline.node_count,
            baseline.element_count,
            baseline.boundary_face_count,
            baseline.region_count,
            observed.node_count,
            observed.element_count,
            observed.boundary_face_count,
            observed.region_count
        ));
    }
    let baseline_quality = baseline.quality.as_ref().expect("checked above");
    let observed_quality = observed.quality.as_ref().expect("checked above");
    compare_quality(relation, baseline_quality, observed_quality)
}

fn compare_quality(
    relation: &MetamorphicRelation,
    baseline: &MeshQualityReport,
    observed: &MeshQualityReport,
) -> Result<(), String> {
    if !within_relative_tolerance(
        baseline.total_volume_m3,
        observed.total_volume_m3,
        relation.volume_relative_tolerance,
    ) {
        return Err(format!(
            "total volume changed from {} m^3 to {} m^3 (relative tolerance {})",
            baseline.total_volume_m3, observed.total_volume_m3, relation.volume_relative_tolerance
        ));
    }
    for (name, baseline_value, observed_value) in [
        (
            "minimum scaled Jacobian",
            baseline.minimum_scaled_jacobian,
            observed.minimum_scaled_jacobian,
        ),
        (
            "maximum edge aspect ratio",
            baseline.maximum_edge_aspect_ratio,
            observed.maximum_edge_aspect_ratio,
        ),
        (
            "maximum radius-edge ratio",
            baseline.maximum_radius_edge_ratio,
            observed.maximum_radius_edge_ratio,
        ),
    ] {
        if !within_relative_tolerance(
            baseline_value,
            observed_value,
            relation.quality_relative_tolerance,
        ) {
            return Err(format!(
                "{name} changed from {baseline_value} to {observed_value} (relative tolerance {})",
                relation.quality_relative_tolerance
            ));
        }
    }
    Ok(())
}

fn within_relative_tolerance(left: f64, right: f64, tolerance: f64) -> bool {
    (left - right).abs() <= tolerance * left.abs().max(right.abs()).max(f64::MIN_POSITIVE)
}
