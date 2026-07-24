use std::collections::BTreeMap;

use super::fixtures::{available, key};
use crate::{
    adaptive::{
        plan_refinement_indicators, RefinementIndicatorAvailability, RefinementIndicatorStatus,
    },
    refinement::{
        MeshRefinementOptions, RefinementIndicatorMode, RefinementIndicatorOverrides,
        RefinementStrategy,
    },
};

#[test]
fn refinement_indicator_plan_merges_defaults_and_overrides() {
    let options = MeshRefinementOptions {
        indicators: RefinementIndicatorOverrides {
            namespaces: BTreeMap::from([(
                "structural".to_string(),
                BTreeMap::from([
                    ("stress_gradient".to_string(), RefinementIndicatorMode::Off),
                    (
                        "strain_energy_density".to_string(),
                        RefinementIndicatorMode::On,
                    ),
                    ("plastic_strain".to_string(), RefinementIndicatorMode::On),
                ]),
            )]),
        },
        ..MeshRefinementOptions::default()
    };

    let summaries = plan_refinement_indicators(
        &options,
        &[key("structural", "stress_gradient")],
        &[
            available("structural", "stress_gradient"),
            available("structural", "strain_energy_density"),
        ],
        false,
        false,
    );

    assert_eq!(summaries.len(), 3);
    assert_eq!(
        summaries
            .iter()
            .find(|summary| summary.name == "stress_gradient")
            .expect("stress gradient summary")
            .status,
        RefinementIndicatorStatus::SkippedNotApplicable
    );
    assert_eq!(
        summaries
            .iter()
            .find(|summary| summary.name == "strain_energy_density")
            .expect("strain energy summary")
            .status,
        RefinementIndicatorStatus::Used
    );
    assert_eq!(
        summaries
            .iter()
            .find(|summary| summary.name == "plastic_strain")
            .expect("plastic strain summary")
            .status,
        RefinementIndicatorStatus::SkippedMissingField
    );
}

#[test]
fn refinement_indicator_plan_reports_budget_and_quality_skips() {
    let options = MeshRefinementOptions::default();
    let defaults = [key("structural", "stress_gradient")];
    let availability = [available("structural", "stress_gradient")];

    let budget = plan_refinement_indicators(&options, &defaults, &availability, true, false);
    assert_eq!(budget[0].status, RefinementIndicatorStatus::SkippedBudget);

    let quality = plan_refinement_indicators(&options, &defaults, &availability, false, true);
    assert_eq!(quality[0].status, RefinementIndicatorStatus::SkippedQuality);
}

#[test]
fn refinement_indicator_plan_distinguishes_missing_and_not_applicable() {
    let options = MeshRefinementOptions::default();
    let defaults = [
        key("structural", "stress_gradient"),
        key("structural", "contact_pressure"),
    ];
    let availability = [
        RefinementIndicatorAvailability {
            key: key("structural", "stress_gradient"),
            applicable: true,
            field_available: false,
        },
        RefinementIndicatorAvailability {
            key: key("structural", "contact_pressure"),
            applicable: false,
            field_available: true,
        },
    ];

    let summaries = plan_refinement_indicators(&options, &defaults, &availability, false, false);

    assert_eq!(
        summaries
            .iter()
            .find(|summary| summary.name == "stress_gradient")
            .expect("stress gradient summary")
            .status,
        RefinementIndicatorStatus::SkippedMissingField
    );
    assert_eq!(
        summaries
            .iter()
            .find(|summary| summary.name == "contact_pressure")
            .expect("contact pressure summary")
            .status,
        RefinementIndicatorStatus::SkippedNotApplicable
    );
}

#[test]
fn refinement_indicator_plan_is_empty_when_refinement_is_disabled() {
    let options = MeshRefinementOptions {
        strategy: RefinementStrategy::None,
        ..MeshRefinementOptions::default()
    };

    assert!(plan_refinement_indicators(
        &options,
        &[key("structural", "stress_gradient")],
        &[available("structural", "stress_gradient")],
        false,
        false
    )
    .is_empty());
}

#[test]
fn refinement_indicator_plan_is_empty_for_uniform_refinement() {
    let options = MeshRefinementOptions {
        strategy: RefinementStrategy::Uniform,
        indicators: RefinementIndicatorOverrides {
            namespaces: BTreeMap::from([(
                "structural".to_string(),
                BTreeMap::from([("stress_gradient".to_string(), RefinementIndicatorMode::On)]),
            )]),
        },
        ..MeshRefinementOptions::default()
    };

    assert!(plan_refinement_indicators(
        &options,
        &[key("structural", "strain_energy_density")],
        &[available("structural", "stress_gradient")],
        false,
        false,
    )
    .is_empty());
}
