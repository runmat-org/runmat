use super::*;
use crate::options::{MeshRefinementOptions, RefinementIndicatorOverrides};

fn key(namespace: &str, name: &str) -> RefinementIndicatorKey {
    RefinementIndicatorKey::new(namespace, name)
}

fn available(namespace: &str, name: &str) -> RefinementIndicatorAvailability {
    RefinementIndicatorAvailability {
        key: key(namespace, name),
        applicable: true,
        field_available: true,
    }
}

#[test]
fn adaptive_convergence_is_disabled_for_nonadaptive_strategies() {
    let mut options = MeshRefinementOptions {
        strategy: RefinementStrategy::None,
        ..MeshRefinementOptions::default()
    };

    assert_eq!(
        evaluate_adaptive_convergence(
            &options,
            AdaptiveConvergenceMetrics {
                field_change: Some(0.0),
                ..AdaptiveConvergenceMetrics::default()
            }
        ),
        AdaptiveConvergenceStatus::Disabled
    );

    options.strategy = RefinementStrategy::Uniform;
    assert_eq!(
        evaluate_adaptive_convergence(
            &options,
            AdaptiveConvergenceMetrics {
                field_change: Some(0.0),
                ..AdaptiveConvergenceMetrics::default()
            }
        ),
        AdaptiveConvergenceStatus::Disabled
    );
}

#[test]
fn adaptive_convergence_reports_hard_stop_statuses_first() {
    let options = MeshRefinementOptions {
        strategy: RefinementStrategy::Adaptive,
        max_iterations: 2,
        ..MeshRefinementOptions::default()
    };

    assert_eq!(
        evaluate_adaptive_convergence(
            &options,
            AdaptiveConvergenceMetrics {
                element_budget_reached: true,
                completed_iterations: 2,
                field_change: Some(0.0),
                ..AdaptiveConvergenceMetrics::default()
            }
        ),
        AdaptiveConvergenceStatus::ElementBudgetReached
    );
    assert_eq!(
        evaluate_adaptive_convergence(
            &options,
            AdaptiveConvergenceMetrics {
                completed_iterations: 2,
                field_change: Some(0.0),
                ..AdaptiveConvergenceMetrics::default()
            }
        ),
        AdaptiveConvergenceStatus::MaxIterationsReached
    );
}

#[test]
fn adaptive_convergence_requires_finite_metric_within_tolerance() {
    let options = MeshRefinementOptions {
        strategy: RefinementStrategy::Adaptive,
        ..MeshRefinementOptions::default()
    };

    assert_eq!(
        evaluate_adaptive_convergence(
            &options,
            AdaptiveConvergenceMetrics {
                field_change: Some(0.05),
                energy_change: Some(0.02),
                ..AdaptiveConvergenceMetrics::default()
            }
        ),
        AdaptiveConvergenceStatus::Converged
    );
    assert_eq!(
        evaluate_adaptive_convergence(
            &options,
            AdaptiveConvergenceMetrics {
                field_change: Some(0.051),
                energy_change: Some(0.02),
                ..AdaptiveConvergenceMetrics::default()
            }
        ),
        AdaptiveConvergenceStatus::Pending
    );
    assert_eq!(
        evaluate_adaptive_convergence(
            &options,
            AdaptiveConvergenceMetrics {
                field_change: Some(f64::NAN),
                ..AdaptiveConvergenceMetrics::default()
            }
        ),
        AdaptiveConvergenceStatus::Pending
    );
    assert_eq!(
        evaluate_adaptive_convergence(&options, AdaptiveConvergenceMetrics::default()),
        AdaptiveConvergenceStatus::Pending
    );
}

#[test]
fn adaptive_convergence_detects_no_topology_growth() {
    let options = MeshRefinementOptions {
        strategy: RefinementStrategy::Adaptive,
        ..MeshRefinementOptions::default()
    };

    assert_eq!(
        evaluate_adaptive_convergence(
            &options,
            AdaptiveConvergenceMetrics {
                completed_iterations: 1,
                previous_node_count: Some(24),
                current_node_count: Some(24),
                previous_element_count: Some(48),
                current_element_count: Some(48),
                ..AdaptiveConvergenceMetrics::default()
            }
        ),
        AdaptiveConvergenceStatus::Converged
    );
    assert_eq!(
        evaluate_adaptive_convergence(
            &options,
            AdaptiveConvergenceMetrics {
                completed_iterations: 1,
                previous_node_count: Some(24),
                current_node_count: Some(25),
                previous_element_count: Some(48),
                current_element_count: Some(49),
                ..AdaptiveConvergenceMetrics::default()
            }
        ),
        AdaptiveConvergenceStatus::Pending
    );
}

#[test]
fn adaptive_convergence_residual_metric_is_opt_in() {
    let mut options = MeshRefinementOptions {
        strategy: RefinementStrategy::Adaptive,
        ..MeshRefinementOptions::default()
    };

    assert_eq!(
        evaluate_adaptive_convergence(
            &options,
            AdaptiveConvergenceMetrics {
                residual: Some(1.0e6),
                field_change: Some(0.0),
                ..AdaptiveConvergenceMetrics::default()
            }
        ),
        AdaptiveConvergenceStatus::Converged
    );

    options.convergence.residual_tolerance = Some(1.0e-6);
    assert_eq!(
        evaluate_adaptive_convergence(
            &options,
            AdaptiveConvergenceMetrics {
                residual: Some(1.0e-7),
                ..AdaptiveConvergenceMetrics::default()
            }
        ),
        AdaptiveConvergenceStatus::Converged
    );
    assert_eq!(
        evaluate_adaptive_convergence(
            &options,
            AdaptiveConvergenceMetrics {
                residual: Some(1.0e-5),
                ..AdaptiveConvergenceMetrics::default()
            }
        ),
        AdaptiveConvergenceStatus::Pending
    );
}

#[test]
fn sizing_field_update_merges_bounds_and_samples() {
    let mut sizing = MeshSizingField {
        global_target_size_m: Some(0.1),
        min_size_m: Some(0.02),
        max_size_m: Some(0.2),
        samples: vec![SizingSample {
            position_m: [0.0, 0.0, 0.0],
            target_size_m: 0.08,
            reason: Some("initial".to_string()),
        }],
        ..MeshSizingField::default()
    };

    SizingFieldUpdate {
        samples: vec![SizingSample {
            position_m: [1.0, 0.0, 0.0],
            target_size_m: 0.01,
            reason: Some("stress_gradient".to_string()),
        }],
        min_size_m: Some(0.01),
        max_size_m: Some(0.25),
    }
    .apply_to(&mut sizing);

    assert_eq!(sizing.global_target_size_m, Some(0.1));
    assert_eq!(sizing.min_size_m, Some(0.01));
    assert_eq!(sizing.max_size_m, Some(0.25));
    assert_eq!(sizing.samples.len(), 2);
}

#[test]
fn adaptive_iteration_summary_round_trips() {
    let summary = AdaptiveIterationSummary {
        iteration_index: 1,
        node_count: 32,
        element_count: 96,
        convergence_status: AdaptiveConvergenceStatus::Pending,
        indicators: vec![RefinementIndicatorSummary {
            namespace: "structural".to_string(),
            name: "stress_gradient".to_string(),
            requested_mode: RefinementIndicatorMode::Auto,
            status: RefinementIndicatorStatus::Used,
            detail: Some("field available".to_string()),
        }],
        markers: vec![RefinementMarker {
            entity_id: "tetrahedron_1".to_string(),
            weight: 1.0,
            reason: "stress_gradient".to_string(),
        }],
        sizing_update: SizingFieldUpdate::default(),
    };

    let encoded = serde_json::to_string(&summary).expect("summary should serialize");
    let decoded: AdaptiveIterationSummary =
        serde_json::from_str(&encoded).expect("summary should deserialize");

    assert_eq!(decoded, summary);
}

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

#[test]
fn structural_static_defaults_are_owned_by_adaptive_policy() {
    let defaults = structural_static_default_refinement_indicators();

    assert_eq!(
        defaults,
        vec![
            key("structural", "stress_gradient"),
            key("structural", "strain_energy_density"),
            key("structural", "load_regions"),
            key("structural", "constraint_regions"),
        ]
    );
}

#[test]
fn analysis_default_indicators_cover_supported_run_kinds() {
    let cases = [
        (
            "linear_static_structural",
            "linear_static",
            key("structural", "stress_gradient"),
        ),
        (
            "modal_structural",
            "modal",
            key("modal", "mode_shape_curvature"),
        ),
        (
            "transient_structural",
            "transient",
            key("structural", "displacement_gradient"),
        ),
        (
            "nonlinear_structural",
            "nonlinear",
            key("structural", "plastic_strain"),
        ),
        (
            "thermal_standalone",
            "thermal",
            key("thermal", "temperature_gradient"),
        ),
        (
            "electromagnetic_static",
            "electromagnetic",
            key("electromagnetic", "flux_density_gradient"),
        ),
        (
            "acoustic_harmonic",
            "acoustic",
            key("acoustic", "wavelength"),
        ),
        ("cfd_steady_state", "cfd", key("cfd", "boundary_layer")),
        (
            "cht_coupled",
            "cht",
            key("cht", "interface_temperature_jump"),
        ),
        ("fsi_coupled", "fsi", key("fsi", "interface_traction_jump")),
    ];

    for (profile, run_kind, expected) in cases {
        let defaults = default_refinement_indicators_for_analysis(profile, run_kind);
        assert!(
            defaults.contains(&expected),
            "{profile}/{run_kind} should include {expected:?}; got {defaults:?}"
        );
    }
}

#[test]
fn analysis_default_indicators_prefer_profile_over_run_kind() {
    let defaults =
        default_refinement_indicators_for_analysis("thermo_mechanical_coupled", "transient");

    assert!(defaults.contains(&key("thermo_mechanical", "thermal_stress")));
    assert!(!defaults.contains(&key("structural", "displacement_gradient")));
}

#[test]
fn analysis_default_indicators_fall_back_to_run_kind() {
    let defaults = default_refinement_indicators_for_analysis("custom_profile", "thermal");

    assert!(defaults.contains(&key("thermal", "temperature_gradient")));
    assert!(defaults.contains(&key("thermal", "heat_flux_gradient")));
}

#[test]
fn refinement_markers_are_deterministic_and_create_sizing_samples() {
    let samples = vec![
        RefinementIndicatorSample {
            entity_id: "tetrahedron_low".to_string(),
            position_m: [0.0, 0.0, 0.0],
            indicator_value: 0.2,
            current_size_m: 0.08,
        },
        RefinementIndicatorSample {
            entity_id: "tetrahedron_b".to_string(),
            position_m: [1.0, 0.0, 0.0],
            indicator_value: 1.0,
            current_size_m: 0.06,
        },
        RefinementIndicatorSample {
            entity_id: "tetrahedron_a".to_string(),
            position_m: [0.0, 1.0, 0.0],
            indicator_value: 1.0,
            current_size_m: 0.04,
        },
        RefinementIndicatorSample {
            entity_id: "tetrahedron_mid".to_string(),
            position_m: [0.0, 0.0, 1.0],
            indicator_value: 0.5,
            current_size_m: 0.1,
        },
    ];

    let (markers, sizing) = build_refinement_markers_from_samples(
        &samples,
        "structural.stress_gradient",
        RefinementMarkerOptions {
            max_markers: 3,
            min_relative_value: 0.5,
            target_size_scale: 0.4,
        },
    )
    .expect("marker generation should succeed");

    assert_eq!(
        markers
            .iter()
            .map(|marker| marker.entity_id.as_str())
            .collect::<Vec<_>>(),
        vec!["tetrahedron_a", "tetrahedron_b", "tetrahedron_mid"]
    );
    assert_eq!(markers[0].weight, 1.0);
    assert_eq!(markers[2].weight, 0.5);
    assert_eq!(sizing.samples.len(), 3);
    assert_eq!(sizing.samples[0].target_size_m, 0.016);
    assert_eq!(
        sizing.samples[0].reason.as_deref(),
        Some("structural.stress_gradient")
    );
}

#[test]
fn refinement_markers_filter_nonfinite_and_empty_samples() {
    let samples = vec![
        RefinementIndicatorSample {
            entity_id: "nan".to_string(),
            position_m: [0.0, 0.0, 0.0],
            indicator_value: f64::NAN,
            current_size_m: 0.08,
        },
        RefinementIndicatorSample {
            entity_id: "zero".to_string(),
            position_m: [0.0, 0.0, 0.0],
            indicator_value: 0.0,
            current_size_m: 0.08,
        },
        RefinementIndicatorSample {
            entity_id: "bad_size".to_string(),
            position_m: [0.0, 0.0, 0.0],
            indicator_value: 1.0,
            current_size_m: 0.0,
        },
    ];

    let (markers, sizing) = build_refinement_markers_from_samples(
        &samples,
        "structural.stress_gradient",
        RefinementMarkerOptions::default(),
    )
    .expect("invalid samples should be filtered, not fail the batch");

    assert!(markers.is_empty());
    assert!(sizing.samples.is_empty());
}

#[test]
fn refinement_marker_options_are_validated() {
    let sample = [RefinementIndicatorSample {
        entity_id: "tetrahedron".to_string(),
        position_m: [0.0, 0.0, 0.0],
        indicator_value: 1.0,
        current_size_m: 0.1,
    }];

    assert_eq!(
        build_refinement_markers_from_samples(
            &sample,
            "reason",
            RefinementMarkerOptions {
                max_markers: 0,
                ..RefinementMarkerOptions::default()
            }
        ),
        Err(RefinementMarkerError::InvalidMaxMarkers)
    );
    assert_eq!(
        build_refinement_markers_from_samples(
            &sample,
            "reason",
            RefinementMarkerOptions {
                min_relative_value: 1.5,
                ..RefinementMarkerOptions::default()
            }
        ),
        Err(RefinementMarkerError::InvalidMinRelativeValue)
    );
    assert_eq!(
        build_refinement_markers_from_samples(
            &sample,
            "reason",
            RefinementMarkerOptions {
                target_size_scale: 1.0,
                ..RefinementMarkerOptions::default()
            }
        ),
        Err(RefinementMarkerError::InvalidTargetSizeScale)
    );
}
