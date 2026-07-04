use super::*;
use crate::refinement::RefinementIndicatorMode;

mod convergence;
mod default_indicators;
mod fixtures;
mod indicator_plan;

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
