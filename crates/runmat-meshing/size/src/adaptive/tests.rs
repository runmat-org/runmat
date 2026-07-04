use super::*;
use crate::refinement::RefinementIndicatorMode;

mod convergence;
mod default_indicators;
mod fixtures;
mod indicator_plan;
mod markers;

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
