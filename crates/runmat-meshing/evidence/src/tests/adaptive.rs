use super::*;

#[test]
fn evidence_summarizes_adaptive_iterations_without_raw_marker_details() {
    let mut mesh = minimal_evidence_mesh();
    mesh.adaptive_iterations = vec![
        AdaptiveIterationSummary {
            iteration_index: 0,
            node_count: 4,
            element_count: 1,
            convergence_status: AdaptiveConvergenceStatus::Pending,
            indicators: vec![RefinementIndicatorSummary {
                namespace: "structural".to_string(),
                name: "load_regions".to_string(),
                requested_mode: RefinementIndicatorMode::Auto,
                status: RefinementIndicatorStatus::Used,
                detail: Some("field available".to_string()),
            }],
            markers: vec![RefinementMarker {
                entity_id: "face_1".to_string(),
                weight: 1.0,
                reason: "structural.load_regions".to_string(),
            }],
            sizing_update: SizingFieldUpdate {
                samples: vec![SizingSample {
                    position_m: [0.0, 0.0, 1.0],
                    target_size_m: 0.25,
                    reason: Some("structural.load_regions".to_string()),
                }],
                min_size_m: None,
                max_size_m: None,
            },
        },
        AdaptiveIterationSummary {
            iteration_index: 1,
            node_count: 5,
            element_count: 2,
            convergence_status: AdaptiveConvergenceStatus::Converged,
            indicators: vec![
                RefinementIndicatorSummary {
                    namespace: "structural".to_string(),
                    name: "stress_gradient".to_string(),
                    requested_mode: RefinementIndicatorMode::Auto,
                    status: RefinementIndicatorStatus::Used,
                    detail: None,
                },
                RefinementIndicatorSummary {
                    namespace: "thermal".to_string(),
                    name: "temperature_gradient".to_string(),
                    requested_mode: RefinementIndicatorMode::Auto,
                    status: RefinementIndicatorStatus::SkippedMissingField,
                    detail: Some("required recovered field is unavailable".to_string()),
                },
            ],
            markers: vec![
                RefinementMarker {
                    entity_id: "tetrahedron_1".to_string(),
                    weight: 1.0,
                    reason: "structural.stress_gradient".to_string(),
                },
                RefinementMarker {
                    entity_id: "tetrahedron_2".to_string(),
                    weight: 0.5,
                    reason: "structural.stress_gradient".to_string(),
                },
            ],
            sizing_update: SizingFieldUpdate {
                samples: vec![
                    SizingSample {
                        position_m: [0.2, 0.2, 0.2],
                        target_size_m: 0.2,
                        reason: Some("structural.stress_gradient".to_string()),
                    },
                    SizingSample {
                        position_m: [0.4, 0.2, 0.2],
                        target_size_m: 0.2,
                        reason: Some("structural.stress_gradient".to_string()),
                    },
                ],
                min_size_m: None,
                max_size_m: None,
            },
        },
    ];

    let evidence = build_mesh_evidence_artifact(&mesh, &AnalysisMeshValidationOptions::default());

    assert_eq!(evidence.topology.adaptive_iteration_count, 2);
    assert_eq!(evidence.adaptive.iteration_count, 2);
    assert_eq!(evidence.adaptive.latest_iteration_index, Some(1));
    assert_eq!(
        evidence.adaptive.latest_convergence_status.as_deref(),
        Some("converged")
    );
    assert_eq!(evidence.adaptive.latest_indicator_count, 2);
    assert_eq!(evidence.adaptive.latest_used_indicator_count, 1);
    assert_eq!(evidence.adaptive.latest_marker_count, 2);
    assert_eq!(evidence.adaptive.latest_sizing_update_sample_count, 2);
    assert_eq!(evidence.adaptive.marker_count, 3);
    assert_eq!(evidence.adaptive.sizing_update_sample_count, 3);
    assert_eq!(
        evidence.adaptive.latest_indicator_status_counts.get("used"),
        Some(&1)
    );
    assert_eq!(
        evidence
            .adaptive
            .latest_indicator_status_counts
            .get("skipped_missing_field"),
        Some(&1)
    );
    assert_eq!(
        evidence
            .adaptive
            .latest_marker_by_reason
            .get("structural.stress_gradient"),
        Some(&2)
    );
    assert_eq!(
        evidence
            .adaptive
            .latest_sizing_update_by_reason
            .get("structural.stress_gradient"),
        Some(&2)
    );
    assert_eq!(
        evidence
            .adaptive
            .marker_by_reason
            .get("structural.load_regions"),
        Some(&1)
    );
    assert_eq!(
        evidence
            .adaptive
            .marker_by_reason
            .get("structural.stress_gradient"),
        Some(&2)
    );
    assert_eq!(
        evidence
            .adaptive
            .sizing_update_by_reason
            .get("structural.load_regions"),
        Some(&1)
    );
}
