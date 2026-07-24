use super::*;

#[test]
fn classifies_slivers_by_aspect_ratio_and_quality() {
    let tetrahedra = vec![
        quality(1, 8.0, 0.6),
        quality(2, 25.0, 0.5),
        quality(3, 30.0, 0.05),
    ];

    let classifications =
        classify_sliver_tetrahedra(&tetrahedra, options()).expect("classification should succeed");

    assert_eq!(
        classifications
            .iter()
            .map(|classification| (classification.tetrahedron_id, classification.reason))
            .collect::<Vec<_>>(),
        vec![
            (3, SliverClassificationReason::AspectRatioAndExactQuality),
            (2, SliverClassificationReason::AspectRatioOnly),
        ]
    );
}

#[test]
fn accepts_targeted_sliver_removal_that_preserves_exact_quality() {
    let current = vec![quality(1, 30.0, 0.42), quality(2, 10.0, 0.50)];
    let proposed = vec![quality(1, 12.0, 0.43), quality(2, 10.0, 0.50)];

    let evaluation = evaluate_sliver_removal(&current, &proposed, options())
        .expect("removal evaluation should succeed");

    assert!(evaluation.accepted);
    assert_eq!(evaluation.initial_sliver_count, 1);
    assert_eq!(evaluation.final_sliver_count, 0);
    assert_eq!(evaluation.removed_sliver_count, 1);
    assert_eq!(evaluation.rejection_reason, None);
    assert_eq!(evaluation.initial_max_aspect_ratio, 30.0);
    assert_eq!(evaluation.final_max_aspect_ratio, 12.0);
}

#[test]
fn rejects_removal_when_sliver_count_does_not_decrease() {
    let current = vec![quality(1, 30.0, 0.42)];
    let proposed = vec![quality(1, 21.0, 0.43)];

    let evaluation = evaluate_sliver_removal(&current, &proposed, options())
        .expect("removal evaluation should succeed");

    assert!(!evaluation.accepted);
    assert_eq!(
        evaluation.rejection_reason,
        Some(SliverRemovalRejectionReason::SliverCountNotReduced)
    );
}

#[test]
fn rejects_removal_when_exact_quality_violation_regresses() {
    let current = vec![quality(1, 30.0, 0.42)];
    let proposed = vec![quality(1, 10.0, 0.05)];

    let evaluation = evaluate_sliver_removal(&current, &proposed, options())
        .expect("removal evaluation should succeed");

    assert!(!evaluation.accepted);
    assert_eq!(
        evaluation.rejection_reason,
        Some(SliverRemovalRejectionReason::ExactQualityViolationRegressed)
    );
}

#[test]
fn rejects_removal_when_minimum_exact_quality_regresses() {
    let current = vec![quality(1, 30.0, 0.42), quality(2, 10.0, 0.30)];
    let proposed = vec![quality(1, 10.0, 0.41), quality(2, 10.0, 0.29)];

    let evaluation = evaluate_sliver_removal(&current, &proposed, options())
        .expect("removal evaluation should succeed");

    assert!(!evaluation.accepted);
    assert_eq!(
        evaluation.rejection_reason,
        Some(SliverRemovalRejectionReason::MinimumExactQualityRegressed)
    );
}

#[test]
fn rejects_non_finite_quality_inputs() {
    let err = classify_sliver_tetrahedra(&[quality(7, f64::INFINITY, 0.4)], options())
        .expect_err("non-finite aspect ratio should fail");

    assert_eq!(
        err,
        SliverRecoveryError::NonFiniteQuality { tetrahedron_id: 7 }
    );
}

fn quality(
    tetrahedron_id: u32,
    aspect_ratio: f64,
    exact_scaled_jacobian: f64,
) -> SliverTetrahedronQuality {
    SliverTetrahedronQuality {
        tetrahedron_id,
        aspect_ratio,
        exact_scaled_jacobian,
    }
}

fn options() -> SliverRecoveryOptions {
    SliverRecoveryOptions {
        sliver_aspect_ratio: 20.0,
        min_exact_scaled_jacobian: 0.15,
        exact_quality_tolerance: 1.0e-12,
    }
}
