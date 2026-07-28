use crate::quality::ElementQuality;

use super::*;

#[test]
fn accepts_boundary_candidate_that_reduces_exact_quality_violations() {
    let current = vec![quality("a", 0.10), quality("b", 0.30)];
    let proposed = vec![quality("a", 0.20), quality("b", 0.30)];

    let evaluation = evaluate_boundary_quality_candidate(
        &current,
        &proposed,
        BoundaryQualityCandidateConstraints::preserved(),
        options(),
    )
    .expect("candidate should evaluate");

    assert!(evaluation.accepted);
    assert_eq!(evaluation.initial_exact_quality_violation_count, 1);
    assert_eq!(evaluation.final_exact_quality_violation_count, 0);
    assert_eq!(evaluation.rejection_reason, None);
}

#[test]
fn accepts_boundary_candidate_that_improves_minimum_quality() {
    let current = vec![quality("a", 0.18), quality("b", 0.30)];
    let proposed = vec![quality("a", 0.24), quality("b", 0.30)];

    let evaluation = evaluate_boundary_quality_candidate(
        &current,
        &proposed,
        BoundaryQualityCandidateConstraints::preserved(),
        options(),
    )
    .expect("candidate should evaluate");

    assert!(evaluation.accepted);
    assert_eq!(evaluation.initial_exact_quality_violation_count, 0);
    assert_eq!(evaluation.final_exact_quality_violation_count, 0);
    assert_eq!(evaluation.initial_min_exact_scaled_jacobian, 0.18);
    assert_eq!(evaluation.final_min_exact_scaled_jacobian, 0.24);
}

#[test]
fn rejects_candidate_when_boundary_recovery_regresses() {
    let mut constraints = BoundaryQualityCandidateConstraints::preserved();
    constraints.boundary_recovery_preserved = false;

    let evaluation = evaluate_boundary_quality_candidate(
        &[quality("a", 0.10)],
        &[quality("a", 0.30)],
        constraints,
        options(),
    )
    .expect("candidate should evaluate");

    assert!(!evaluation.accepted);
    assert_eq!(
        evaluation.rejection_reason,
        Some(BoundaryQualityCandidateRejectionReason::BoundaryRecoveryRegressed)
    );
}

#[test]
fn rejects_candidate_when_target_volume_regresses() {
    let mut constraints = BoundaryQualityCandidateConstraints::preserved();
    constraints.target_volume_preserved = false;

    let evaluation = evaluate_boundary_quality_candidate(
        &[quality("a", 0.10)],
        &[quality("a", 0.30)],
        constraints,
        options(),
    )
    .expect("candidate should evaluate");

    assert!(!evaluation.accepted);
    assert_eq!(
        evaluation.rejection_reason,
        Some(BoundaryQualityCandidateRejectionReason::TargetVolumeRegressed)
    );
}

#[test]
fn rejects_candidate_when_source_provenance_regresses() {
    let mut constraints = BoundaryQualityCandidateConstraints::preserved();
    constraints.source_provenance_preserved = false;

    let evaluation = evaluate_boundary_quality_candidate(
        &[quality("a", 0.10)],
        &[quality("a", 0.30)],
        constraints,
        options(),
    )
    .expect("candidate should evaluate");

    assert!(!evaluation.accepted);
    assert_eq!(
        evaluation.rejection_reason,
        Some(BoundaryQualityCandidateRejectionReason::SourceProvenanceRegressed)
    );
}

#[test]
fn rejects_candidate_when_exact_quality_violations_regress() {
    let evaluation = evaluate_boundary_quality_candidate(
        &[quality("a", 0.20), quality("b", 0.30)],
        &[quality("a", 0.20), quality("b", 0.05)],
        BoundaryQualityCandidateConstraints::preserved(),
        options(),
    )
    .expect("candidate should evaluate");

    assert!(!evaluation.accepted);
    assert_eq!(
        evaluation.rejection_reason,
        Some(BoundaryQualityCandidateRejectionReason::ExactQualityViolationRegressed)
    );
}

#[test]
fn rejects_candidate_when_minimum_quality_regresses() {
    let evaluation = evaluate_boundary_quality_candidate(
        &[quality("a", 0.20), quality("b", 0.30)],
        &[quality("a", 0.19), quality("b", 0.30)],
        BoundaryQualityCandidateConstraints::preserved(),
        options(),
    )
    .expect("candidate should evaluate");

    assert!(!evaluation.accepted);
    assert_eq!(
        evaluation.rejection_reason,
        Some(BoundaryQualityCandidateRejectionReason::MinimumExactQualityRegressed)
    );
}

#[test]
fn accepts_above_threshold_minimum_regression_when_explicitly_allowed() {
    let evaluation = evaluate_boundary_quality_candidate(
        &[quality("a", 0.20), quality("b", 0.30)],
        &[quality("a", 0.19), quality("b", 0.30)],
        BoundaryQualityCandidateConstraints::preserved(),
        BoundaryQualityCandidateOptions {
            allow_min_exact_quality_regression_above_threshold: true,
            require_exact_quality_improvement: false,
            ..options()
        },
    )
    .expect("candidate should evaluate");

    assert!(evaluation.accepted);
    assert_eq!(evaluation.rejection_reason, None);
}

#[test]
fn rejects_candidate_when_quality_does_not_improve() {
    let evaluation = evaluate_boundary_quality_candidate(
        &[quality("a", 0.20), quality("b", 0.30)],
        &[quality("a", 0.20), quality("b", 0.30)],
        BoundaryQualityCandidateConstraints::preserved(),
        options(),
    )
    .expect("candidate should evaluate");

    assert!(!evaluation.accepted);
    assert_eq!(
        evaluation.rejection_reason,
        Some(BoundaryQualityCandidateRejectionReason::ExactQualityNotImproved)
    );
}

#[test]
fn accepts_no_regression_candidate_when_improvement_is_not_required() {
    let mut options = options();
    options.require_exact_quality_improvement = false;

    let evaluation = evaluate_boundary_quality_candidate(
        &[quality("a", 0.20), quality("b", 0.30)],
        &[quality("a", 0.20), quality("b", 0.30)],
        BoundaryQualityCandidateConstraints::preserved(),
        options,
    )
    .expect("candidate should evaluate");

    assert!(evaluation.accepted);
    assert_eq!(evaluation.rejection_reason, None);
}

#[test]
fn rejects_non_finite_quality_inputs() {
    let err = evaluate_boundary_quality_candidate(
        &[quality("a", f64::NAN)],
        &[quality("a", 0.30)],
        BoundaryQualityCandidateConstraints::preserved(),
        options(),
    )
    .expect_err("non-finite exact quality should fail");

    assert_eq!(
        err,
        BoundaryQualityCandidateError::NonFiniteQuality {
            element_id: "a".to_string()
        }
    );
}

fn quality(element_id: &str, exact_scaled_jacobian: f64) -> ElementQuality {
    ElementQuality {
        element_id: element_id.to_string(),
        scaled_jacobian: exact_scaled_jacobian.max(0.0),
        exact_scaled_jacobian,
        aspect_ratio: 1.0,
        volume_m3: 1.0,
    }
}

fn options() -> BoundaryQualityCandidateOptions {
    BoundaryQualityCandidateOptions {
        min_exact_scaled_jacobian: 0.15,
        exact_quality_tolerance: 1.0e-12,
        require_exact_quality_improvement: true,
        allow_min_exact_quality_regression_above_threshold: false,
    }
}
