use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SliverRecoveryOptions {
    pub sliver_aspect_ratio: f64,
    pub min_exact_scaled_jacobian: f64,
    pub exact_quality_tolerance: f64,
}

impl Default for SliverRecoveryOptions {
    fn default() -> Self {
        Self {
            sliver_aspect_ratio: 20.0,
            min_exact_scaled_jacobian: 0.15,
            exact_quality_tolerance: 1.0e-12,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SliverTetrahedronQuality {
    pub tetrahedron_id: u32,
    pub aspect_ratio: f64,
    pub exact_scaled_jacobian: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SliverClassificationReason {
    AspectRatioOnly,
    AspectRatioAndExactQuality,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SliverClassification {
    pub tetrahedron_id: u32,
    pub aspect_ratio: f64,
    pub exact_scaled_jacobian: f64,
    pub reason: SliverClassificationReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SliverRemovalRejectionReason {
    NoInitialSlivers,
    SliverCountNotReduced,
    ExactQualityViolationRegressed,
    MinimumExactQualityRegressed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SliverRemovalEvaluation {
    pub accepted: bool,
    pub initial_sliver_count: usize,
    pub final_sliver_count: usize,
    pub removed_sliver_count: usize,
    pub initial_exact_quality_violation_count: usize,
    pub final_exact_quality_violation_count: usize,
    pub initial_min_exact_scaled_jacobian: f64,
    pub final_min_exact_scaled_jacobian: f64,
    pub initial_max_aspect_ratio: f64,
    pub final_max_aspect_ratio: f64,
    #[serde(default)]
    pub rejection_reason: Option<SliverRemovalRejectionReason>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SliverRecoveryError {
    InvalidOptions,
    NonFiniteQuality { tetrahedron_id: u32 },
}

pub fn classify_sliver_tetrahedra(
    tetrahedra: &[SliverTetrahedronQuality],
    options: SliverRecoveryOptions,
) -> Result<Vec<SliverClassification>, SliverRecoveryError> {
    validate_options(options)?;
    let mut classifications = Vec::<SliverClassification>::new();
    for tetrahedron in tetrahedra {
        validate_tetrahedron_quality(*tetrahedron)?;
        if tetrahedron.aspect_ratio <= options.sliver_aspect_ratio {
            continue;
        }
        classifications.push(SliverClassification {
            tetrahedron_id: tetrahedron.tetrahedron_id,
            aspect_ratio: tetrahedron.aspect_ratio,
            exact_scaled_jacobian: tetrahedron.exact_scaled_jacobian,
            reason: if tetrahedron.exact_scaled_jacobian < options.min_exact_scaled_jacobian {
                SliverClassificationReason::AspectRatioAndExactQuality
            } else {
                SliverClassificationReason::AspectRatioOnly
            },
        });
    }
    classifications.sort_by(|left, right| {
        right
            .aspect_ratio
            .total_cmp(&left.aspect_ratio)
            .then_with(|| left.tetrahedron_id.cmp(&right.tetrahedron_id))
    });
    Ok(classifications)
}

pub fn evaluate_sliver_removal(
    current: &[SliverTetrahedronQuality],
    proposed: &[SliverTetrahedronQuality],
    options: SliverRecoveryOptions,
) -> Result<SliverRemovalEvaluation, SliverRecoveryError> {
    validate_options(options)?;
    validate_quality_set(current)?;
    validate_quality_set(proposed)?;

    let current_summary = sliver_quality_summary(current, options);
    let proposed_summary = sliver_quality_summary(proposed, options);
    let removed_sliver_count = current_summary
        .sliver_count
        .saturating_sub(proposed_summary.sliver_count);
    let rejection_reason = if current_summary.sliver_count == 0 {
        Some(SliverRemovalRejectionReason::NoInitialSlivers)
    } else if proposed_summary.sliver_count >= current_summary.sliver_count {
        Some(SliverRemovalRejectionReason::SliverCountNotReduced)
    } else if proposed_summary.exact_quality_violation_count
        > current_summary.exact_quality_violation_count
    {
        Some(SliverRemovalRejectionReason::ExactQualityViolationRegressed)
    } else if proposed_summary.min_exact_scaled_jacobian + options.exact_quality_tolerance
        < current_summary.min_exact_scaled_jacobian
    {
        Some(SliverRemovalRejectionReason::MinimumExactQualityRegressed)
    } else {
        None
    };

    Ok(SliverRemovalEvaluation {
        accepted: rejection_reason.is_none(),
        initial_sliver_count: current_summary.sliver_count,
        final_sliver_count: proposed_summary.sliver_count,
        removed_sliver_count,
        initial_exact_quality_violation_count: current_summary.exact_quality_violation_count,
        final_exact_quality_violation_count: proposed_summary.exact_quality_violation_count,
        initial_min_exact_scaled_jacobian: current_summary.min_exact_scaled_jacobian,
        final_min_exact_scaled_jacobian: proposed_summary.min_exact_scaled_jacobian,
        initial_max_aspect_ratio: current_summary.max_aspect_ratio,
        final_max_aspect_ratio: proposed_summary.max_aspect_ratio,
        rejection_reason,
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct SliverQualitySummary {
    sliver_count: usize,
    exact_quality_violation_count: usize,
    min_exact_scaled_jacobian: f64,
    max_aspect_ratio: f64,
}

fn sliver_quality_summary(
    tetrahedra: &[SliverTetrahedronQuality],
    options: SliverRecoveryOptions,
) -> SliverQualitySummary {
    let mut sliver_count = 0_usize;
    let mut exact_quality_violation_count = 0_usize;
    let mut min_exact_scaled_jacobian = f64::INFINITY;
    let mut max_aspect_ratio = 0.0_f64;
    for tetrahedron in tetrahedra {
        sliver_count += usize::from(tetrahedron.aspect_ratio > options.sliver_aspect_ratio);
        exact_quality_violation_count +=
            usize::from(tetrahedron.exact_scaled_jacobian < options.min_exact_scaled_jacobian);
        min_exact_scaled_jacobian =
            min_exact_scaled_jacobian.min(tetrahedron.exact_scaled_jacobian);
        max_aspect_ratio = max_aspect_ratio.max(tetrahedron.aspect_ratio);
    }
    if tetrahedra.is_empty() {
        min_exact_scaled_jacobian = 0.0;
    }
    SliverQualitySummary {
        sliver_count,
        exact_quality_violation_count,
        min_exact_scaled_jacobian,
        max_aspect_ratio,
    }
}

fn validate_options(options: SliverRecoveryOptions) -> Result<(), SliverRecoveryError> {
    if !options.sliver_aspect_ratio.is_finite()
        || options.sliver_aspect_ratio <= 0.0
        || !options.min_exact_scaled_jacobian.is_finite()
        || options.min_exact_scaled_jacobian < 0.0
        || !options.exact_quality_tolerance.is_finite()
        || options.exact_quality_tolerance < 0.0
    {
        return Err(SliverRecoveryError::InvalidOptions);
    }
    Ok(())
}

fn validate_quality_set(
    tetrahedra: &[SliverTetrahedronQuality],
) -> Result<(), SliverRecoveryError> {
    for tetrahedron in tetrahedra {
        validate_tetrahedron_quality(*tetrahedron)?;
    }
    Ok(())
}

fn validate_tetrahedron_quality(
    tetrahedron: SliverTetrahedronQuality,
) -> Result<(), SliverRecoveryError> {
    if !tetrahedron.aspect_ratio.is_finite()
        || tetrahedron.aspect_ratio <= 0.0
        || !tetrahedron.exact_scaled_jacobian.is_finite()
    {
        return Err(SliverRecoveryError::NonFiniteQuality {
            tetrahedron_id: tetrahedron.tetrahedron_id,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_slivers_by_aspect_ratio_and_quality() {
        let tetrahedra = vec![
            quality(1, 8.0, 0.6),
            quality(2, 25.0, 0.5),
            quality(3, 30.0, 0.05),
        ];

        let classifications = classify_sliver_tetrahedra(&tetrahedra, options())
            .expect("classification should succeed");

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
}
