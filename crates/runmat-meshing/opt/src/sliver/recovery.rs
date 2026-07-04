mod types;

pub use types::{
    SliverClassification, SliverClassificationReason, SliverRecoveryError, SliverRecoveryOptions,
    SliverRemovalEvaluation, SliverRemovalRejectionReason, SliverTetrahedronQuality,
};

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
mod tests;
