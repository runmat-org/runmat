//! Refinement policy uses the complete affected quality under its own resolved
//! metric. Construction-source geometry never substitutes for trial quality.
use super::{DelaunayVolumeRefinementStepError, DelaunayVolumeRefinementStepErrorKind};
use crate::cdt::volume_quality::{
    violation_priority_key,
    working::{SpectrumDelta, WorkingVolumeQuality},
    QualityObjective,
};
use crate::cdt::DelaunayVolumeQuality;
use runmat_meshing_core::MeshingCancellationSignal;
use std::cmp::Ordering;

/// Standalone subdivision preserves shape; the complete coupled scheduler may
/// transfer a defect only while its shared size/shape spectrum strictly improves.
#[derive(Clone, Copy, Default)]
pub(super) enum RefinementProgressPolicy {
    #[default]
    PreserveShape,
    Coupled,
}
impl RefinementProgressPolicy {
    /// The first progress clause a rejected trial fails, for rejection diagnostics.
    pub(super) fn rejection(
        self,
        quality: &WorkingVolumeQuality,
        parent: &SpectrumDelta,
    ) -> Option<&'static str> {
        if quality.compare_target(parent, QualityObjective::Refinement) != Ordering::Less {
            return Some("refinement spectrum did not improve");
        }
        match self {
            Self::PreserveShape => {
                if quality.compare_target(parent, QualityObjective::Shape) == Ordering::Greater {
                    return Some("shape spectrum regressed");
                }
            }
            Self::Coupled => {
                // Each insertion must outrank what it creates with what it removes (multiset
                // order on the joint defect histogram). Requiring it to improve the single
                // worst defect of the whole mesh instead rejected every insertion away from
                // that defect and stalled refinement on plain plates and beams.
                if quality.compare_joint_target(parent) != Ordering::Less {
                    return Some("joint defect histogram did not decrease");
                }
            }
        }
        quality
            .creates_all_boundary_refinement_violation()
            .then_some("created an all-boundary refinement violation")
    }
}

pub(super) fn canonical_progress(
    prior: &DelaunayVolumeQuality,
    result: &DelaunayVolumeQuality,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<bool, DelaunayVolumeRefinementStepError> {
    Ok(
        compare(prior, result, QualityObjective::Refinement, cancellation)? == Ordering::Less
            && compare(prior, result, QualityObjective::Shape, cancellation)? != Ordering::Greater,
    )
}

/// A public single subdivision may preserve its affected quality ceiling while
/// splitting a cell. Unlike the coupled batch convergence policy, it does not
/// promise strict spectrum progress in one point. Complete-loop budgets and the
/// independent terminal quality authority remain mandatory. Compare actual
/// changed cells under their own metrics; unrelated bad cells cannot mask harm.
pub(super) fn canonical_single_step_nonregression(
    prior: &DelaunayVolumeQuality,
    result: &DelaunayVolumeQuality,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<bool, DelaunayVolumeRefinementStepError> {
    let (mut before, mut after) = (0usize, 0usize);
    let (mut prior_worst, mut resulting_worst) = (0.0_f64, 0.0_f64);
    let mut examined = 0usize;
    let key = |cell: &crate::cdt::DelaunayTetrahedronQuality| {
        let mut identities = cell.node_identities;
        identities.sort_unstable();
        identities
    };
    // Independently admitted quality inventories are ordered by canonical cell
    // identities. A merge walk needs no second whole-mesh dictionary or copy.
    while before < prior.tetrahedra.len() || after < result.tetrahedra.len() {
        if examined.is_multiple_of(128) && cancellation.is_cancelled() {
            return Err(super::insertion::error(
                DelaunayVolumeRefinementStepErrorKind::Cancelled,
                "single refinement comparison cancelled",
            ));
        }
        examined += 1;
        let left = prior.tetrahedra.get(before);
        let right = result.tetrahedra.get(after);
        match (left, right) {
            (Some(left), Some(right)) if key(left) == key(right) => {
                if left != right {
                    prior_worst = prior_worst.max(left.refinement_violation_ratio);
                    resulting_worst = resulting_worst.max(right.refinement_violation_ratio);
                }
                before += 1;
                after += 1;
            }
            (Some(left), Some(right)) if key(left) < key(right) => {
                prior_worst = prior_worst.max(left.refinement_violation_ratio);
                before += 1;
            }
            (Some(_), Some(right)) => {
                resulting_worst = resulting_worst.max(right.refinement_violation_ratio);
                after += 1;
            }
            (Some(left), None) => {
                prior_worst = prior_worst.max(left.refinement_violation_ratio);
                before += 1;
            }
            (None, Some(right)) => {
                resulting_worst = resulting_worst.max(right.refinement_violation_ratio);
                after += 1;
            }
            (None, None) => break,
        }
    }
    if cancellation.is_cancelled() {
        return Err(super::insertion::error(
            DelaunayVolumeRefinementStepErrorKind::Cancelled,
            "single refinement comparison cancelled",
        ));
    }
    Ok(
        violation_priority_key(resulting_worst) <= violation_priority_key(prior_worst)
            && !(prior_worst == f64::MAX && resulting_worst == f64::MAX),
    )
}

pub(super) fn compare(
    prior: &DelaunayVolumeQuality,
    result: &DelaunayVolumeQuality,
    objective: QualityObjective,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Ordering, DelaunayVolumeRefinementStepError> {
    let checkpoint = || {
        if cancellation.is_cancelled() {
            Err(super::insertion::error(
                DelaunayVolumeRefinementStepErrorKind::Cancelled,
                "refinement quality comparison cancelled",
            ))
        } else {
            Ok(())
        }
    };
    let spectrum = |quality: &DelaunayVolumeQuality| {
        let mut values = Vec::new();
        for (index, cell) in quality.tetrahedra.iter().enumerate() {
            if index.is_multiple_of(128) {
                checkpoint()?;
            }
            let ratio = match objective {
                QualityObjective::Refinement => cell.refinement_violation_ratio,
                QualityObjective::Shape => cell.optimization_violation_ratio,
            };
            if ratio > 1.0 {
                values.push(violation_priority_key(ratio));
            }
        }
        runmat_cpu_execution::sort_in_place_with_checkpoint(
            &mut values,
            |left, right| right.cmp(left),
            checkpoint,
        )?;
        Ok::<_, DelaunayVolumeRefinementStepError>(values)
    };
    let left = spectrum(result)?;
    let right = spectrum(prior)?;
    for (index, (left, right)) in left.iter().zip(&right).enumerate() {
        if index.is_multiple_of(128) {
            checkpoint()?;
        }
        let order = left.cmp(right);
        if order != Ordering::Equal {
            return Ok(order);
        }
    }
    checkpoint()?;
    Ok(left.len().cmp(&right.len()))
}

#[cfg(test)]
mod tests;
