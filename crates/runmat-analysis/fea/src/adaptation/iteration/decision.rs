//! Execution choice from the shared accuracy verdict; no adaptive tolerance model.
use super::{
    StructuralAdaptationConvergenceDecision, StructuralAdaptationDecisionStatus as Status,
    StructuralAdaptationIterationError as Error,
};
use crate::results::accuracy::{AccuracyEvaluation, AccuracyStatus};
use runmat_analysis_core::{AccuracyLimitAction, AccuracyPlan};

pub(super) fn changes(
    source_estimator: f64,
    target_estimator: f64,
    source_quantity: f64,
    target_quantity: f64,
) -> Result<(Option<f64>, f64, f64), Error> {
    let absolute = (target_quantity - source_quantity).abs();
    let scale = source_quantity.abs().max(target_quantity.abs());
    let relative = if scale == 0.0 { 0.0 } else { absolute / scale };
    let reduction = if source_estimator == 0.0 {
        (target_estimator == 0.0).then_some(0.0)
    } else {
        Some(1.0 - target_estimator / source_estimator)
    };
    if !absolute.is_finite() || !relative.is_finite() || reduction.is_some_and(|v| !v.is_finite()) {
        return Err(Error::InvalidTargetQuantity);
    }
    Ok((reduction, absolute, relative))
}

pub(super) fn decide(
    plan: &AccuracyPlan,
    evaluation: &AccuracyEvaluation,
    iteration_index: u64,
    solvers_converged: bool,
    metrics: (Option<f64>, f64, f64),
) -> StructuralAdaptationConvergenceDecision {
    let completed = iteration_index.checked_add(1);
    let ceiling = u64::try_from(plan.resources.maximum_refinement_cycles).ok();
    let exhausted = completed
        .zip(ceiling)
        .is_none_or(|(done, limit)| done >= limit);
    let exceeds = completed
        .zip(ceiling)
        .is_none_or(|(done, limit)| done > limit);
    let status = if !solvers_converged || exceeds {
        Status::Rejected
    } else {
        match evaluation.report().status {
            AccuracyStatus::Satisfied => Status::Converged,
            AccuracyStatus::NotSatisfied
                if plan.exceeded_limit == AccuracyLimitAction::Refine && !exhausted =>
            {
                Status::Continue
            }
            AccuracyStatus::NotSatisfied | AccuracyStatus::Indeterminate => Status::Rejected,
        }
    };
    StructuralAdaptationConvergenceDecision {
        status,
        estimator_reduction: metrics.0,
        target_absolute_change: metrics.1,
        target_relative_change: metrics.2,
        refinement_ceiling_exhausted: exhausted,
    }
}
