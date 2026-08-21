use super::{
    SolverFieldTransferErrorEvidence, StructuralAdaptationConvergenceDecision,
    StructuralAdaptationDecisionStatus, StructuralAdaptationIterationError,
    StructuralAdaptationPolicy, StructuralAdaptationSolverResult,
};

pub(super) fn decide(
    estimator_error: f64,
    transfer_errors: &[SolverFieldTransferErrorEvidence],
    solver: &StructuralAdaptationSolverResult,
    target_value: f64,
    previous_estimator: Option<f64>,
    previous_target: Option<f64>,
    policy: StructuralAdaptationPolicy,
) -> StructuralAdaptationConvergenceDecision {
    let estimator_reduction = previous_estimator.map(|previous| {
        if previous == 0.0 {
            if estimator_error == 0.0 {
                0.0
            } else {
                -1.0
            }
        } else {
            (previous - estimator_error) / previous
        }
    });
    let target_absolute_change = previous_target.map(|previous| (target_value - previous).abs());
    let target_relative_change = previous_target.map(|previous| {
        let scale = previous.abs().max(target_value.abs());
        if scale == 0.0 {
            0.0
        } else {
            (target_value - previous).abs() / scale
        }
    });
    let transfer_accepted = transfer_errors.iter().all(|error| {
        error
            .relative_l2_error
            .is_some_and(|value| value <= policy.maximum_transfer_relative_error)
    });
    let estimator_reduction_accepted = estimator_reduction
        .is_some_and(|reduction| reduction >= policy.minimum_estimator_reduction);
    let estimator_target_met = estimator_error <= policy.estimator_tolerance;
    let target_quantity_converged = target_absolute_change
        .zip(target_relative_change)
        .is_some_and(|(absolute, relative)| {
            absolute <= policy.target_absolute_tolerance
                || relative <= policy.target_relative_tolerance
        });
    let status = if !solver.converged
        || !transfer_accepted
        || (previous_estimator.is_some() && !estimator_reduction_accepted)
    {
        StructuralAdaptationDecisionStatus::Rejected
    } else if estimator_reduction_accepted && estimator_target_met && target_quantity_converged {
        StructuralAdaptationDecisionStatus::Converged
    } else {
        StructuralAdaptationDecisionStatus::Continue
    };
    StructuralAdaptationConvergenceDecision {
        status,
        estimator_reduction,
        target_absolute_change,
        target_relative_change,
        solver_converged: solver.converged,
        transfer_accepted,
        estimator_reduction_accepted,
        estimator_target_met,
        target_quantity_converged,
    }
}

pub(super) fn validate_policy(
    policy: StructuralAdaptationPolicy,
) -> Result<(), StructuralAdaptationIterationError> {
    let values = [
        policy.estimator_tolerance,
        policy.minimum_estimator_reduction,
        policy.target_absolute_tolerance,
        policy.target_relative_tolerance,
        policy.maximum_transfer_relative_error,
    ];
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
        || policy.minimum_estimator_reduction == 0.0
        || policy.minimum_estimator_reduction > 1.0
        || policy.target_absolute_tolerance == 0.0
        || policy.target_relative_tolerance == 0.0
    {
        return Err(StructuralAdaptationIterationError::InvalidPolicy);
    }
    Ok(())
}
