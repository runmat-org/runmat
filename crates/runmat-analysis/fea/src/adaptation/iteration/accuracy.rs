//! Adapt physical measures into the existing study accuracy authority.
use super::{
    StructuralAdaptationIterationError as Error, StructuralAdaptationSolverResult,
    StructuralTargetQuantity,
};
use crate::adaptation::{SolverFieldTransferErrorEvidence, StructuralRecoveryEstimate};
use crate::results::accuracy::{
    evaluate_accuracy, AccuracyEvaluation, AccuracyEvidence, AccuracyEvidenceScope, AccuracyMeasure,
};
use runmat_analysis_core::{
    AccuracyPlan, AccuracyTarget, AccuracyTolerance, ErrorContributionKind as Contribution,
    ObservationId, RequiredAccuracyEvidence as Kind,
};

pub(super) struct AccuracyInput<'a> {
    pub plan: &'a AccuracyPlan,
    pub source_estimator: &'a StructuralRecoveryEstimate,
    pub estimator: &'a StructuralRecoveryEstimate,
    pub transfer_errors: &'a [SolverFieldTransferErrorEvidence],
    pub solver_result: &'a StructuralAdaptationSolverResult,
    pub target_quantity: &'a StructuralTargetQuantity,
    pub absolute_change: f64,
    pub relative_change: f64,
}
fn target(quantity: &StructuralTargetQuantity) -> AccuracyTarget {
    AccuracyTarget::Observation(ObservationId(quantity.quantity_id.clone()))
}

pub(super) fn evaluate(input: AccuracyInput<'_>) -> Result<AccuracyEvaluation, Error> {
    validate_plan(
        input.plan,
        &target(input.target_quantity),
        input.target_quantity.quantity,
    )?;
    let scope = AccuracyEvidenceScope::SolverResult(input.solver_result.result_identity);
    let mut evidence = Vec::new();
    let mut add =
        |target, contribution, kind, measured, assumptions: Vec<String>| -> Result<(), Error> {
            evidence.push(
                AccuracyEvidence::new(target, contribution, kind, measured, assumptions, scope)
                    .map_err(|e| Error::InvalidAccuracyEvidence(e.to_string()))?,
            );
            Ok(())
        };
    if let Some(relative) = input.estimator.relative_error() {
        add(AccuracyTarget::NumericalResult, Contribution::Recovery, Kind::APosterioriEstimator, AccuracyMeasure::Relative(relative), vec!["Frobenius volume-L2 discrepancy divided by the original solved stress norm; recovery estimator is not a rigorous bound".into()])?;
    }
    let ratio = if input.source_estimator.total_error == 0.0 {
        (input.estimator.total_error == 0.0).then_some(0.0)
    } else {
        Some(input.estimator.total_error / input.source_estimator.total_error)
    };
    if let Some(ratio) = ratio {
        add(AccuracyTarget::NumericalResult, Contribution::Recovery, Kind::ConvergenceStudy, AccuracyMeasure::Relative(ratio), vec!["Ratio of paired original recovery discrepancies; zero/zero means both discrepancies vanish".into()])?;
    }
    if let Some(criterion) = input.plan.criteria.iter().find(|c| {
        c.target == target(input.target_quantity)
            && c.contribution == Contribution::SpatialDiscretization
            && c.evidence == Kind::ConvergenceStudy
    }) {
        let measured = match criterion.tolerance {
            AccuracyTolerance::Relative(_) => AccuracyMeasure::Relative(input.relative_change),
            AccuracyTolerance::Absolute { quantity, .. } => AccuracyMeasure::Absolute {
                quantity,
                canonical_value: input.absolute_change,
            },
        };
        add(target(input.target_quantity), Contribution::SpatialDiscretization, Kind::ConvergenceStudy, measured, vec!["Change of the paired solved target quantity; this is convergence evidence, not an independent exact error bound".into()])?;
    }
    if input
        .transfer_errors
        .iter()
        .all(|error| error.relative_l2_error.is_some())
    {
        for transfer in input.transfer_errors {
            if let Some(relative) = transfer.relative_l2_error {
                add(
                    AccuracyTarget::NumericalResult,
                    Contribution::FieldTransfer,
                    Kind::TransferBound,
                    AccuracyMeasure::Relative(relative),
                    vec![format!(
                        "Original transferred/reference fields {:?} / {:?} on {}",
                        transfer.transferred_field_digest,
                        transfer.reference_field_digest,
                        transfer.transfer.topology_id
                    )],
                )?;
            }
        }
    }
    add(
        AccuracyTarget::GlobalEquilibrium,
        Contribution::Algebraic,
        Kind::ResidualBound,
        AccuracyMeasure::Relative(input.solver_result.normalized_residual),
        Vec::new(),
    )?;
    evaluate_accuracy(input.plan, &evidence)
        .map_err(|e| Error::InvalidAccuracyEvidence(e.to_string()))
}

fn validate_plan(
    plan: &AccuracyPlan,
    observation: &AccuracyTarget,
    quantity: runmat_analysis_core::ObservationQuantity,
) -> Result<(), Error> {
    let required = [
        (
            AccuracyTarget::NumericalResult,
            Contribution::Recovery,
            Kind::APosterioriEstimator,
        ),
        (
            AccuracyTarget::NumericalResult,
            Contribution::Recovery,
            Kind::ConvergenceStudy,
        ),
        (
            observation.clone(),
            Contribution::SpatialDiscretization,
            Kind::ConvergenceStudy,
        ),
        (
            AccuracyTarget::NumericalResult,
            Contribution::FieldTransfer,
            Kind::TransferBound,
        ),
        (
            AccuracyTarget::GlobalEquilibrium,
            Contribution::Algebraic,
            Kind::ResidualBound,
        ),
    ];
    for (target, contribution, kind) in required {
        let matching = plan
            .criteria
            .iter()
            .filter(|c| c.target == target && c.contribution == contribution && c.evidence == kind)
            .collect::<Vec<_>>();
        if matching
            .iter()
            .any(|c| matches!(c.tolerance, AccuracyTolerance::Relative(_)))
            && matching
                .iter()
                .any(|c| matches!(c.tolerance, AccuracyTolerance::Absolute { .. }))
        {
            return Err(Error::InvalidAccuracyPlan);
        }
        if !matching.iter().any(|c| c.mandatory)
            || matching.iter().any(|c| match c.tolerance {
                AccuracyTolerance::Relative(limit) => {
                    contribution == Contribution::Recovery
                        && kind == Kind::ConvergenceStudy
                        && !(0.0..1.0).contains(&limit)
                }
                AccuracyTolerance::Absolute { quantity: q, .. } => {
                    target != *observation || q != quantity
                }
            })
        {
            return Err(Error::InvalidAccuracyPlan);
        }
    }
    // Shared validation owns numerical tolerances, actions and resource shapes.
    evaluate_accuracy(plan, &[]).map_err(|_| Error::InvalidAccuracyPlan)?;
    Ok(())
}
