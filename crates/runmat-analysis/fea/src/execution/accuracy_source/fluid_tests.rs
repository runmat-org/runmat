use super::*;
use crate::results::accuracy::{
    evaluate_accuracy, publication_gate, AccuracyStatus, PublicationDecision,
};
use runmat_analysis_core::{AccuracyCriterion, AccuracyPlan, AccuracyTarget, AccuracyTolerance};

#[test]
fn original_fluid_closure_cannot_manufacture_mandatory_spatial_accuracy() {
    let (mut w, m) = crate::execution::fluid::tests::authored();
    let mut plan = AccuracyPlan::baseline("fluid_original_accuracy");
    plan.criteria = [
        (
            "momentum",
            ErrorContributionKind::Algebraic,
            RequiredAccuracyEvidence::ResidualBound,
        ),
        (
            "conservation",
            ErrorContributionKind::Algebraic,
            RequiredAccuracyEvidence::ConservationBalance,
        ),
        (
            "spatial",
            ErrorContributionKind::SpatialDiscretization,
            RequiredAccuracyEvidence::ConvergenceStudy,
        ),
    ]
    .into_iter()
    .map(|(id, contribution, evidence)| AccuracyCriterion {
        criterion_id: id.into(),
        target: if contribution == ErrorContributionKind::SpatialDiscretization {
            AccuracyTarget::NumericalResult
        } else {
            AccuracyTarget::GlobalEquilibrium
        },
        contribution,
        evidence,
        tolerance: AccuracyTolerance::Relative(1e-7),
        mandatory: true,
    })
    .collect();
    w.model.sequence.steps[0].accuracy = plan.clone();
    // The requirement and the carried ceiling are one law: admission at exactly the
    // derived workspace succeeds, one byte less refuses.
    let source_bound = crate::planning::resource::fluid_accuracy_source_workspace(
        &w,
        &m,
        w.policy.resident_byte_budget(),
    )
    .unwrap();
    assert_eq!(
        source_bound,
        crate::planning::resource::derive_fluid_steady_resource_projection(&w, &m)
            .unwrap()
            .phase_resident_bytes_upper_bound
            .0
    );
    assert_eq!(
        crate::planning::resource::fluid_accuracy_source_workspace(&w, &m, source_bound).unwrap(),
        source_bound
    );
    assert!(
        crate::planning::resource::fluid_accuracy_source_workspace(&w, &m, source_bound - 1)
            .is_err()
    );
    let result = w.execute(&m).unwrap();
    let receipt = FluidNumericalReceiptArtifact::new(&w, &result, &m).unwrap();
    let source = AccuracySourceReceipt::from_canonical_bytes(
        AccuracySourceKind::FluidSteady,
        &receipt.canonical_bytes().unwrap(),
    )
    .unwrap();
    let request = AccuracyEvaluationRequest::new(
        receipt.receipt().producer,
        AccuracySourceKind::FluidSteady,
        receipt.source_result(),
        None,
        plan,
    )
    .unwrap();
    let evidence = source.evidence(&request).unwrap();
    assert_eq!(evidence.len(), 2);
    assert!(evidence
        .iter()
        .all(|e| e.contribution == ErrorContributionKind::Algebraic
            && e.scope == AccuracyEvidenceScope::SolverResult(receipt.source_result())));
    let evaluation = evaluate_accuracy(&request.plan, &evidence).unwrap();
    assert_eq!(evaluation.report().status, AccuracyStatus::Indeterminate);
    for c in &evaluation.report().contributions {
        assert_eq!(
            c.status,
            if c.criterion_id == "spatial" {
                AccuracyStatus::Indeterminate
            } else {
                AccuracyStatus::Satisfied
            }
        );
    }
    assert_eq!(
        publication_gate(&request.plan, &evaluation)
            .unwrap()
            .decision,
        PublicationDecision::Block
    );
    let mut original_algebraic = request.plan.clone();
    original_algebraic.criteria.truncate(2);
    assert_eq!(
        evaluate_accuracy(&original_algebraic, &evidence)
            .unwrap()
            .report()
            .status,
        AccuracyStatus::Satisfied
    );
    let mut wrong = w.clone();
    wrong.model.sequence.sequence_id = "foreign".into();
    let forged = AccuracyEvaluationRequest::new(
        wrong.producer_identity().unwrap(),
        request.source_kind,
        request.source_result,
        None,
        request.plan.clone(),
    )
    .unwrap();
    assert!(source.evidence(&forged).is_err());
    assert!(receipt.validate_against(&wrong, &result, &m).is_err());
    assert!(AccuracySourceReceipt::from_canonical_bytes(
        AccuracySourceKind::Thermal,
        &receipt.canonical_bytes().unwrap()
    )
    .is_err());
}
