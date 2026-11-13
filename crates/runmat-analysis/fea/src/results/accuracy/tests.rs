use super::*;

fn scope() -> AccuracyEvidenceScope {
    let sequence = crate::execution::SequenceIdentity::new(Digest::sha256("sequence")).unwrap();
    AccuracyEvidenceScope::Step(StepIdentity::new(sequence, Digest::sha256("step")).unwrap())
}

fn step_identity() -> StepIdentity {
    let AccuracyEvidenceScope::Step(step) = scope() else {
        unreachable!()
    };
    step
}

fn evidence(value: f64) -> AccuracyEvidence {
    AccuracyEvidence::new(
        AccuracyTarget::GlobalEquilibrium,
        ErrorContributionKind::Algebraic,
        RequiredAccuracyEvidence::ResidualBound,
        AccuracyMeasure::Relative(value),
        vec!["canonical residual norm".into()],
        scope(),
    )
    .unwrap()
}

// witness: FEA-SHARED-ACCURACY-001/mandatory_acceptance_states @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Scalar residual criterion failure remains blocking through a typed accuracy artifact;
//   no complete governed family corpus, release attestation or public receipt qualification.
#[test]
fn production_accuracy_failure_produces_a_blocking_gate() {
    let plan = AccuracyPlan::baseline("accuracy");
    let evaluation = evaluate_accuracy(&plan, &[evidence(1.0e-6)]).unwrap();
    let gate = publication_gate(&plan, &evaluation).unwrap();
    assert_eq!(gate.decision, PublicationDecision::Block);
    let producer = step_identity();
    let source_result =
        SolverResultIdentity::from_step_result(producer, Digest::sha256("solver-result")).unwrap();
    let request = crate::execution::AccuracyEvaluationRequest::new(
        producer,
        crate::execution::AccuracySourceKind::StructuralLinear,
        source_result,
        None,
        plan,
    )
    .unwrap();
    let artifact = AccuracyArtifact::new(AccuracyArtifactInput {
        evaluation_request: request.identity(),
        producer,
        source_kind: request.source_kind,
        source_result,
        source_observations: None,
        evidence: vec![evidence(1.0e-6)],
        evaluation,
        gate,
    })
    .unwrap();
    assert_eq!(artifact.gate().decision, PublicationDecision::Block);
}

// witness: FEA-SHARED-ACCURACY-001/mandatory_acceptance_states @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: Mandatory evidence governs status and publication.
#[test]
fn mandatory_evidence_governs_status_and_production_publication() {
    let plan = AccuracyPlan::baseline("accuracy");
    let satisfied = evaluate_accuracy(&plan, &[evidence(1.0e-9)]).unwrap();
    assert_eq!(satisfied.report.status, AccuracyStatus::Satisfied);
    assert_eq!(
        publication_gate(&plan, &satisfied).unwrap().decision,
        PublicationDecision::Allow
    );

    let failed = evaluate_accuracy(&plan, &[evidence(1.0e-6)]).unwrap();
    assert_eq!(failed.report.status, AccuracyStatus::NotSatisfied);
    assert_eq!(
        publication_gate(&plan, &failed).unwrap().decision,
        PublicationDecision::Block
    );
}

// witness: FEA-SHARED-ACCURACY-001/mandatory_acceptance_states @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: Missing mandatory evidence is indeterminate and blocks publication by default.
#[test]
fn missing_mandatory_evidence_is_indeterminate_and_blocks_by_default() {
    let plan = AccuracyPlan::baseline("accuracy");
    let evaluation = evaluate_accuracy(&plan, &[]).unwrap();
    assert_eq!(evaluation.report.status, AccuracyStatus::Indeterminate);
    assert_eq!(
        publication_gate(&plan, &evaluation).unwrap().decision,
        PublicationDecision::Block
    );

    // Publishing an indeterminate result is an explicit authored choice; it stays labeled.
    let mut opted_in = AccuracyPlan::baseline("accuracy");
    opted_in.unavailable_evidence =
        runmat_analysis_core::UnavailableEvidencePolicy::MarkIndeterminate;
    let evaluation = evaluate_accuracy(&opted_in, &[]).unwrap();
    let gate = publication_gate(&opted_in, &evaluation).unwrap();
    assert_eq!(gate.decision, PublicationDecision::Allow);
    assert_eq!(gate.status, AccuracyStatus::Indeterminate);
}

#[test]
fn report_identity_is_independent_of_evidence_submission_order() {
    let mut plan = AccuracyPlan::baseline("accuracy");
    plan.criteria.push(runmat_analysis_core::AccuracyCriterion {
        criterion_id: "precision".into(),
        target: AccuracyTarget::ExecutionEquivalence,
        contribution: ErrorContributionKind::ExecutionEquivalence,
        tolerance: AccuracyTolerance::Relative(1.0e-10),
        evidence: RequiredAccuracyEvidence::ResidualBound,
        mandatory: false,
    });
    let second = AccuracyEvidence::new(
        AccuracyTarget::ExecutionEquivalence,
        ErrorContributionKind::ExecutionEquivalence,
        RequiredAccuracyEvidence::ResidualBound,
        AccuracyMeasure::Relative(1.0e-12),
        vec!["same canonical operator".into()],
        scope(),
    )
    .unwrap();
    let first = evidence(1.0e-9);
    let forward = evaluate_accuracy(&plan, &[first.clone(), second.clone()]).unwrap();
    let reverse = evaluate_accuracy(&plan, &[second, first]).unwrap();
    assert_eq!(forward.identity, reverse.identity);
    assert_eq!(forward.report, reverse.report);
}

#[test]
fn publication_rejects_a_changed_plan_with_the_same_authored_id() {
    let plan = AccuracyPlan::baseline("accuracy");
    let evaluation = evaluate_accuracy(&plan, &[evidence(1.0e-9)]).unwrap();
    let mut changed = plan.clone();
    changed.criteria[0].tolerance = AccuracyTolerance::Relative(1.0e-4);

    assert!(publication_gate(&changed, &evaluation).is_err());
}

#[test]
fn publication_rejects_mutated_report_content() {
    let plan = AccuracyPlan::baseline("accuracy");
    let mut evaluation = evaluate_accuracy(&plan, &[evidence(1.0e-9)]).unwrap();
    evaluation.report.status = AccuracyStatus::NotSatisfied;

    assert!(publication_gate(&plan, &evaluation).is_err());
}

#[test]
fn deserialization_rejects_a_nonconvergent_report_identity() {
    let plan = AccuracyPlan::baseline("accuracy");
    let evaluation = evaluate_accuracy(&plan, &[evidence(1.0e-9)]).unwrap();
    let mut encoded = serde_json::to_value(evaluation).unwrap();
    encoded["report"]["status"] = serde_json::json!("not_satisfied");

    assert!(serde_json::from_value::<AccuracyEvaluation>(encoded).is_err());
}

#[test]
fn evidence_identity_is_derived_from_typed_scope_and_content() {
    let baseline = evidence(1.0e-9);
    let mut encoded = serde_json::to_value(&baseline).unwrap();
    encoded["measured"]["relative"] = serde_json::json!(1.0e-3);

    assert!(serde_json::from_value::<AccuracyEvidence>(encoded).is_err());

    let different_scope = AccuracyEvidence::new(
        AccuracyTarget::GlobalEquilibrium,
        ErrorContributionKind::Algebraic,
        RequiredAccuracyEvidence::ResidualBound,
        AccuracyMeasure::Relative(1.0e-9),
        vec!["canonical residual norm".into()],
        AccuracyEvidenceScope::Problem(
            ProblemIdentity::new(
                match scope() {
                    AccuracyEvidenceScope::Step(step) => step,
                    _ => unreachable!(),
                },
                Digest::sha256("model"),
                None,
                crate::execution::ConstraintPlanIdentity::new(
                    match scope() {
                        AccuracyEvidenceScope::Step(step) => step,
                        _ => unreachable!(),
                    },
                    Digest::sha256("mesh"),
                    Digest::sha256("constraints"),
                )
                .unwrap(),
                Vec::new(),
            )
            .unwrap(),
        ),
    )
    .unwrap();
    assert_ne!(baseline.identity(), different_scope.identity());
}
