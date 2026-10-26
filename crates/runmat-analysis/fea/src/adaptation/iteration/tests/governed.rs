use super::*;
use crate::execution::{SequenceIdentity, SolverResultIdentity, StepIdentity};
use crate::results::accuracy::AccuracyStatus;
use runmat_analysis_core::{
    AccuracyCriterion, AccuracyLimitAction, AccuracyPlan, AccuracyTarget, AccuracyTolerance,
    ErrorContributionKind as C, ObservationId, ObservationQuantity, RequiredAccuracyEvidence as K,
};
use runmat_execution::Digest;

pub(super) fn model() -> runmat_analysis_core::AnalysisModel {
    let mut model =
        crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
    model.constraints.clear();
    model
}
fn solver(
    model: &runmat_analysis_core::AnalysisModel,
    bytes: &[u8],
) -> StructuralAdaptationSolverResult {
    let sequence = SequenceIdentity::from_solver_contract(&model.sequence).unwrap();
    let step = StepIdentity::from_solver_contract(sequence, &model.sequence.steps[0]).unwrap();
    StructuralAdaptationSolverResult {
        result_identity: SolverResultIdentity::from_step_result(step, Digest::sha256(bytes))
            .unwrap(),
        converged: true,
        iteration_count: 8,
        normalized_residual: 1.0e-10,
    }
}
pub(in crate::adaptation) fn plan() -> AccuracyPlan {
    let mut plan = AccuracyPlan::baseline("adaptive");
    plan.exceeded_limit = AccuracyLimitAction::Refine;
    plan.resources.maximum_refinement_cycles = 4;
    plan.criteria = [
        (
            AccuracyTarget::NumericalResult,
            C::Recovery,
            K::APosterioriEstimator,
            0.2,
        ),
        (
            AccuracyTarget::NumericalResult,
            C::Recovery,
            K::ConvergenceStudy,
            0.95,
        ),
        (
            AccuracyTarget::Observation(ObservationId("tip_displacement".into())),
            C::SpatialDiscretization,
            K::ConvergenceStudy,
            1.0e-3,
        ),
        (
            AccuracyTarget::NumericalResult,
            C::FieldTransfer,
            K::TransferBound,
            0.05,
        ),
        (
            AccuracyTarget::GlobalEquilibrium,
            C::Algebraic,
            K::ResidualBound,
            1.0e-8,
        ),
    ]
    .into_iter()
    .enumerate()
    .map(
        |(i, (target, contribution, evidence, tolerance))| AccuracyCriterion {
            criterion_id: format!("criterion_{i}"),
            target,
            contribution,
            tolerance: AccuracyTolerance::Relative(tolerance),
            evidence,
            mandatory: true,
        },
    )
    .collect();
    plan
}
fn quantity(value: f64) -> StructuralTargetQuantity {
    StructuralTargetQuantity {
        quantity_id: "tip_displacement".into(),
        quantity: ObservationQuantity::Displacement,
        value,
    }
}

// witness: FEA-ADAPTATION-GOVERNANCE-001/durable_accuracy_lineage @
//   adaptive_analysis/tet4/numerical_kernel;
//   boundary: Canonical adaptation records use constructed converged solver facts and quantities,
//   real mesh refinement/transfer fixtures; rejects missing/indeterminate evidence and capped
//   refinement. No independently solved adaptive target convergence or remote lifecycle claim.
#[test]
fn iteration_binds_canonical_solver_mesh_lineage_transfer_and_evidence() {
    let (source, target, transfer, lineage) = adaptation_fixture();
    let model = model();
    let source_estimator = estimate(&source);
    let estimator = estimate(&target);
    let errors = [transfer_error(&source, &target)];
    let source_solver = solver(&model, b"source");
    let target_solver = solver(&model, b"target");
    let baseline = quantity(0.01);
    let target_quantity = quantity(0.011);
    let plan = plan();
    let input = StructuralAdaptationIterationInput {
        model: &model,
        source_artifact: &source,
        target_artifact: &target,
        transfer_map: &transfer,
        lineage: &lineage,
        source_estimator: &source_estimator,
        estimator: &estimator,
        transfer_errors: &errors,
        source_solver_result: &source_solver,
        solver_result: &target_solver,
        source_target_quantity: &baseline,
        target_quantity: &target_quantity,
        previous: None,
    };
    let iteration = build_structural_adaptation_iteration(input, &plan).unwrap();
    assert_eq!(
        iteration.decision.status,
        StructuralAdaptationDecisionStatus::Continue
    );
    assert_eq!(
        iteration.accuracy_evaluation.report().status,
        AccuracyStatus::NotSatisfied
    );
    iteration
        .validate_against(&model, &source, &target, &transfer, &lineage)
        .unwrap();
    let encoded = iteration.canonical_encode().unwrap();
    assert_eq!(
        StructuralAdaptationIteration::canonical_decode(&encoded).unwrap(),
        iteration
    );
    for mutation in 0..5 {
        let mut bad = iteration.clone();
        match mutation {
            0 => bad.schema_version = 2,
            1 => bad.estimator.reference_norm = 0.0,
            2 => bad.decision.target_relative_change = 0.0,
            3 => bad.constraint_lineage.target_plan_digest = Digest::sha256(b"foreign"),
            _ => bad.previous_iteration_digest = Some(StableDigest::from_bytes([5; 32])),
        }
        let invalid = if mutation == 3 {
            bad.validate_against(&model, &source, &target, &transfer, &lineage)
        } else {
            bad.validate()
        };
        assert!(invalid.is_err(), "mutation {mutation}");
    }
    // Continue from a cold retained record through a second real mesh lineage.
    let previous = StructuralAdaptationIteration::canonical_decode(&encoded).unwrap();
    let (next_source, next_target, next_transfer, next_lineage) =
        refine_fixture(target.clone(), StableDigest::from_bytes([78; 32]));
    let next_estimator = estimate(&next_target);
    let next_errors = [transfer_error(&next_source, &next_target)];
    let next_solver = solver(&model, b"second-target");
    let next = build_structural_adaptation_iteration(
        StructuralAdaptationIterationInput {
            model: &model,
            source_artifact: &next_source,
            target_artifact: &next_target,
            transfer_map: &next_transfer,
            lineage: &next_lineage,
            source_estimator: &previous.estimator,
            estimator: &next_estimator,
            transfer_errors: &next_errors,
            source_solver_result: &previous.solver_result,
            solver_result: &next_solver,
            source_target_quantity: &previous.target_quantity,
            target_quantity: &previous.target_quantity,
            previous: Some(&previous),
        },
        &plan,
    )
    .unwrap();
    next.validate_previous(&previous).unwrap();
    let mut wrong_previous = previous.clone();
    wrong_previous.source_solver_result.iteration_count += 1;
    // A changed previous record may be internally valid but has a different digest.
    wrong_previous.validate().unwrap();
    assert!(next.validate_previous(&wrong_previous).is_err());
    let mut policy_missing = plan.clone();
    policy_missing.criteria.remove(0);
    assert_eq!(
        build_structural_adaptation_iteration(input, &policy_missing).unwrap_err(),
        StructuralAdaptationIterationError::InvalidAccuracyPlan
    );
    let mut drifting_plan = plan.clone();
    drifting_plan.criteria[1].tolerance = AccuracyTolerance::Relative(1.01);
    assert_eq!(
        build_structural_adaptation_iteration(input, &drifting_plan).unwrap_err(),
        StructuralAdaptationIterationError::InvalidAccuracyPlan
    );
    let final_quantity = quantity(0.010000001);
    let converged = build_structural_adaptation_iteration(
        StructuralAdaptationIterationInput {
            target_quantity: &final_quantity,
            ..input
        },
        &plan,
    )
    .unwrap();
    assert_eq!(
        converged.decision.status,
        StructuralAdaptationDecisionStatus::Converged
    );
    assert_eq!(
        converged.accuracy_evaluation.report().status,
        AccuracyStatus::Satisfied
    );
    let mut absolute = plan.clone();
    absolute.criteria[2].tolerance = AccuracyTolerance::Absolute {
        quantity: ObservationQuantity::Displacement,
        canonical_value: 1.0e-6,
    };
    assert_eq!(
        build_structural_adaptation_iteration(
            StructuralAdaptationIterationInput {
                target_quantity: &final_quantity,
                ..input
            },
            &absolute
        )
        .unwrap()
        .decision
        .status,
        StructuralAdaptationDecisionStatus::Converged
    );
    absolute.criteria[2].tolerance = AccuracyTolerance::Absolute {
        quantity: ObservationQuantity::Stress,
        canonical_value: 1.0e-6,
    };
    assert_eq!(
        build_structural_adaptation_iteration(input, &absolute).unwrap_err(),
        StructuralAdaptationIterationError::InvalidAccuracyPlan
    );
    let mut incomplete = plan.clone();
    incomplete.criteria.push(AccuracyCriterion {
        criterion_id: "mandatory_geometry".into(),
        target: AccuracyTarget::NumericalResult,
        contribution: C::GeometryAdmission,
        tolerance: AccuracyTolerance::Relative(1.0e-3),
        evidence: K::ReferenceComparison,
        mandatory: true,
    });
    assert_eq!(
        build_structural_adaptation_iteration(
            StructuralAdaptationIterationInput {
                target_quantity: &final_quantity,
                ..input
            },
            &incomplete
        )
        .unwrap()
        .decision
        .status,
        StructuralAdaptationDecisionStatus::Rejected
    );
    let mut tight = plan.clone();
    tight.resources.maximum_refinement_cycles = 1;
    let exhausted = build_structural_adaptation_iteration(input, &tight).unwrap();
    assert_eq!(
        exhausted.decision.status,
        StructuralAdaptationDecisionStatus::Rejected
    );
    assert!(exhausted.decision.refinement_ceiling_exhausted);
    let mut missing = errors.clone();
    missing[0].relative_l2_error = None;
    let indeterminate = build_structural_adaptation_iteration(
        StructuralAdaptationIterationInput {
            transfer_errors: &missing,
            target_quantity: &final_quantity,
            ..input
        },
        &plan,
    )
    .unwrap();
    assert_eq!(
        indeterminate.accuracy_evaluation.report().status,
        AccuracyStatus::Indeterminate
    );
    assert_eq!(
        indeterminate.decision.status,
        StructuralAdaptationDecisionStatus::Rejected
    );
    let mut false_solver = target_solver.clone();
    false_solver.converged = false;
    assert_eq!(
        build_structural_adaptation_iteration(
            StructuralAdaptationIterationInput {
                solver_result: &false_solver,
                target_quantity: &final_quantity,
                ..input
            },
            &plan
        )
        .unwrap()
        .decision
        .status,
        StructuralAdaptationDecisionStatus::Rejected
    );
}
