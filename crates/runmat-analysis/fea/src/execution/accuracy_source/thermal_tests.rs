use super::*;
use crate::{
    execution::{SequenceIdentity, StepIdentity, ThermalWorkload},
    fixtures::{fixture_model, FixtureId},
    results::accuracy::{evaluate_accuracy, publication_gate, AccuracyStatus, PublicationDecision},
    ComputeBackend, ThermalProcedureControl,
};
use runmat_analysis_core::{
    AccuracyCriterion, AccuracyPlan, AccuracyTarget, AccuracyTolerance, AnalysisSequence,
    ProcedureStep, Temperature, ThermalDomain,
};
use runmat_execution::Digest;
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

mod nonlinear;

fn receipt(order: ElementOrder) -> ThermalNumericalReceiptArtifact {
    receipt_with_energy(order, None)
}

fn receipt_with_energy(
    order: ElementOrder,
    energy: Option<crate::contracts::ThermalEnergyBalance>,
) -> ThermalNumericalReceiptArtifact {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    model.sequence = AnalysisSequence::from_steps(
        "thermal_accuracy",
        vec![ProcedureStep::thermal_transient("heat")],
    );
    let workload = ThermalWorkload {
        initial_fields: Vec::new(),
        step_id: model.sequence.steps[0].step_id.clone(),
        model,
        backend: ComputeBackend::Cpu,
        procedure: ThermalProcedureControl::BackwardEuler { snapshot_count: 3 },
        residual_target: 1e-4,
    };
    let mut result = workload
        .execute(&canonical_tetrahedron_solver_mesh(order))
        .unwrap();
    // Controlled accepted-residual fixture exercises policy thresholds; this injected value is
    // not presented as independently measured numerical accuracy of the equilibrium solution.
    result.residual_norms[0] = 5e-6;
    if let Some(energy) = energy {
        // Controlled accounting evidence, not a claim about the fixture's actual heating.
        result.energy_balances.fill(energy);
    }
    ThermalNumericalReceiptArtifact::new(&workload, &result).unwrap()
}

#[test]
fn conservation_accuracy_does_not_improve_when_power_units_shrink() {
    for exponent in [-100, -40, 0, 40, 100] {
        let power = 2.0_f64.powi(exponent);
        let fraction = 2.0_f64.powi(-20);
        let balance = crate::contracts::ThermalEnergyBalance::from_powers(
            runmat_types::physical::Power::new(power),
            runmat_types::physical::Power::new(0.0),
            runmat_types::physical::Power::new(power * (1.0 - fraction)),
            runmat_types::physical::Power::new(0.0),
        );
        let receipt = receipt_with_energy(ElementOrder::Tet4, Some(balance));
        let source = AccuracySourceReceipt::from_canonical_bytes(
            AccuracySourceKind::Thermal,
            &receipt.canonical_bytes().unwrap(),
        )
        .unwrap();
        let request = request(
            &receipt,
            vec![criterion(
                "conservation",
                RequiredAccuracyEvidence::ConservationBalance,
                fraction / 2.0,
            )],
        );
        let evidence = source.evidence(&request).unwrap();
        assert_eq!(evidence.len(), 1);
        assert_eq!(evidence[0].measured, AccuracyMeasure::Relative(fraction));
        let evaluated = evaluate_accuracy(&request.plan, &evidence).unwrap();
        assert_eq!(evaluated.report().status, AccuracyStatus::NotSatisfied);
        assert_eq!(
            publication_gate(&request.plan, &evaluated)
                .unwrap()
                .decision,
            PublicationDecision::Block
        );
    }
}

fn criterion(id: &str, evidence: RequiredAccuracyEvidence, tolerance: f64) -> AccuracyCriterion {
    AccuracyCriterion {
        criterion_id: id.into(),
        target: AccuracyTarget::GlobalEquilibrium,
        contribution: ErrorContributionKind::Algebraic,
        tolerance: AccuracyTolerance::Relative(tolerance),
        evidence,
        mandatory: true,
    }
}

fn request(
    receipt: &ThermalNumericalReceiptArtifact,
    criteria: Vec<AccuracyCriterion>,
) -> AccuracyEvaluationRequest {
    let mut plan = AccuracyPlan::baseline("thermal_accuracy");
    plan.criteria = criteria;
    AccuracyEvaluationRequest::new(
        receipt.receipt().producer,
        AccuracySourceKind::Thermal,
        receipt.source_result(),
        None,
        plan,
    )
    .unwrap()
}

#[test]
fn thermal_receipt_adapter_preserves_identity_and_supported_residual_evidence() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let receipt = receipt(order);
        let source = AccuracySourceReceipt::from_canonical_bytes(
            AccuracySourceKind::Thermal,
            &receipt.canonical_bytes().unwrap(),
        )
        .unwrap();
        assert_eq!(source, AccuracySourceReceipt::Thermal(receipt.clone()));
        assert_eq!(source.identity(), receipt.identity());
        assert_eq!(source.source_workload(), receipt.source_workload);
        assert_eq!(source.source_result(), receipt.source_result());
        let request = request(
            &receipt,
            vec![
                criterion("linear", RequiredAccuracyEvidence::ResidualBound, 1e-5),
                criterion(
                    "energy",
                    RequiredAccuracyEvidence::ConservationBalance,
                    1e-4,
                ),
            ],
        );
        let evidence = source.evidence(&request).unwrap();
        assert_eq!(evidence.len(), 2);
        for item in &evidence {
            assert_eq!(
                item.scope,
                AccuracyEvidenceScope::SolverResult(receipt.source_result())
            );
            let expected = match item.kind {
                RequiredAccuracyEvidence::ResidualBound => {
                    receipt.receipt().maximum_relative_linear_residual
                }
                RequiredAccuracyEvidence::ConservationBalance => {
                    receipt.receipt().maximum_relative_energy_residual
                }
                _ => panic!("unexpected thermal evidence"),
            };
            assert_eq!(item.measured, AccuracyMeasure::Relative(expected));
        }
        let evaluated = evaluate_accuracy(&request.plan, &evidence).unwrap();
        assert_eq!(evaluated.report().status, AccuracyStatus::Satisfied);
        assert_eq!(
            publication_gate(&request.plan, &evaluated)
                .unwrap()
                .decision,
            PublicationDecision::Allow
        );
        let mut stricter = request.plan.clone();
        stricter.criteria[0].tolerance = AccuracyTolerance::Relative(1e-6);
        let evaluated = evaluate_accuracy(&stricter, &evidence).unwrap();
        assert_eq!(evaluated.report().status, AccuracyStatus::NotSatisfied);
        assert_eq!(
            publication_gate(&stricter, &evaluated).unwrap().decision,
            PublicationDecision::Block
        );
    }
}

#[test]
fn thermal_receipt_cannot_supply_discretization_or_reference_accuracy() {
    let receipt = receipt(ElementOrder::Tet4);
    let source = AccuracySourceReceipt::Thermal(receipt.clone());
    for (contribution, kind) in [
        (
            ErrorContributionKind::SpatialDiscretization,
            RequiredAccuracyEvidence::ConvergenceStudy,
        ),
        (
            ErrorContributionKind::Temporal,
            RequiredAccuracyEvidence::ConvergenceStudy,
        ),
        (
            ErrorContributionKind::Algebraic,
            RequiredAccuracyEvidence::ReferenceComparison,
        ),
    ] {
        let mut criterion = criterion("unsupported", kind, 1e-3);
        criterion.contribution = contribution;
        let request = request(&receipt, vec![criterion]);
        let evidence = source.evidence(&request).unwrap();
        assert!(evidence.is_empty());
        let evaluated = evaluate_accuracy(&request.plan, &evidence).unwrap();
        assert_eq!(evaluated.report().status, AccuracyStatus::Indeterminate);
        assert_eq!(
            publication_gate(&request.plan, &evaluated)
                .unwrap()
                .decision,
            PublicationDecision::Block
        );
    }
}

#[test]
fn thermal_receipt_rejects_different_source_kind_result_and_producer() {
    let receipt = receipt(ElementOrder::Tet4);
    let source = AccuracySourceReceipt::Thermal(receipt.clone());
    let plan = request(
        &receipt,
        vec![criterion(
            "linear",
            RequiredAccuracyEvidence::ResidualBound,
            1e-5,
        )],
    )
    .plan;
    let other_step = ProcedureStep::thermal_transient("different");
    let other_sequence = AnalysisSequence::from_steps("different", vec![other_step.clone()]);
    let other_producer = StepIdentity::from_solver_contract(
        SequenceIdentity::from_solver_contract(&other_sequence).unwrap(),
        &other_step,
    )
    .unwrap();
    let other_result =
        SolverResultIdentity::new(receipt.source_workload, Digest::sha256("different result"))
            .unwrap();
    for (kind, result, producer) in [
        (
            AccuracySourceKind::StructuralLinear,
            receipt.source_result(),
            receipt.receipt().producer,
        ),
        (
            AccuracySourceKind::Thermal,
            other_result,
            receipt.receipt().producer,
        ),
        (
            AccuracySourceKind::Thermal,
            receipt.source_result(),
            other_producer,
        ),
    ] {
        let request =
            AccuracyEvaluationRequest::new(producer, kind, result, None, plan.clone()).unwrap();
        assert!(source.evidence(&request).is_err());
    }
    assert!(AccuracySourceReceipt::from_canonical_bytes(
        AccuracySourceKind::StructuralLinear,
        &receipt.canonical_bytes().unwrap()
    )
    .is_err());
}
