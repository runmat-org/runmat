use super::*;
use crate::{
    execution::{AccuracyEvaluationRequest, AccuracySourceKind, AccuracySourceReceipt},
    results::accuracy::{evaluate_accuracy, AccuracyMeasure, AccuracyStatus},
    topology::test_meshes::{quadratic_slab_mesh, slab_mesh},
};
use runmat_analysis_core::{
    AccuracyCriterion, AccuracyPlan, AccuracyTarget, AccuracyTolerance, ErrorContributionKind,
    RequiredAccuracyEvidence,
};

#[test]
fn acoustic_receipt_roundtrip_binds_exact_workload_result_and_mesh() {
    for mesh in [slab_mesh(2), quadratic_slab_mesh(2)] {
        let workload = super::super::tests::workload();
        let result = workload.execute(&mesh).unwrap();
        let receipt = AcousticNumericalReceiptArtifact::new(&workload, &result, &mesh).unwrap();
        let bytes = receipt.canonical_bytes().unwrap();
        assert!(bytes.len() as u64 <= ACOUSTIC_RECEIPT_MAX_ENCODED_BYTES);
        let cold = AcousticNumericalReceiptArtifact::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(receipt, cold);
        cold.validate_against(&workload, &result, &mesh).unwrap();
        assert_eq!(cold.receipt().relative_residual, result.relative_residual());
        assert_eq!(
            cold.receipt().power_relative_error,
            result.power_relative_error()
        );
        let mut foreign = workload.clone();
        foreign.power_balance_relative_tolerance *= 0.5;
        assert!(cold.validate_against(&foreign, &result, &mesh).is_err());
        let other = foreign.execute(&mesh).unwrap();
        assert!(cold.validate_against(&foreign, &other, &mesh).is_err());
        let mut other_mesh = mesh.clone();
        {
            let quantity_change = 0.1;
            let quantity_index = 0;
            let quantity_slot = &mut other_mesh.topology.nodes[0].coordinates_m;
            let mut quantity_coordinates = quantity_slot.canonical_value();
            quantity_coordinates[quantity_index] += quantity_change;
            *quantity_slot = runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(quantity_coordinates);
        };
        other_mesh.seal_canonical_digest().unwrap();
        assert!(cold
            .validate_against(&workload, &result, &other_mesh)
            .is_err());
        for field in [
            "relative_residual",
            "power_relative_error",
            "frequency_hz",
            "solver_relative_tolerance",
        ] {
            let mut tampered = serde_json::to_value(&cold).unwrap();
            tampered["receipt"][field] = serde_json::json!(0.5);
            assert!(serde_json::from_value::<AcousticNumericalReceiptArtifact>(tampered).is_err());
        }
        let mut tampered = serde_json::to_value(&cold).unwrap();
        tampered["receipt"]["unknown_evidence"] = serde_json::json!(0.0);
        assert!(serde_json::from_value::<AcousticNumericalReceiptArtifact>(tampered).is_err());
        assert!(AcousticNumericalReceiptArtifact::from_canonical_bytes(
            &result.canonical_bytes().unwrap()
        )
        .is_err());
    }
}

#[test]
fn acoustic_accuracy_receipt_supplies_only_algebraic_closure_and_rejects_foreign_producer() {
    let mesh = slab_mesh(2);
    let workload = super::super::tests::workload();
    let result = workload.execute(&mesh).unwrap();
    let receipt = AcousticNumericalReceiptArtifact::new(&workload, &result, &mesh).unwrap();
    let source = AccuracySourceReceipt::from_canonical_bytes(
        AccuracySourceKind::AcousticHarmonic,
        &receipt.canonical_bytes().unwrap(),
    )
    .unwrap();
    assert_eq!(source.identity(), receipt.identity());
    assert_eq!(source.source_workload(), receipt.source_workload());
    let mut plan = AccuracyPlan::baseline("acoustic_accuracy");
    plan.criteria = [
        RequiredAccuracyEvidence::ResidualBound,
        RequiredAccuracyEvidence::ConservationBalance,
        RequiredAccuracyEvidence::ConvergenceStudy,
    ]
    .into_iter()
    .enumerate()
    .map(|(i, evidence)| AccuracyCriterion {
        criterion_id: format!("criterion_{i}"),
        target: AccuracyTarget::GlobalEquilibrium,
        contribution: ErrorContributionKind::Algebraic,
        tolerance: AccuracyTolerance::Relative(1e-7),
        evidence,
        mandatory: true,
    })
    .collect();
    plan.criteria.push(AccuracyCriterion {
        criterion_id: "spatial_field_accuracy".into(),
        target: AccuracyTarget::NumericalResult,
        contribution: ErrorContributionKind::SpatialDiscretization,
        tolerance: AccuracyTolerance::Relative(1e-7),
        evidence: RequiredAccuracyEvidence::ConvergenceStudy,
        mandatory: true,
    });
    let request = AccuracyEvaluationRequest::new(
        receipt.receipt().producer,
        AccuracySourceKind::AcousticHarmonic,
        receipt.source_result(),
        None,
        plan,
    )
    .unwrap();
    let evidence = source.evidence(&request).unwrap();
    assert_eq!(evidence.len(), 2);
    for e in &evidence {
        assert_eq!(
            e.measured,
            AccuracyMeasure::Relative(match e.kind {
                RequiredAccuracyEvidence::ResidualBound => result.relative_residual(),
                RequiredAccuracyEvidence::ConservationBalance => result.power_relative_error(),
                _ => panic!("fabricated convergence evidence"),
            })
        );
    }
    assert_ne!(
        evaluate_accuracy(&request.plan, &evidence)
            .unwrap()
            .report()
            .status,
        AccuracyStatus::Satisfied
    );
    let mut foreign_workload = workload.clone();
    foreign_workload.model.sequence.steps[0].step_id = ProcedureStepId("foreign".into());
    foreign_workload.step_id = ProcedureStepId("foreign".into());
    let foreign = AccuracyEvaluationRequest::new(
        foreign_workload.producer_identity().unwrap(),
        request.source_kind,
        request.source_result,
        None,
        request.plan.clone(),
    )
    .unwrap();
    assert!(source.evidence(&foreign).is_err());
    assert!(AccuracySourceReceipt::from_canonical_bytes(
        AccuracySourceKind::Thermal,
        &receipt.canonical_bytes().unwrap()
    )
    .is_err());
}
