use super::*;
use crate::{
    execution::ElectricalStaticWorkload,
    physics::electrical::tests::{mesh, model, prescribe},
    results::accuracy::{evaluate_accuracy, AccuracyStatus},
};
use runmat_analysis_core::{
    AccuracyCriterion, AccuracyPlan, AccuracyTarget, AccuracyTolerance, ProcedureStepId,
    RegionTarget, StaticElectricalEquation,
};
use runmat_meshing_core::ElementOrder;

// witness: FEA-EM-QUALIFICATION-001/accuracy @ electrostatic/tet4|tet10/public_study,
//   electrical_conduction/tet4|tet10/public_study;
//   boundary: Both equations/orders retained receipts provide only algebraic residual/closure
//   evidence; independent spatial/convergence obligations remain indeterminate. Not field-accuracy
//   qualification.
#[test]
fn electrical_receipts_supply_only_retained_algebraic_equilibrium_evidence() {
    for equation in [
        StaticElectricalEquation::Electrostatics,
        StaticElectricalEquation::SteadyConduction,
    ] {
        for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
            let mut model = model(equation);
            prescribe(
                &mut model,
                "left".into(),
                RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_left")),
                0.0,
            );
            prescribe(
                &mut model,
                "right".into(),
                RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_right")),
                2.0,
            );
            let workload = ElectricalStaticWorkload {
                model,
                step_id: ProcedureStepId("potential".into()),
                relative_residual_target: 1e-11,
                element_temperature_k: None,
                operating_time_s: None,
            };
            let mesh = mesh(order, 2);
            let result = workload.execute(&mesh).unwrap();
            let receipt =
                ElectricalNumericalReceiptArtifact::new(&workload, &result, &mesh).unwrap();
            let bytes = receipt.canonical_bytes().unwrap();
            let source = AccuracySourceReceipt::from_canonical_bytes(
                AccuracySourceKind::ElectricalStatic,
                &bytes,
            )
            .unwrap();
            assert_eq!(
                source,
                AccuracySourceReceipt::ElectricalStatic(receipt.clone())
            );
            assert_eq!(source.source_workload(), workload.identity().unwrap());
            assert_eq!(
                source.source_result(),
                result.source_result_identity().unwrap()
            );
            let mut plan = AccuracyPlan::baseline("electrical_accuracy");
            plan.criteria = [
                (
                    "residual",
                    ErrorContributionKind::Algebraic,
                    RequiredAccuracyEvidence::ResidualBound,
                ),
                (
                    "closure",
                    ErrorContributionKind::Algebraic,
                    RequiredAccuracyEvidence::ConservationBalance,
                ),
                (
                    "spatial",
                    ErrorContributionKind::SpatialDiscretization,
                    RequiredAccuracyEvidence::ConvergenceStudy,
                ),
                (
                    "convergence",
                    ErrorContributionKind::Algebraic,
                    RequiredAccuracyEvidence::ConvergenceStudy,
                ),
            ]
            .into_iter()
            .map(|(id, contribution, evidence)| AccuracyCriterion {
                criterion_id: id.into(),
                target: AccuracyTarget::GlobalEquilibrium,
                contribution,
                evidence,
                tolerance: AccuracyTolerance::Relative(1e-7),
                mandatory: true,
            })
            .collect();
            let request = AccuracyEvaluationRequest::new(
                receipt.receipt().producer,
                AccuracySourceKind::ElectricalStatic,
                receipt.source_result(),
                None,
                plan,
            )
            .unwrap();
            let evidence = source.evidence(&request).unwrap();
            assert_eq!(evidence.len(), 2);
            for item in &evidence {
                assert_eq!(
                    item.scope,
                    AccuracyEvidenceScope::SolverResult(receipt.source_result())
                );
                let expected = match item.kind {
                    RequiredAccuracyEvidence::ResidualBound => {
                        receipt.receipt().linear_relative_residual
                    }
                    RequiredAccuracyEvidence::ConservationBalance => receipt
                        .receipt()
                        .relative_flux_imbalance
                        .max(receipt.receipt().relative_energy_identity_error),
                    _ => panic!("electrical receipt manufactured unsupported evidence"),
                };
                assert_eq!(item.measured, AccuracyMeasure::Relative(expected));
            }
            assert_ne!(
                evaluate_accuracy(&request.plan, &evidence)
                    .unwrap()
                    .report()
                    .status,
                AccuracyStatus::Satisfied
            );
            let mut foreign = workload.clone();
            foreign.model.sequence.sequence_id = "foreign".into();
            let foreign_request = AccuracyEvaluationRequest::new(
                foreign.producer_identity().unwrap(),
                request.source_kind,
                request.source_result,
                None,
                request.plan.clone(),
            )
            .unwrap();
            assert!(source.evidence(&foreign_request).is_err());
            assert!(AccuracySourceReceipt::from_canonical_bytes(
                AccuracySourceKind::AcousticHarmonic,
                &bytes
            )
            .is_err());
        }
    }
}
