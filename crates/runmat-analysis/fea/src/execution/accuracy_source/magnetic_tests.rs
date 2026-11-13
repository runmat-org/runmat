use super::*;
use crate::{
    execution::MagnetostaticWorkload,
    physics::magnetic::test_support::authored_model,
    results::accuracy::{evaluate_accuracy, publication_gate, AccuracyStatus, PublicationDecision},
    topology::test_meshes::{quadratic_slab_mesh, slab_mesh},
};
use runmat_analysis_core::{
    AccuracyCriterion, AccuracyPlan, AccuracyTarget, AccuracyTolerance, ProcedureStepId,
};
use runmat_meshing_core::ElementOrder;

fn plan() -> AccuracyPlan {
    let mut plan = AccuracyPlan::baseline("magnetic_accuracy");
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
    plan
}

// witness: FEA-EM-QUALIFICATION-001/accuracy @ magnetostatic/tet4|tet10/public_study;
//   boundary: Both affine geometry orders/signs receipts preserve original algebraic
//   residual/gauge/energy claims and block production when independent convergence evidence is
//   missing; no process or distributed proof.
#[test]
fn magnetic_receipts_supply_only_original_algebraic_equilibrium_evidence() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = match order {
            ElementOrder::Tet4 => slab_mesh(2),
            ElementOrder::Tet10 => quadratic_slab_mesh(2),
        };
        for current in [-2.0, 0.0, 2.0] {
            let workload = MagnetostaticWorkload {
                model: authored_model(current),
                step_id: ProcedureStepId("potential".into()),
                relative_residual_target: 1e-10,
            };
            let result = workload.execute(&mesh).unwrap();
            let receipt = MagneticNumericalReceiptArtifact::new(&workload, &result, &mesh).unwrap();
            let bytes = receipt.canonical_bytes().unwrap();
            // Decode the original immutable receipt; no recovered solve/evidence is substituted.
            let source = AccuracySourceReceipt::from_canonical_bytes(
                AccuracySourceKind::Magnetostatic,
                &bytes,
            )
            .unwrap();
            assert_eq!(
                source,
                AccuracySourceReceipt::Magnetostatic(receipt.clone())
            );
            assert_eq!(source.identity(), receipt.identity());
            assert_eq!(source.source_workload(), workload.identity().unwrap());
            assert_eq!(
                source.source_result(),
                result.source_result_identity().unwrap()
            );
            let request = AccuracyEvaluationRequest::new(
                receipt.receipt().producer,
                AccuracySourceKind::Magnetostatic,
                receipt.source_result(),
                None,
                plan(),
            )
            .unwrap();
            let evidence = source.evidence(&request).unwrap();
            assert_eq!(evidence.len(), 2);
            for item in &evidence {
                assert_eq!(item.target, AccuracyTarget::GlobalEquilibrium);
                assert_eq!(item.contribution, ErrorContributionKind::Algebraic);
                assert_eq!(
                    item.scope,
                    AccuracyEvidenceScope::SolverResult(receipt.source_result())
                );
                let expected = match item.kind {
                    RequiredAccuracyEvidence::ResidualBound => receipt
                        .receipt()
                        .relative_physical_residual
                        .max(receipt.receipt().relative_gauge_residual),
                    RequiredAccuracyEvidence::ConservationBalance => {
                        receipt.receipt().relative_energy_work_error
                    }
                    _ => panic!("magnetic receipt manufactured unsupported evidence"),
                };
                assert_eq!(item.measured, AccuracyMeasure::Relative(expected));
                assert!(item
                    .assumptions
                    .iter()
                    .any(|s| s.contains("not independent field accuracy")));
                if current == 0.0 {
                    assert_eq!(expected, 0.0);
                }
            }
            let evaluated = evaluate_accuracy(&request.plan, &evidence).unwrap();
            assert_eq!(evaluated.report().status, AccuracyStatus::Indeterminate);
            for contribution in &evaluated.report().contributions {
                let expected =
                    if matches!(contribution.criterion_id.as_str(), "residual" | "closure") {
                        AccuracyStatus::Satisfied
                    } else {
                        AccuracyStatus::Indeterminate
                    };
                assert_eq!(contribution.status, expected);
            }
            assert_eq!(
                publication_gate(&request.plan, &evaluated)
                    .unwrap()
                    .decision,
                PublicationDecision::Block
            );
            // The two algebraic criteria can be satisfied; this never qualifies spatial accuracy.
            let mut algebraic = request.plan.clone();
            algebraic.criteria.truncate(2);
            let evaluated = evaluate_accuracy(&algebraic, &evidence).unwrap();
            assert_eq!(evaluated.report().status, AccuracyStatus::Satisfied);
            let mut wrong_target = request.plan.clone();
            for criterion in &mut wrong_target.criteria {
                criterion.target = AccuracyTarget::NumericalResult;
            }
            let wrong_target = AccuracyEvaluationRequest::new(
                request.producer,
                request.source_kind,
                request.source_result,
                None,
                wrong_target,
            )
            .unwrap();
            assert!(source.evidence(&wrong_target).unwrap().is_empty());
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
            let foreign_result = foreign.execute(&mesh).unwrap();
            let foreign_source = AccuracySourceReceipt::Magnetostatic(
                MagneticNumericalReceiptArtifact::new(&foreign, &foreign_result, &mesh).unwrap(),
            );
            assert!(foreign_source.evidence(&request).is_err());
            assert!(AccuracySourceReceipt::from_canonical_bytes(
                AccuracySourceKind::ElectricalStatic,
                &bytes
            )
            .is_err());
            assert!(AccuracySourceReceipt::from_canonical_bytes(
                AccuracySourceKind::AcousticHarmonic,
                &bytes
            )
            .is_err());
        }
    }
}
