use super::*;
use crate::solve::nonlinear::{solid::continuum_qualification::fixture, NonlinearSolveOptions};

#[test]
fn cold_blocks_bind_original_history_predecessor_and_output_order() {
    for quadratic in [false, true] {
        let (model, mesh, _) = fixture::prepare(1, quadratic);
        let workload = NonlinearSolidWorkload {
            model,
            controls: NonlinearSolveOptions {
                increment_count: 2,
                max_accepted_increments: 8,
                max_newton_iters: 32,
                tolerance: 1e-9,
                linear_tolerance: 1e-12,
                max_line_search_backtracks: 16,
                ..NonlinearSolveOptions::default()
            },
        };
        let topology = Digest::from_bytes(*mesh.canonical_digest.bytes());
        let request = NonlinearSolidBlockRequest {
            workload: workload.identity().unwrap(),
            topology,
            block_ordinal: 0,
            first_output_ordinal: 1,
            accepted_block_limit: 1,
            predecessor: None,
        };
        let first = workload
            .execute_block(mesh.clone(), &request, None)
            .unwrap();
        let receipt = NonlinearSolidNumericalReceiptArtifact::new(&workload, &first).unwrap();
        let receipt_bytes = receipt.canonical_bytes().unwrap();
        assert!(receipt_bytes.len() as u64 <= NONLINEAR_SOLID_RECEIPT_MAX_ENCODED_BYTES);
        let cold_receipt =
            NonlinearSolidNumericalReceiptArtifact::from_canonical_bytes(&receipt_bytes).unwrap();
        cold_receipt.validate_against(&workload, &first).unwrap();
        assert!(
            cold_receipt.receipt().accepted_relative_increment_maximum
                <= workload.controls.tolerance
        );
        assert_eq!(
            cold_receipt.receipt().relative_increment_tolerance,
            workload.controls.tolerance
        );
        assert!(
            first.checkpoint.work().linear_solves() > first.checkpoint.work().newton_iterations()
        );
        assert_eq!(
            cold_receipt.receipt().cumulative_work,
            first.checkpoint.work()
        );
        assert!(!first.complete);
        let original_state = first.result.accepted_state.clone();
        let reference = NonlinearSolidBlockReference {
            artifact_digest: first.identity,
            controller_digest: first.checkpoint.content_digest(),
        };
        let bytes = first.canonical_bytes().unwrap();
        drop(first);
        let cold =
            NonlinearSolidBlockArtifact::from_canonical_bytes(&bytes, reference.artifact_digest)
                .unwrap();
        assert_eq!(cold.result.accepted_state, original_state);
        let mut malformed = cold.result.clone();
        malformed.total_strain_snapshots[0].pop();
        // Construction rehashes the changed payload; rejection must come from shape,
        // not merely comparison with an obsolete digest.
        assert!(NonlinearSolidBlockArtifact::new(
            cold.request.clone(),
            crate::pipeline::nonlinear::NonlinearBlockOutput {
                result: malformed,
                checkpoint: cold.checkpoint.clone(),
                complete: cold.complete,
            }
        )
        .is_err());
        let next = NonlinearSolidBlockRequest {
            block_ordinal: 1,
            first_output_ordinal: 2,
            accepted_block_limit: 8,
            predecessor: Some(reference),
            ..request.clone()
        };
        let last = workload
            .execute_block(mesh.clone(), &next, Some(&cold))
            .unwrap();
        assert!(cold_receipt.validate_against(&workload, &last).is_err());
        assert!(last.complete);
        let whole_request = NonlinearSolidBlockRequest {
            accepted_block_limit: 8,
            ..request.clone()
        };
        let whole = workload.execute_block(mesh, &whole_request, None).unwrap();
        assert_eq!(last.result.accepted_state, whole.result.accepted_state);
        assert_eq!(
            last.result.stress_snapshots.last(),
            whole.result.stress_snapshots.last()
        );
        assert_eq!(
            last.result.total_strain_snapshots.last(),
            whole.result.total_strain_snapshots.last()
        );
        for substitution in 0..4 {
            let mut altered = next.clone();
            match substitution {
                0 => altered.first_output_ordinal += 1,
                1 => altered.block_ordinal += 1,
                2 => {
                    altered.predecessor.as_mut().unwrap().controller_digest =
                        Digest::sha256(b"foreign")
                }
                _ => {
                    altered.predecessor.as_mut().unwrap().artifact_digest =
                        Digest::sha256(b"foreign")
                }
            }
            assert!(altered
                .validate_for(&workload, topology, Some(&cold))
                .is_err());
        }
        assert!(next.validate_for(&workload, topology, None).is_err());
        assert!(NonlinearSolidBlockArtifact::from_canonical_bytes(
            &bytes,
            Digest::sha256(b"foreign")
        )
        .is_err());
        let mut damaged = cold.clone();
        damaged.result.stress_snapshots[0][0] += 1.0;
        assert!(damaged.validate().is_err());
        let mut skipped = cold;
        skipped.result.load_factors.clear();
        assert!(skipped.validate().is_err());
    }
}

#[test]
fn original_recovery_rejects_rehashed_terminal_projection_substitutions() {
    for quadratic in [false, true] {
        let (model, mesh, _) = fixture::prepare(1, quadratic);
        let workload = NonlinearSolidWorkload {
            model,
            controls: NonlinearSolveOptions {
                increment_count: 1,
                max_accepted_increments: 8,
                tolerance: 1e-9,
                linear_tolerance: 1e-12,
                ..NonlinearSolveOptions::default()
            },
        };
        let request = NonlinearSolidBlockRequest::initial(
            &workload,
            Digest::from_bytes(*mesh.canonical_digest.bytes()),
            8,
        )
        .unwrap();
        let original = workload
            .execute_block(mesh.clone(), &request, None)
            .unwrap();
        crate::pipeline::nonlinear::recover_original_nonlinear_block(
            &workload, &mesh, &original, 0,
        )
        .unwrap();
        for field in 0..6 {
            let mut result = original.result.clone();
            let values = match field {
                0 => &mut result.internal_force_snapshots,
                1 => &mut result.stress_snapshots,
                2 => &mut result.total_strain_snapshots,
                3 => &mut result.plastic_strain_snapshots,
                4 => &mut result.equivalent_plastic_strain_snapshots,
                5 => &mut result.von_mises_snapshots,
                _ => unreachable!(),
            };
            values.last_mut().unwrap()[0] += 1e-3;
            let altered = NonlinearSolidBlockArtifact::new(
                request.clone(),
                crate::pipeline::nonlinear::NonlinearBlockOutput {
                    result,
                    checkpoint: original.checkpoint.clone(),
                    complete: original.complete,
                },
            )
            .unwrap();
            altered.validate().unwrap();
            let error = crate::pipeline::nonlinear::recover_original_nonlinear_block(
                &workload, &mesh, &altered, 0,
            )
            .unwrap_err();
            assert!(
                error.to_string().contains("terminal projection differs"),
                "{error}"
            );
        }
    }
}

// witness: FEA-NONLINEAR-GOVERNANCE-001/inequality_restart_converged_base @
//   nonlinear_structural/tet4/numerical_kernel; coverage: partial;
//   boundary: The default structural-nonlinear accuracy plan (mandatory algebraic global-equilibrium
//   residual bound) is decided by the achieved tangent-solve residual the receipt retains: allowed
//   at the authored target, refused (not indeterminate) when the target is below the achieved value;
//   one slab block, not contact or restart.
#[test]
fn default_nonlinear_plan_is_decided_by_achieved_tangent_solve_residuals() {
    use crate::execution::{AccuracyEvaluationRequest, AccuracySourceKind, AccuracySourceReceipt};
    use crate::results::accuracy::{
        evaluate_accuracy_artifact, AccuracyStatus, PublicationDecision,
    };
    use runmat_analysis_core::{AccuracyPlan, AccuracyTolerance};
    let (mut model, mesh, _) = fixture::prepare(1, false);
    model.sequence.steps[0].accuracy = AccuracyPlan::baseline("load.accuracy");
    let workload = NonlinearSolidWorkload {
        model,
        controls: NonlinearSolveOptions {
            increment_count: 2,
            max_accepted_increments: 8,
            max_newton_iters: 32,
            tolerance: 1e-9,
            linear_tolerance: 1e-12,
            max_line_search_backtracks: 16,
            ..NonlinearSolveOptions::default()
        },
    };
    let request = NonlinearSolidBlockRequest {
        workload: workload.identity().unwrap(),
        topology: Digest::from_bytes(*mesh.canonical_digest.bytes()),
        block_ordinal: 0,
        first_output_ordinal: 1,
        accepted_block_limit: 8,
        predecessor: None,
    };
    let block = workload.execute_block(mesh, &request, None).unwrap();
    assert!(block.complete);
    let receipt = NonlinearSolidNumericalReceiptArtifact::from_canonical_bytes(
        &NonlinearSolidNumericalReceiptArtifact::new(&workload, &block)
            .unwrap()
            .canonical_bytes()
            .unwrap(),
    )
    .unwrap();
    let achieved = receipt.receipt().maximum_linear_relative_residual;
    assert!(achieved.is_finite() && achieved <= workload.controls.linear_tolerance);
    assert_eq!(
        achieved,
        block.checkpoint.maximum_linear_relative_residual()
    );
    let evaluate = |plan: AccuracyPlan| {
        let request = AccuracyEvaluationRequest::new(
            workload.producer_identity().unwrap(),
            AccuracySourceKind::NonlinearSolid,
            receipt.source_result(),
            None,
            plan,
        )
        .unwrap();
        let source = AccuracySourceReceipt::NonlinearSolid(receipt.clone());
        evaluate_accuracy_artifact(&request, source.evidence(&request).unwrap()).unwrap()
    };
    let authored = workload.model.sequence.steps[0].accuracy.clone();
    let allowed = evaluate(authored.clone());
    assert_eq!(allowed.gate().decision, PublicationDecision::Allow);
    assert_eq!(allowed.gate().status, AccuracyStatus::Satisfied);
    // Unavailable mandatory evidence still blocks: the J2 receipt carries no spatial
    // convergence study, so such a criterion is indeterminate and the default policy refuses.
    let mut unavailable = authored.clone();
    unavailable
        .criteria
        .push(runmat_analysis_core::AccuracyCriterion {
            criterion_id: "spatial_convergence".into(),
            target: runmat_analysis_core::AccuracyTarget::NumericalResult,
            contribution: runmat_analysis_core::ErrorContributionKind::SpatialDiscretization,
            tolerance: AccuracyTolerance::Relative(0.05),
            evidence: runmat_analysis_core::RequiredAccuracyEvidence::ConvergenceStudy,
            mandatory: true,
        });
    let blocked = evaluate(unavailable.clone());
    assert_eq!(blocked.gate().status, AccuracyStatus::Indeterminate);
    assert_eq!(blocked.gate().decision, PublicationDecision::Block);
    unavailable.unavailable_evidence =
        runmat_analysis_core::UnavailableEvidencePolicy::MarkIndeterminate;
    let labeled = evaluate(unavailable);
    assert_eq!(labeled.gate().status, AccuracyStatus::Indeterminate);
    assert_eq!(labeled.gate().decision, PublicationDecision::Allow);
    let mut unreachable = authored;
    unreachable.criteria[0].tolerance = AccuracyTolerance::Relative(achieved / 2.0);
    if achieved > 0.0 {
        let refused = evaluate(unreachable);
        assert_eq!(refused.gate().decision, PublicationDecision::Block);
        assert_eq!(refused.gate().status, AccuracyStatus::NotSatisfied);
    }
}
