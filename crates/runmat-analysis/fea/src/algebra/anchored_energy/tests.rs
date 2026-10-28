use super::*;
use crate::algebra::{MatrixProperties, OperatorRole, RealSparseMatrix};

fn matrix(values: &[f64], n: usize) -> RealSparseMatrix {
    RealSparseMatrix::from_dense(
        values,
        n,
        OperatorRole::Stiffness,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    )
    .unwrap()
}
fn check<'a>(matrix: &'a RealSparseMatrix, fixed: &'a [bool]) -> AnchoredEnergyCheck<'a> {
    AnchoredEnergyCheck::new(
        matrix,
        fixed,
        CoercivityWorkEnvelope::derive(matrix.pattern()).unwrap(),
    )
    .unwrap()
}

#[test]
fn eliminated_anchor_and_permuted_selection_match_independent_chain_eigenvalue() {
    // Full vertices [ground, near, far], actual CSR order [far, near].
    let matrix = matrix(&[1., -1., -1., 2.], 2);
    let selection = [None, Some(1), Some(0)];
    let mut proof = AnchoredEnergyCheck::new_selection(
        &matrix,
        &selection,
        CoercivityWorkEnvelope::derive(matrix.pattern()).unwrap(),
    )
    .unwrap();
    for nodes in [[0, 1], [1, 2]] {
        proof.push_block(&nodes, &[1., -1., -1., 1.]).unwrap();
    }
    let EnergyVerdict::Verified(proof) = proof.finish().unwrap() else {
        panic!("grounded chain")
    };
    assert!(proof.lower_bound() > 0.0);
    assert!(proof.lower_bound() < (3.0 - 5.0_f64.sqrt()) / 2.0);
    assert!(proof.assembly_error_upper() < 1e-12);
}

#[test]
fn incomplete_duplicate_and_out_of_range_selection_reject() {
    let matrix = matrix(&[1., 0., 0., 1.], 2);
    for selection in [
        vec![None, Some(0)],
        vec![None, Some(0), Some(0)],
        vec![None, Some(0), Some(2)],
    ] {
        assert!(matches!(
            AnchoredEnergyCheck::new_selection(
                &matrix,
                &selection,
                CoercivityWorkEnvelope::derive(matrix.pattern()).unwrap()
            ),
            Err(EnergyError::Shape)
        ));
    }
}

#[test]
fn reduced_operator_corruption_cannot_borrow_eliminated_anchor_proof() {
    let matrix = matrix(&[0.1, -1., -1., 2.], 2);
    let selection = [None, Some(1), Some(0)];
    let mut proof = AnchoredEnergyCheck::new_selection(
        &matrix,
        &selection,
        CoercivityWorkEnvelope::derive(matrix.pattern()).unwrap(),
    )
    .unwrap();
    for nodes in [[0, 1], [1, 2]] {
        proof.push_block(&nodes, &[1., -1., -1., 1.]).unwrap();
    }
    assert!(matches!(
        proof.finish().unwrap(),
        EnergyVerdict::Indeterminate { .. }
    ));
}
#[test]
fn anchored_chain_bound_is_below_independent_exact_smallest_eigenvalue() {
    let matrix = matrix(&[1., 0., 0., 0., 2., -1., 0., -1., 1.], 3);
    let mut proof = check(&matrix, &[true, false, false]);
    proof.push_block(&[0, 1], &[1., -1., -1., 1.]).unwrap();
    proof.push_block(&[1, 2], &[1., -1., -1., 1.]).unwrap();
    let EnergyVerdict::Verified(certificate) = proof.finish().unwrap() else {
        panic!("chain should certify")
    };
    let exact = (3.0 - 5.0_f64.sqrt()) / 2.0;
    assert!(certificate.lower_bound > 0.0 && certificate.lower_bound < exact);
    assert_eq!(
        certificate.operator_identity,
        matrix.numerical_identity().unwrap()
    );
    assert!(certificate.scalar_operations > 0 && certificate.resident_bytes > 0);
    assert!(certificate.assembly_error_upper < 1e-12);
}
#[test]
fn actual_operator_corruption_cannot_be_hidden_by_local_rank_or_labels() {
    let matrix = matrix(&[1., 0., 0., 0., 0.1, -1., 0., -1., 1.], 3);
    let mut proof = check(&matrix, &[true, false, false]);
    proof.push_block(&[0, 1], &[1., -1., -1., 1.]).unwrap();
    proof.push_block(&[1, 2], &[1., -1., -1., 1.]).unwrap();
    let EnergyVerdict::Indeterminate {
        graph_lower,
        error_upper,
    } = proof.finish().unwrap()
    else {
        panic!("corruption must not certify")
    };
    assert!(error_upper > graph_lower);
}
#[test]
fn extra_local_modes_and_nonfinite_inputs_reject() {
    for q in [
        [0., 0., 0., 0.],
        [1., 1., 1., -1.],
        [1., -1., -1., f64::INFINITY],
    ] {
        assert!(!matches!(local::lower_bound(&q, 2), Ok(Some(_))));
    }
    assert!(local::lower_bound(&[1., -1., -1., f64::INFINITY], 2).is_err());
    // Nonpositive local modes are uncertified, not invalid: the global proof is undecided.
    assert_eq!(local::lower_bound(&[0., 0., 0., 0.], 2), Ok(None));
    assert_eq!(local::lower_bound(&[1., 1., 1., -1.], 2), Ok(None));
    assert_eq!(
        local::lower_bound(&[2., -1., -1., -1., 1., 1., -1., 1., 1.], 3),
        Ok(None)
    );
}
#[test]
fn missing_anchor_and_uncovered_nodes_reject() {
    let m = matrix(&[1., -1., -1., 1.], 2);
    let mut proof = check(&m, &[false, false]);
    proof.push_block(&[0, 1], &[1., -1., -1., 1.]).unwrap();
    assert_eq!(proof.finish().unwrap_err(), EnergyError::Unanchored);
    let m = matrix(&[1., 0., 0., 1.], 2);
    assert_eq!(
        check(&m, &[true, false]).finish().unwrap_err(),
        EnergyError::Unanchored
    );
}
#[test]
fn duplicate_indices_and_exhausted_original_work_budget_reject() {
    let m = matrix(&[1., 0., 0., 1.], 2);
    let mut proof = check(&m, &[true, false]);
    assert_eq!(
        proof.push_block(&[0, 0], &[1., -1., -1., 1.]),
        Err(EnergyError::InvalidBlock)
    );
    assert_eq!(proof.finish().unwrap_err(), EnergyError::InvalidBlock);
    let mut proof = check(&m, &[true, false]);
    let mut failed = false;
    for _ in 0..100 {
        if matches!(
            proof.push_block(&[0, 1], &[1., -1., -1., 1.]),
            Err(EnergyError::Admission { .. })
        ) {
            failed = true;
            break;
        }
    }
    assert!(failed);
}
#[test]
fn positive_local_factor_is_verified_even_when_its_candidate_rounds() {
    let width = 4;
    let q = [
        0., 0., 0., 0., 0., 2., 0.3, 0.2, 0., 0.3, 1., 0.1, 0., 0.2, 0.1, 0.5,
    ];
    let bound = local::lower_bound(&q, width).unwrap().unwrap();
    assert!(bound > 0.0 && bound < 0.5);
    for value in [f64::MAX, f64::MIN_POSITIVE] {
        let mut broken = q;
        broken[5] = value;
        // Extreme candidates may be indeterminate; they may never return a nonfinite bound.
        if let Ok(Some(bound)) = local::lower_bound(&broken, width) {
            assert!(bound.is_finite() && bound > 0.0);
        }
    }
}

#[test]
fn cached_local_bounds_verify_full_perturbations_and_failed_stream_cannot_finish() {
    let m = matrix(&[1., 0., 0., 0., 4., 0.6, 0., 0.6, 2.], 3);
    let mut proof = check(&m, &[true, false, false]);
    let values = [3.6, -2.3, -1.3, -2.3, 2., 0.3, -1.3, 0.3, 1.];
    proof.push_block(&[0, 1, 2], &values).unwrap();
    proof.push_block(&[0, 1, 2], &values).unwrap();
    assert_eq!(proof.local_proofs.len(), 1);
    assert!(matches!(
        proof.finish().unwrap(),
        EnergyVerdict::Verified(_)
    ));
    let mut proof = check(&m, &[true, false, false]);
    proof.push_block(&[0, 1, 2], &values).unwrap();
    let mut altered = values;
    altered[4] = -2.;
    // An indefinite element energy is no disproof of the assembled operator: the sufficient
    // check stops and reports an indeterminate verdict for the factorization proof to decide.
    proof.push_block(&[0, 1, 2], &altered).unwrap();
    assert!(matches!(
        proof.finish().unwrap(),
        EnergyVerdict::Indeterminate { .. }
    ));
    let mut proof = check(&m, &[true, false, false]);
    assert_eq!(
        proof.push_block(&[0, 1, 1], &values),
        Err(EnergyError::InvalidBlock)
    );
    assert_eq!(proof.finish().unwrap_err(), EnergyError::InvalidBlock);
}

#[test]
fn declared_symmetry_does_not_hide_asymmetric_values() {
    let m = matrix(&[1., 0., 0., 0., 2., 1., 0., -1., 2.], 3);
    assert!(matches!(
        AnchoredEnergyCheck::new(
            &m,
            &[true, false, false],
            CoercivityWorkEnvelope::derive(m.pattern()).unwrap()
        ),
        Err(EnergyError::Asymmetric)
    ));
}

#[test]
fn perturbation_reuse_subtracts_verified_coefficient_error_and_rejects_rank_loss() {
    let original = [1.5, -1., -0.5, -1., 1., 0., -0.5, 0., 0.5];
    let mut perturbed = original;
    perturbed[4] = 0.95;
    let bound = local::perturbation_lower(&original, 0.4, &perturbed, 3).unwrap();
    assert!(bound > 0.34 && bound <= 0.35);
    perturbed[4] = -1.;
    assert!(local::perturbation_lower(&original, 0.4, &perturbed, 3).is_none());
}
