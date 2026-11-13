use super::*;
use runmat_simulation_usage::{UsageContext, UsageLedger};
use std::sync::Arc;

#[test]
fn anderson_reference_batches_conserve_selected_prefix_and_refusal() {
    let policy = AndersonOnePolicy::default();
    let accepted = OperationCount::from_operations(15); // n=2: 2 + 6 + 1 + 6.
    let ledger = Arc::new(UsageLedger::new(accepted, accepted).unwrap());
    let context = UsageContext::new(Arc::clone(&ledger));
    let proposal = context.run_sync(|| {
        propose(
            &[1.0, 4.0],
            &[3.0, 2.0],
            &[2.0, 6.0],
            &[1.0, 1.0],
            policy,
            false,
        )
    });
    assert!(matches!(proposal, Ok(Some(_))));
    let usage = ledger.snapshot().unwrap();
    assert_eq!(usage.completed.solve, accepted);
    assert_eq!(usage.reserved, OperationCount::ZERO);

    let degenerate = OperationCount::from_operations(8); // n=2: difference and both dots.
    let ledger = Arc::new(UsageLedger::new(degenerate, degenerate).unwrap());
    let context = UsageContext::new(Arc::clone(&ledger));
    assert_eq!(
        context.run_sync(|| propose(
            &[1.0, 2.0],
            &[0.0, 0.0],
            &[3.0, 4.0],
            &[0.0, 0.0],
            policy,
            false
        )),
        Ok(None)
    );
    assert_eq!(ledger.snapshot().unwrap().completed.solve, degenerate);

    let first_batch = OperationCount::from_operations(2);
    let ledger = Arc::new(UsageLedger::new(first_batch, first_batch).unwrap());
    let context = UsageContext::new(Arc::clone(&ledger));
    assert!(matches!(
        context.run_sync(|| propose(
            &[1.0, 4.0],
            &[3.0, 2.0],
            &[2.0, 6.0],
            &[1.0, 1.0],
            policy,
            false
        )),
        Err(FixedPointError::Accounting(
            LedgerError::BudgetExceeded { .. }
        ))
    ));
    let usage = ledger.snapshot().unwrap();
    assert_eq!(usage.completed.solve, first_batch);
    assert_eq!(usage.reserved, OperationCount::ZERO);
}

#[test]
fn affine_scalar_fixed_points_are_independently_recovered() {
    // Independent exact fixed point b/(1-a), including an unstable ordinary
    // iteration and a negative eigenvalue. No numerical solver supplies truth.
    for a in [0.9423, -0.5, 1.25, 0.0] {
        let b = 0.75;
        let x_previous = -0.4;
        let g_previous = a * x_previous + b;
        let x = g_previous;
        let g = a * x + b;
        let proposed = propose(
            &[g_previous],
            &[g_previous - x_previous],
            &[g],
            &[g - x],
            AndersonOnePolicy::default(),
            false,
        )
        .unwrap()
        .unwrap();
        let exact = b / (1.0 - a);
        assert!((proposed.image[0] - exact).abs() < 1e-12);
        assert!((a * proposed.image[0] + b - proposed.image[0]).abs() < 1e-12);
    }
}

#[test]
fn vector_proposal_uses_one_shared_least_squares_coefficient() {
    let proposal = propose(
        &[1.0, 4.0],
        &[3.0, 2.0],
        &[2.0, 6.0],
        &[1.0, 1.0],
        AndersonOnePolicy::default(),
        false,
    )
    .unwrap()
    .unwrap();
    // dr=(-2,-1), gamma=(-3)/5: an exact independent two-vector calculation.
    assert_eq!(proposal.coefficient, -0.6);
    assert_eq!(proposal.image, vec![2.6, 7.2]);
}

#[test]
fn exact_zero_residual_and_policy_boundary_are_not_modified_or_clipped() {
    let policy = AndersonOnePolicy::default();
    let stationary = propose(&[3.0], &[1.0], &[4.0], &[0.0], policy, false)
        .unwrap()
        .unwrap();
    assert_eq!(stationary.coefficient, 0.0);
    assert_eq!(stationary.image, vec![4.0]);
    let boundary = propose(&[1.0], &[31.0], &[2.0], &[32.0], policy, false)
        .unwrap()
        .unwrap();
    assert_eq!(boundary.coefficient, 32.0);
    assert_eq!(boundary.image, vec![-30.0]);
    let encoded = serde_json::to_string(&policy).unwrap();
    assert_eq!(
        serde_json::from_str::<AndersonOnePolicy>(&encoded).unwrap(),
        policy
    );
    assert!(serde_json::from_str::<AndersonOnePolicy>(
        r#"{"maximum_absolute_coefficient":32,"ignored_control":1}"#
    )
    .is_err());
}

#[test]
fn degeneracy_and_excessive_coefficients_decline_without_repair() {
    let policy = AndersonOnePolicy::default();
    assert_eq!(propose(&[], &[], &[], &[], policy, false), Ok(None));
    assert_eq!(
        propose(&[1.0], &[0.0], &[1.0], &[0.0], policy, false),
        Ok(None)
    );
    assert_eq!(
        propose(&[1.0], &[2.0], &[3.0], &[2.0], policy, false),
        Ok(None)
    );
    // gamma=100 exceeds the explicit default32 bound; do not clip it.
    assert_eq!(
        propose(&[1.0], &[99.0], &[2.0], &[100.0], policy, false),
        Ok(None)
    );
    // No dimensional denominator floor: uniformly small representable vectors
    // retain the same coefficient as the unscaled problem.
    let small = propose(&[1.0], &[3e-100], &[2.0], &[1e-100], policy, false)
        .unwrap()
        .unwrap();
    assert!((small.coefficient + 0.5).abs() < 1e-15);
}

#[test]
fn malformed_policy_inventory_arithmetic_and_cancellation_are_typed() {
    let policy = AndersonOnePolicy::default();
    for bound in [0.0, -1.0, f64::INFINITY, f64::NAN] {
        assert_eq!(
            AndersonOnePolicy {
                maximum_absolute_coefficient: bound
            }
            .validate(),
            Err(FixedPointError::InvalidPolicy)
        );
    }
    assert_eq!(
        propose(&[], &[1.0], &[1.0], &[1.0], policy, false),
        Err(FixedPointError::InvalidInventory)
    );
    assert_eq!(
        propose(&[f64::NAN], &[1.0], &[1.0], &[1.0], policy, false),
        Err(FixedPointError::InvalidInventory)
    );
    assert_eq!(
        propose(&[0.0], &[-f64::MAX], &[1.0], &[f64::MAX], policy, false),
        Err(FixedPointError::NonfiniteArithmetic)
    );
    assert_eq!(
        propose(&[0.0], &[0.0], &[1.0], &[f64::MAX], policy, false),
        Err(FixedPointError::NonfiniteArithmetic)
    );
    // Finite inputs and coefficient can still overflow the proposed image.
    assert_eq!(
        propose(&[-f64::MAX], &[2.0], &[f64::MAX], &[1.0], policy, false),
        Err(FixedPointError::NonfiniteArithmetic)
    );
    assert_eq!(
        propose(&[], &[], &[], &[], policy, true),
        Err(FixedPointError::Cancelled)
    );
}
