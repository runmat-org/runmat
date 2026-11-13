//! Connected to the original-model terminal mechanism control.
use super::*;
pub(in crate::builtins::math::optim) async fn check_arithmetic(policy: Policy) {
    let mut c = ScalarCooperation::for_builtin(policy.identity, None);
    let a = Rational::from_float(f64::MAX).unwrap();
    let tiny = Rational::from_float(f64::from_bits(1)).unwrap();
    let one = Rational::from_float(1.).unwrap();
    let product = a.mul(&tiny, policy, &mut c).await.unwrap();
    assert!(product
        .div(&tiny, policy, &mut c)
        .await
        .unwrap()
        .compare(&a, policy, &mut c)
        .await
        .unwrap()
        .is_eq());
    let sum = a.add(&tiny, policy, &mut c).await.unwrap();
    assert!(sum
        .sub(&a, policy, &mut c)
        .await
        .unwrap()
        .compare(&tiny, policy, &mut c)
        .await
        .unwrap()
        .is_eq());
    assert!(sum.compare(&a, policy, &mut c).await.unwrap().is_gt());
    assert!(sum
        .negated()
        .compare(&a.negated(), policy, &mut c)
        .await
        .unwrap()
        .is_lt());
    let three = Rational::from_float(3.).unwrap();
    let third = one.div(&three, policy, &mut c).await.unwrap();
    let whole = third
        .add(&third, policy, &mut c)
        .await
        .unwrap()
        .add(&third, policy, &mut c)
        .await
        .unwrap();
    assert!(whole.compare(&one, policy, &mut c).await.unwrap().is_eq());
    let library = num_rational::BigRational::new(1.into(), 3.into());
    let imported = Rational::import(&library, policy, &mut c)
        .await
        .unwrap()
        .unwrap();
    assert!(third
        .compare(&imported, policy, &mut c)
        .await
        .unwrap()
        .is_eq());
    // A bounded independent library oracle checks limb/sign normalization,
    // rather than only composing this implementation with its own inverse.
    for (left, right) in [
        (4_294_967_295., 4_294_967_297.),
        (-9_007_199_254_740_991., 3.),
        (-0.125, 0.375),
        (f64::MAX, f64::from_bits(1)),
    ] {
        let a = Rational::from_float(left).unwrap();
        let b = Rational::from_float(right).unwrap();
        let expected_a = num_rational::BigRational::from_float(left).unwrap();
        let expected_b = num_rational::BigRational::from_float(right).unwrap();
        for (actual, expected) in [
            (
                a.add(&b, policy, &mut c).await.unwrap(),
                &expected_a + &expected_b,
            ),
            (
                a.sub(&b, policy, &mut c).await.unwrap(),
                &expected_a - &expected_b,
            ),
            (
                a.mul(&b, policy, &mut c).await.unwrap(),
                &expected_a * &expected_b,
            ),
            (
                a.div(&b, policy, &mut c).await.unwrap(),
                &expected_a / &expected_b,
            ),
        ] {
            assert_eq!(actual.export(policy, &mut c).await.unwrap(), expected);
            let parts = actual.materialize(policy, &mut c).await.unwrap();
            assert_eq!(
                actual.is_negative(),
                expected.numer().sign() == num_bigint::Sign::Minus
            );
            assert_eq!(
                parts.numerator,
                expected.numer().iter_u32_digits().collect::<Vec<_>>()
            );
            assert_eq!(
                parts.denominator,
                expected.denom().iter_u32_digits().collect::<Vec<_>>()
            );
        }
    }
    let mut power = Rational::from_float(2.).unwrap();
    for _ in 0..12 {
        power = power.mul(&power, policy, &mut c).await.unwrap();
    }
    assert!(power
        .div(&power, policy, &mut c)
        .await
        .unwrap()
        .compare(&one, policy, &mut c)
        .await
        .unwrap()
        .is_eq());

    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
    use runmat_execution::{cancellation::CancellationSource, CancellationReason};
    use std::{cell::Cell, num::NonZeroU64};
    let source = Rc::new(CancellationSource::new());
    let turns = Rc::new(Cell::new(0usize));
    let cancel = Rc::clone(&source);
    let observed = Rc::clone(&turns);
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        move || {
            observed.set(observed.get() + 1);
            cancel.cancel(CancellationReason::User);
            Box::pin(async { Ok(()) })
        },
    );
    let mut interrupted = ScalarCooperation::with_builtin_budget(policy.identity, budget);
    let error = power
        .mul(&sum, policy, &mut interrupted)
        .await
        .err()
        .expect("original cancellation interrupts scalar limb work");
    assert!(crate::runtime_error::error_has_control_cancellation_origin(
        &error
    ));
    assert_eq!(turns.get(), 1);
    assert!(
        power.is_positive(),
        "immutable original operand survives cancellation"
    );
}
