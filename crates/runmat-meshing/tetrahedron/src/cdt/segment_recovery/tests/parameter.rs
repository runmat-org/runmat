//! Exact parameter admission occurs after reduction, with unchanged range limits.
use super::*;

fn context() -> SegmentContext {
    SegmentContext {
        constraint_index: 17,
        first_identity: StableDigest::from_bytes([1; 32]),
        last_identity: StableDigest::from_bytes([2; 32]),
        first_coordinates: [0.0; 3],
        last_coordinates: [1.0, 0.0, 0.0],
    }
}

fn dyadic(numerator: u64, exponent: u8) -> DyadicNode {
    DyadicNode {
        identity: StableDigest::ZERO,
        numerator,
        exponent,
    }
}

#[test]
fn midpoint_reduces_before_testing_representation_limit() {
    let midpoint = dyadic(3, 62).midpoint(dyadic(5, 62), context()).unwrap();
    assert_eq!((midpoint.numerator, midpoint.exponent), (1, 60));
    assert_eq!(midpoint.parameter().unwrap(), 2.0_f64.powi(-60));
}

#[test]
fn midpoint_handles_full_range_without_intermediate_overflow() {
    let midpoint = dyadic((1_u64 << 62) - 3, 62)
        .midpoint(dyadic((1_u64 << 62) - 1, 62), context())
        .unwrap();
    assert_eq!(
        (midpoint.numerator, midpoint.exponent),
        ((1_u64 << 61) - 1, 61)
    );
}

#[test]
fn midpoint_rejects_genuinely_unrepresentable_parameter() {
    let failure = dyadic(1, 62)
        .midpoint(dyadic(2, 62), context())
        .err()
        .expect("an exponent-63 midpoint must reject");
    assert_eq!(
        failure.kind,
        DelaunaySegmentRecoveryErrorKind::UnsatisfiableConstraint
    );
    assert_eq!(failure.constraint_index, Some(17));
    assert!(failure.reason.contains("after exact reduction"));
}

#[test]
fn unrepresentable_midpoint_takes_the_coarsest_inner_parameter() {
    // The exact midpoint of 1/2^62 and 3/4 needs exponent 63; 1/2 lies in the middle half.
    let inner = dyadic(1, 62)
        .midpoint(dyadic(3, 2), context())
        .expect("a coarser inner parameter exists");
    assert_eq!((inner.numerator, inner.exponent), (1, 1));
    // Width 3 at exponent 62 with no middle-half point coarser than the ends: strictly inside.
    let inner = dyadic(4, 62).midpoint(dyadic(7, 62), context()).unwrap();
    assert!(inner.exponent <= 62);
    let value = inner.parameter().unwrap();
    assert!(value > 4.0 * 2.0_f64.powi(-62) && value < 7.0 * 2.0_f64.powi(-62));
}

#[test]
fn off_centre_splits_reserve_precision_for_later_subdivision() {
    let mut left = dyadic(0, 0);
    let right = dyadic(1, 0);
    for depth in 1..=48 {
        let start = left.parameter().unwrap();
        let interval = right.parameter().unwrap() - start;
        let preferred = start + 0.49 * interval;
        let split = left.near_parameter(right, preferred, context()).unwrap();
        let actual = split.parameter().unwrap();
        assert!(actual > start && actual < right.parameter().unwrap());
        assert!((actual - preferred).abs() <= interval / 512.0);
        assert!(split.exponent <= depth + 8);
        left = split;
    }
}
