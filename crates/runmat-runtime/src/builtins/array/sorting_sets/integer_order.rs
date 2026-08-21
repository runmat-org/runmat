//! Exact ordering helpers shared by integer-aware sorting-set builtins.

use std::cmp::Ordering;

use num_bigint::{BigInt, Sign};
use runmat_value::IntValue;

pub(super) fn compare(
    a: &IntValue,
    b: &IntValue,
    descending: bool,
    comparison_by_abs: bool,
) -> Ordering {
    let primary = if comparison_by_abs {
        absolute_magnitude(a).cmp(&absolute_magnitude(b))
    } else {
        Ordering::Equal
    };
    let ordering = if primary == Ordering::Equal {
        if comparison_by_abs {
            compare_raw(a, b).reverse()
        } else {
            compare_raw(a, b)
        }
    } else {
        primary
    };
    if descending {
        ordering.reverse()
    } else {
        ordering
    }
}

pub(super) fn compare_complex(
    a: (&IntValue, &IntValue),
    b: (&IntValue, &IntValue),
    descending: bool,
    comparison_by_real: bool,
) -> Ordering {
    let ordering = if comparison_by_real {
        compare_raw(a.0, b.0).then_with(|| compare_raw(a.1, b.1))
    } else {
        compare_complex_magnitude(a, b).then_with(|| compare_complex_phase(a, b))
    };
    if descending {
        ordering.reverse()
    } else {
        ordering
    }
}

fn compare_raw(a: &IntValue, b: &IntValue) -> Ordering {
    match (a, b) {
        (IntValue::I8(a), IntValue::I8(b)) => a.cmp(b),
        (IntValue::I16(a), IntValue::I16(b)) => a.cmp(b),
        (IntValue::I32(a), IntValue::I32(b)) => a.cmp(b),
        (IntValue::I64(a), IntValue::I64(b)) => a.cmp(b),
        (IntValue::U8(a), IntValue::U8(b)) => a.cmp(b),
        (IntValue::U16(a), IntValue::U16(b)) => a.cmp(b),
        (IntValue::U32(a), IntValue::U32(b)) => a.cmp(b),
        (IntValue::U64(a), IntValue::U64(b)) => a.cmp(b),
        _ => unreachable!("integer storage is homogeneous"),
    }
}

fn absolute_magnitude(value: &IntValue) -> u128 {
    match value {
        IntValue::I8(value) => i128::from(*value).unsigned_abs(),
        IntValue::I16(value) => i128::from(*value).unsigned_abs(),
        IntValue::I32(value) => i128::from(*value).unsigned_abs(),
        IntValue::I64(value) => i128::from(*value).unsigned_abs(),
        IntValue::U8(value) => u128::from(*value),
        IntValue::U16(value) => u128::from(*value),
        IntValue::U32(value) => u128::from(*value),
        IntValue::U64(value) => u128::from(*value),
    }
}

fn compare_complex_magnitude(a: (&IntValue, &IntValue), b: (&IntValue, &IntValue)) -> Ordering {
    let (a_real, a_imag) = (to_bigint(a.0), to_bigint(a.1));
    let (b_real, b_imag) = (to_bigint(b.0), to_bigint(b.1));
    ((&a_real * &a_real) + (&a_imag * &a_imag)).cmp(&((&b_real * &b_real) + (&b_imag * &b_imag)))
}

fn compare_complex_phase(a: (&IntValue, &IntValue), b: (&IntValue, &IntValue)) -> Ordering {
    let (a_real, a_imag) = (to_bigint(a.0), to_bigint(a.1));
    let (b_real, b_imag) = (to_bigint(b.0), to_bigint(b.1));
    let a_lower_half = a_imag.sign() == Sign::Minus;
    let b_lower_half = b_imag.sign() == Sign::Minus;
    match a_lower_half.cmp(&b_lower_half).reverse() {
        Ordering::Equal => {
            let cross = (&a_real * &b_imag) - (&a_imag * &b_real);
            match cross.sign() {
                Sign::Plus => Ordering::Less,
                Sign::Minus => Ordering::Greater,
                Sign::NoSign if a_imag.sign() == Sign::NoSign && b_imag.sign() == Sign::NoSign => {
                    b_real.cmp(&a_real)
                }
                Sign::NoSign => Ordering::Equal,
            }
        }
        ordering => ordering,
    }
}

fn to_bigint(value: &IntValue) -> BigInt {
    match value {
        IntValue::I8(value) => BigInt::from(*value),
        IntValue::I16(value) => BigInt::from(*value),
        IntValue::I32(value) => BigInt::from(*value),
        IntValue::I64(value) => BigInt::from(*value),
        IntValue::U8(value) => BigInt::from(*value),
        IntValue::U16(value) => BigInt::from(*value),
        IntValue::U32(value) => BigInt::from(*value),
        IntValue::U64(value) => BigInt::from(*value),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex_integer_order_is_exact_at_uint64_limits() {
        let maximum = IntValue::U64(u64::MAX);
        let below = IntValue::U64(u64::MAX - 1);
        let one = IntValue::U64(1);
        let zero = IntValue::U64(0);
        assert_eq!(
            compare_complex((&below, &one), (&maximum, &zero), false, false),
            Ordering::Less
        );
    }

    #[test]
    fn complex_integer_magnitude_ties_follow_phase_on_open_closed_interval() {
        let negative = IntValue::I64(-5);
        let positive = IntValue::I64(5);
        let zero = IntValue::I64(0);
        let values = [
            (&zero, &negative),
            (&positive, &zero),
            (&zero, &positive),
            (&negative, &zero),
        ];
        for pair in values.windows(2) {
            assert_eq!(
                compare_complex(pair[0], pair[1], false, false),
                Ordering::Less
            );
        }
        assert_eq!(
            compare_complex((&positive, &zero), (&negative, &zero), false, false),
            Ordering::Less
        );
    }

    #[test]
    fn complex_integer_real_method_and_descending_reverse_both_components() {
        let one = IntValue::I32(1);
        let two = IntValue::I32(2);
        let three = IntValue::I32(3);
        assert_eq!(
            compare_complex((&one, &three), (&two, &one), false, true),
            Ordering::Less
        );
        assert_eq!(
            compare_complex((&one, &three), (&one, &two), true, true),
            Ordering::Less
        );
    }
}
