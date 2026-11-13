//! Magnitude/phase ordering and repeated-fill rank selection in native precision.
use super::*;

pub(super) fn complex_median<T: Float>(values: &[(T, T)]) -> (T, T) {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| compare_complex_auto(a, b));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        (
            (sorted[mid - 1].0 + sorted[mid].0) / number::<T>(2),
            (sorted[mid - 1].1 + sorted[mid].1) / number::<T>(2),
        )
    }
}

pub(super) fn complex_median_with_fill<T: Float>(
    values: &[(T, T)],
    fill: (T, T),
    fill_count: usize,
) -> (T, T) {
    if fill_count == 0 {
        return complex_median(values);
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| compare_complex_auto(a, b));
    let total = sorted.len() + fill_count;
    if total == 0 {
        return (T::nan(), T::nan());
    }
    let mid = total / 2;
    if total % 2 == 1 {
        kth_complex_with_repeated_fill(&sorted, fill, fill_count, mid)
    } else {
        let lo = kth_complex_with_repeated_fill(&sorted, fill, fill_count, mid - 1);
        let hi = kth_complex_with_repeated_fill(&sorted, fill, fill_count, mid);
        (
            (lo.0 + hi.0) / number::<T>(2),
            (lo.1 + hi.1) / number::<T>(2),
        )
    }
}

pub(super) fn kth_complex_with_repeated_fill<T: Float>(
    sorted: &[(T, T)],
    fill: (T, T),
    fill_count: usize,
    k: usize,
) -> (T, T) {
    let less = sorted.partition_point(|value| compare_complex_auto(value, &fill) == Ordering::Less);
    let equal = sorted[less..]
        .partition_point(|value| compare_complex_auto(value, &fill) == Ordering::Equal);
    if k < less {
        sorted[k]
    } else if k < less + equal + fill_count {
        fill
    } else {
        sorted[k - fill_count]
    }
}

pub(super) fn complex_pow_usize<T: Float>(mut base: (T, T), mut exponent: usize) -> (T, T) {
    let mut result = (T::one(), T::zero());
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = complex_mul(result, base);
        }
        exponent >>= 1;
        if exponent > 0 {
            base = complex_mul(base, base);
        }
    }
    result
}

pub(super) fn complex_mul<T: Float>(a: (T, T), b: (T, T)) -> (T, T) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

pub(super) fn compare_complex_auto<T: Float>(a: &(T, T), b: &(T, T)) -> Ordering {
    let a_abs = a.0.hypot(a.1);
    let b_abs = b.0.hypot(b.1);
    a_abs
        .partial_cmp(&b_abs)
        .unwrap_or(Ordering::Equal)
        .then_with(|| {
            a.1.atan2(a.0)
                .partial_cmp(&b.1.atan2(b.0))
                .unwrap_or(Ordering::Equal)
        })
}
