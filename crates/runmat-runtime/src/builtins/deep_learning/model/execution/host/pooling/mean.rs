//! Native-precision mean shared by real and complex pooling components.

use crate::builtins::deep_learning::model::execution::host::numeric::Real;

pub(super) fn stable_mean<T: Real>(size: usize, value_at: impl Fn(usize) -> T) -> T {
    debug_assert!(size > 0);
    let mut sum = T::zero();
    let mut finite_overflow = false;
    for index in 0..size {
        let value = value_at(index);
        let next = sum + value;
        if sum.is_finite() && value.is_finite() && !next.is_finite() {
            finite_overflow = true;
            break;
        }
        sum = next;
    }
    if !finite_overflow {
        return sum / T::from_f64(size as f64);
    }
    // Keep the established sum/divide rounding on the common path. Recompute
    // only finite overflow in the input precision; no wider storage mirror.
    let mut mean = T::zero();
    for index in 0..size {
        let count = T::from_f64((index + 1) as f64);
        let previous_weight = T::from_f64(index as f64) / count;
        mean = mean * previous_weight + value_at(index) / count;
    }
    mean
}
