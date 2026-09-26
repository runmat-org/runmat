//! Display-limit normalization shared by scene queries and the renderer.
pub fn positive_log_bounds(lo: f64, hi: f64) -> (f64, f64) {
    if lo.is_finite() && hi.is_finite() && lo > 0.0 && hi > 0.0 && lo != hi {
        return (lo, hi);
    }
    let upper = if hi.is_finite() && hi > 0.0 {
        hi
    } else if lo.is_finite() && lo > 0.0 {
        lo
    } else {
        1.0
    };
    let lower = (upper / 1000.0).max(f64::MIN_POSITIVE);
    if lower < upper {
        (lower, upper)
    } else {
        (upper * 0.1, upper * 10.0)
    }
}
