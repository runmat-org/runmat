//! Decisions use the exact certified bounds, never rounded output gap fields.
use num_rational::BigRational;
use num_traits::{Signed, Zero};
pub(super) fn satisfies(
    upper: &BigRational,
    lower: &BigRational,
    absolute: f64,
    relative: f64,
) -> bool {
    if lower > upper {
        return false;
    }
    let gap = upper - lower;
    let Some(absolute) = BigRational::from_float(absolute) else {
        return false;
    };
    let Some(relative) = BigRational::from_float(relative) else {
        return false;
    };
    gap <= absolute || gap <= relative * upper.abs()
}
pub(super) fn relative(upper: &BigRational, lower: &BigRational) -> Option<BigRational> {
    if lower > upper {
        return None;
    }
    if upper.is_zero() {
        return if lower.is_zero() {
            Some(BigRational::zero())
        } else {
            None
        };
    }
    Some((upper - lower) / upper.abs())
}
