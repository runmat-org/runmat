//! Representable discrete quantiles shared by named and fitted distributions.

use super::regularized_gamma_q;
use crate::BuiltinResult;

/// Select the first integer-valued `f64` whose CDF reaches the probability.
/// Above the consecutive-integer range the search advances through
/// representable values, so every iteration makes progress.
pub(crate) fn first_representable_discrete_quantile(
    probability: f64,
    mut hi: f64,
    mut cdf: impl FnMut(f64) -> f64,
    poll: &impl Fn() -> BuiltinResult<()>,
) -> BuiltinResult<f64> {
    let mut lo: f64 = 0.0;
    poll()?;
    if cdf(lo) >= probability {
        return Ok(lo);
    }
    while lo < hi {
        poll()?;
        let low_bits = lo.to_bits();
        let high_bits = hi.to_bits();
        if high_bits - low_bits <= 1 {
            return Ok(hi);
        }
        let candidate = f64::from_bits(low_bits + (high_bits - low_bits) / 2).ceil();
        let next = if lo < 9_007_199_254_740_992.0 {
            lo + 1.0
        } else {
            f64::from_bits(low_bits + 1)
        };
        let mid = if candidate <= lo || candidate >= hi {
            next
        } else {
            candidate
        };
        if mid >= hi {
            return Ok(hi);
        }
        if cdf(mid) >= probability {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Ok(hi)
}

pub(crate) fn poisson_inverse(
    probability: f64,
    lambda: f64,
    poll: impl Fn() -> BuiltinResult<()>,
) -> BuiltinResult<f64> {
    poll()?;
    if probability.is_nan()
        || lambda.is_nan()
        || lambda < 0.0
        || !(0.0..=1.0).contains(&probability)
    {
        return Ok(f64::NAN);
    }
    if probability == 1.0 {
        return Ok(f64::INFINITY);
    }
    if probability == 0.0 || lambda == 0.0 {
        return Ok(0.0);
    }
    let mut hi = lambda.ceil().max(1.0);
    let mut iterations = 0;
    while poisson_cdf(hi, lambda) < probability {
        poll()?;
        hi *= 2.0;
        iterations += 1;
        if !hi.is_finite() || iterations > 2048 {
            return Ok(f64::INFINITY);
        }
    }
    first_representable_discrete_quantile(probability, hi, |x| poisson_cdf(x, lambda), &poll)
}

fn poisson_cdf(k: f64, lambda: f64) -> f64 {
    if k.is_nan() || lambda.is_nan() || lambda < 0.0 {
        return f64::NAN;
    }
    if k < 0.0 {
        return 0.0;
    }
    if lambda == 0.0 {
        return 1.0;
    }
    regularized_gamma_q(k.floor() + 1.0, lambda)
}
