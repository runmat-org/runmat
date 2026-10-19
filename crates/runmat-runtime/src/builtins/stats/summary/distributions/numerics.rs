use super::*;

pub(super) const SQRT_2: f64 = std::f64::consts::SQRT_2;
pub(super) const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

pub(super) fn normpdf_scalar(x: f64, mu: f64, sigma: f64) -> f64 {
    if sigma.is_nan() || x.is_nan() || mu.is_nan() {
        return f64::NAN;
    }
    if sigma <= 0.0 {
        return f64::NAN;
    }
    let z = (x - mu) / sigma;
    INV_SQRT_2PI / sigma * (-0.5 * z * z).exp()
}

pub(super) fn normcdf_scalar(x: f64, mu: f64, sigma: f64) -> f64 {
    if sigma.is_nan() || x.is_nan() || mu.is_nan() {
        return f64::NAN;
    }
    if sigma < 0.0 {
        return f64::NAN;
    }
    if sigma == 0.0 {
        return if x < mu { 0.0 } else { 1.0 };
    }
    0.5 * libm::erfc(-(x - mu) / (sigma * SQRT_2))
}

pub(super) fn normcdf_upper_scalar(x: f64, mu: f64, sigma: f64) -> f64 {
    if sigma.is_nan() || x.is_nan() || mu.is_nan() {
        return f64::NAN;
    }
    if sigma < 0.0 {
        return f64::NAN;
    }
    if sigma == 0.0 {
        return if x < mu { 1.0 } else { 0.0 };
    }
    0.5 * libm::erfc((x - mu) / (sigma * SQRT_2))
}

pub(super) fn norminv_scalar(p: f64, mu: f64, sigma: f64) -> f64 {
    if sigma.is_nan() || p.is_nan() || mu.is_nan() {
        return f64::NAN;
    }
    if sigma <= 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    mu - sigma * SQRT_2 * erfcinv_scalar(2.0 * p)
}

pub(super) fn binocdf_scalar(x: f64, n: f64, p: f64, tail: DistributionTail) -> f64 {
    if x.is_nan()
        || n.is_nan()
        || p.is_nan()
        || n < 0.0
        || n.fract() != 0.0
        || !(0.0..=1.0).contains(&p)
    {
        return f64::NAN;
    }
    if x < 0.0 {
        return if tail == DistributionTail::Upper {
            1.0
        } else {
            0.0
        };
    }
    let k = x.floor();
    if k >= n {
        return if tail == DistributionTail::Upper {
            0.0
        } else {
            1.0
        };
    }
    if p == 0.0 {
        return if tail == DistributionTail::Upper {
            0.0
        } else {
            1.0
        };
    }
    if p == 1.0 {
        return if tail == DistributionTail::Upper {
            1.0
        } else {
            0.0
        };
    }
    if tail == DistributionTail::Upper {
        distribution_math::regularized_beta(p, k + 1.0, n - k)
    } else {
        distribution_math::regularized_beta(1.0 - p, n - k, k + 1.0)
    }
}

pub(super) fn chi2cdf_scalar(x: f64, nu: f64, tail: DistributionTail) -> f64 {
    if x.is_nan() || nu.is_nan() || nu <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return if tail == DistributionTail::Upper {
            1.0
        } else {
            0.0
        };
    }
    if tail == DistributionTail::Upper {
        distribution_math::regularized_gamma_q(nu / 2.0, x / 2.0)
    } else {
        distribution_math::regularized_gamma_p(nu / 2.0, x / 2.0)
    }
}

pub(super) fn wblinv_scalar(p: f64, a: f64, b: f64) -> f64 {
    if p.is_nan() || a.is_nan() || b.is_nan() || a <= 0.0 || b <= 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    a * (-(-p).ln_1p()).powf(1.0 / b)
}

pub(super) fn chi2inv_scalar(p: f64, nu: f64) -> f64 {
    if p.is_nan() || nu.is_nan() || nu <= 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    let mut lo = 0.0;
    let mut hi = nu.max(1.0);
    let mut iterations = 0;
    while chi2cdf_scalar(hi, nu, DistributionTail::Lower) < p {
        hi *= 2.0;
        iterations += 1;
        if !hi.is_finite() || iterations > 2048 {
            return f64::INFINITY;
        }
    }
    for _ in 0..160 {
        let mid = 0.5 * (lo + hi);
        if chi2cdf_scalar(mid, nu, DistributionTail::Lower) >= p {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

pub(super) fn binoinv_scalar(
    probability: f64,
    n: f64,
    success_probability: f64,
    poll: impl Fn() -> BuiltinResult<()>,
) -> BuiltinResult<f64> {
    poll()?;
    if probability.is_nan()
        || n.is_nan()
        || success_probability.is_nan()
        || n < 0.0
        || n.fract() != 0.0
        || !(0.0..=1.0).contains(&probability)
        || !(0.0..=1.0).contains(&success_probability)
    {
        return Ok(f64::NAN);
    }
    if probability == 0.0 || success_probability == 0.0 {
        return Ok(0.0);
    }
    if probability == 1.0 || success_probability == 1.0 {
        return Ok(n);
    }
    distribution_math::first_representable_discrete_quantile(
        probability,
        n,
        |x| binocdf_scalar(x, n, success_probability, DistributionTail::Lower),
        &poll,
    )
}

pub(super) fn expinv_scalar(p: f64, mu: f64) -> f64 {
    if p.is_nan() || mu.is_nan() || mu <= 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    -mu * (-p).ln_1p()
}

pub(super) fn unifinv_scalar(p: f64, a: f64, b: f64) -> f64 {
    if p.is_nan() || a.is_nan() || b.is_nan() || a > b || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    a + p * (b - a)
}

pub(super) fn logninv_scalar(p: f64, mu: f64, sigma: f64) -> f64 {
    let normal = norminv_scalar(p, mu, sigma);
    if normal.is_nan() {
        f64::NAN
    } else {
        normal.exp()
    }
}

pub(super) fn gaminv_scalar(p: f64, shape: f64, scale: f64) -> f64 {
    if p.is_nan()
        || shape.is_nan()
        || scale.is_nan()
        || shape <= 0.0
        || scale <= 0.0
        || !(0.0..=1.0).contains(&p)
    {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    invert_continuous_positive(p, shape * scale, |x| {
        distribution_math::regularized_gamma_p(shape, x / scale)
    })
}

pub(super) fn betainv_scalar(p: f64, a: f64, b: f64) -> f64 {
    if p.is_nan() || a.is_nan() || b.is_nan() || a <= 0.0 || b <= 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return 1.0;
    }
    let mut lo = 0.0;
    let mut hi = 1.0;
    for _ in 0..180 {
        let mid = 0.5 * (lo + hi);
        if distribution_math::regularized_beta(mid, a, b) >= p {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

pub(super) fn finv_scalar(p: f64, v1: f64, v2: f64) -> f64 {
    if p.is_nan()
        || v1.is_nan()
        || v2.is_nan()
        || v1 <= 0.0
        || v2 <= 0.0
        || !(0.0..=1.0).contains(&p)
    {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    invert_continuous_positive(p, 1.0, |x| {
        let scaled = v1 * x;
        let beta_x = scaled / (scaled + v2);
        distribution_math::regularized_beta(beta_x, v1 / 2.0, v2 / 2.0)
    })
}

pub(super) fn poissinv_scalar(
    p: f64,
    lambda: f64,
    poll: impl Fn() -> BuiltinResult<()>,
) -> BuiltinResult<f64> {
    distribution_math::poisson_inverse(p, lambda, poll)
}

// Retained for inverse-quantile boundary witnesses; production search owns
// its CDF evaluation in the shared distribution-math service.
#[cfg(test)]
pub(super) fn poisscdf_scalar(k: f64, lambda: f64) -> f64 {
    if k.is_nan() || lambda.is_nan() || lambda < 0.0 {
        return f64::NAN;
    }
    if k < 0.0 {
        return 0.0;
    }
    if lambda == 0.0 {
        return 1.0;
    }
    distribution_math::regularized_gamma_q(k.floor() + 1.0, lambda)
}

pub(super) fn invert_continuous_positive<F>(p: f64, initial_hi: f64, mut cdf: F) -> f64
where
    F: FnMut(f64) -> f64,
{
    let mut lo = 0.0;
    let mut hi = initial_hi.max(1.0);
    let mut iterations = 0;
    while cdf(hi) < p {
        hi *= 2.0;
        iterations += 1;
        if !hi.is_finite() || iterations > 2048 {
            return f64::INFINITY;
        }
    }
    for _ in 0..180 {
        let mid = 0.5 * (lo + hi);
        if cdf(mid) >= p {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
mod discrete_search_tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn discrete_inverse_preserves_cdf_crossing_and_wide_endpoint() {
        let binomial = binoinv_scalar(0.75, 10.0, 0.5, || Ok(())).unwrap();
        assert_eq!(binomial, 6.0);
        assert!(binocdf_scalar(binomial - 1.0, 10.0, 0.5, DistributionTail::Lower) < 0.75);
        assert!(binocdf_scalar(binomial, 10.0, 0.5, DistributionTail::Lower) >= 0.75);

        let poisson = poissinv_scalar(0.8, 3.0, || Ok(())).unwrap();
        assert_eq!(poisson, 4.0);
        assert!(poisscdf_scalar(poisson - 1.0, 3.0) < 0.8);
        assert!(poisscdf_scalar(poisson, 3.0) >= 0.8);

        let wide_trials = (1_u64 << 54) as f64;
        assert_eq!(
            binoinv_scalar(1.0, wide_trials, 0.5, || Ok(())).unwrap(),
            wide_trials
        );
    }

    #[test]
    fn wide_representable_threshold_progresses_and_polls() {
        let threshold = f64::from_bits(((1_u64 << 54) as f64).to_bits() + 3);
        let polls = Cell::new(0);
        let result = distribution_math::first_representable_discrete_quantile(
            0.5,
            threshold * 2.0,
            |value| if value >= threshold { 1.0 } else { 0.0 },
            &|| {
                polls.set(polls.get() + 1);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(result, threshold);
        assert!(polls.get() > 1);
        assert!(
            polls.get() <= 70,
            "ordered-float bisection must make progress"
        );
    }

    #[test]
    fn discrete_search_propagates_cooperative_cancellation() {
        let polls = Cell::new(0);
        let error = distribution_math::first_representable_discrete_quantile(
            0.5,
            1.0e18,
            |value| if value >= 1.0e16 { 1.0 } else { 0.0 },
            &|| {
                polls.set(polls.get() + 1);
                if polls.get() > 2 {
                    Err(normal_error("icdf", "icdf: cancelled"))
                } else {
                    Ok(())
                }
            },
        )
        .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:icdf:InvalidArgument"));
    }
}
