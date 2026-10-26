//! Shared numerical helpers for probability distribution builtins.

#[path = "distribution_math/discrete_quantiles.rs"]
mod discrete_quantiles;
pub(crate) use discrete_quantiles::{first_representable_discrete_quantile, poisson_inverse};

use crate::builtins::math::elementwise::error_functions::erfcinv::erfcinv_scalar;
use crate::builtins::math::elementwise::gamma_functions::gammaln::gammaln_nonnegative_scalar;

const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
const SQRT_2: f64 = std::f64::consts::SQRT_2;
const NORMAL_APPROX_NU: f64 = 1.0e8;

pub(crate) fn standard_normal_cdf(z: f64) -> f64 {
    if z == f64::INFINITY {
        1.0
    } else if z == f64::NEG_INFINITY {
        0.0
    } else {
        0.5 * libm::erfc(-z / std::f64::consts::SQRT_2)
    }
}

pub(crate) fn standard_normal_cdf_upper(z: f64) -> f64 {
    if z == f64::INFINITY {
        0.0
    } else if z == f64::NEG_INFINITY {
        1.0
    } else {
        0.5 * libm::erfc(z / SQRT_2)
    }
}

pub(crate) fn standard_normal_inv(p: f64) -> f64 {
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    -SQRT_2 * erfcinv_scalar(2.0 * p)
}

pub(crate) fn standard_normal_pdf(z: f64) -> f64 {
    INV_SQRT_2PI * (-0.5 * z * z).exp()
}

/// Log form for density compositions whose final scaling can recover a
/// representable result even when the unscaled normal PDF underflows.
pub(crate) fn standard_normal_log_pdf(z: f64) -> f64 {
    INV_SQRT_2PI.ln() - 0.5 * z * z
}

pub(crate) fn student_t_pdf(t: f64, nu: f64) -> f64 {
    if t.is_nan() || nu.is_nan() || nu <= 0.0 {
        return f64::NAN;
    }
    if t.is_infinite() {
        return 0.0;
    }
    if nu >= NORMAL_APPROX_NU {
        return standard_normal_pdf(t);
    }
    let log_num = gammaln_nonnegative_scalar((nu + 1.0) / 2.0);
    let log_den =
        0.5 * (nu.ln() + std::f64::consts::PI.ln()) + gammaln_nonnegative_scalar(nu / 2.0);
    let log_kernel = -((nu + 1.0) / 2.0) * log1p_t2_over_nu(t, nu);
    (log_num - log_den + log_kernel).exp()
}

pub(crate) fn student_t_cdf(t: f64, nu: f64) -> f64 {
    student_t_tail(t, nu, false)
}

pub(crate) fn student_t_cdf_upper(t: f64, nu: f64) -> f64 {
    student_t_tail(t, nu, true)
}

fn student_t_tail(t: f64, nu: f64, upper: bool) -> f64 {
    if t.is_nan() || nu.is_nan() || nu <= 0.0 {
        return f64::NAN;
    }
    if t == f64::INFINITY {
        return if upper { 0.0 } else { 1.0 };
    }
    if t == f64::NEG_INFINITY {
        return if upper { 1.0 } else { 0.0 };
    }
    if nu >= NORMAL_APPROX_NU {
        return if upper {
            standard_normal_cdf_upper(t)
        } else {
            standard_normal_cdf(t)
        };
    }
    let Some(x) = beta_arg_for_t(t, nu) else {
        let tail = student_t_upper_tail_asymptotic(t.abs(), nu);
        return if upper == (t >= 0.0) {
            tail
        } else {
            1.0 - tail
        };
    };
    let ib = regularized_beta(x, nu / 2.0, 0.5);
    if upper {
        if t >= 0.0 {
            0.5 * ib
        } else {
            1.0 - 0.5 * ib
        }
    } else if t >= 0.0 {
        1.0 - 0.5 * ib
    } else {
        0.5 * ib
    }
}

pub(crate) fn student_t_inv(p: f64, nu: f64) -> f64 {
    if p.is_nan() || nu.is_nan() || nu <= 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    if p == 0.5 {
        return 0.0;
    }
    if nu >= NORMAL_APPROX_NU {
        return standard_normal_inv(p);
    }
    if p < 0.5 {
        return -student_t_inv_upper_tail(p, nu);
    }
    student_t_inv_upper_tail(1.0 - p, nu)
}

/// Positive Student-t quantile from an upper-tail probability. Keeping the
/// tail probability avoids rounding `1 - alpha/2` to one for small alpha.
pub(crate) fn student_t_inv_upper_tail(q: f64, nu: f64) -> f64 {
    if q == 0.0 {
        return f64::INFINITY;
    }
    let mut lo = 0.0;
    let mut hi = 1.0;
    let mut iterations = 0;
    while student_t_cdf_upper(hi, nu) > q {
        hi *= 2.0;
        iterations += 1;
        if !hi.is_finite() || iterations > 2048 {
            return f64::INFINITY;
        }
    }
    for _ in 0..160 {
        let mid = 0.5 * (lo + hi);
        if student_t_cdf_upper(mid, nu) <= q {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

fn beta_arg_for_t(t: f64, nu: f64) -> Option<f64> {
    let scaled = t.abs() / nu.sqrt();
    if scaled.is_infinite() {
        return None;
    }
    let denom = 1.0 + scaled * scaled;
    if denom.is_infinite() {
        None
    } else {
        let x = 1.0 / denom;
        (x > 0.0).then_some(x)
    }
}

fn log1p_t2_over_nu(t: f64, nu: f64) -> f64 {
    let scaled = t.abs() / nu.sqrt();
    if scaled.is_finite() {
        (scaled * scaled).ln_1p()
    } else {
        2.0 * t.abs().ln() - nu.ln()
    }
}

fn student_t_upper_tail_asymptotic(t: f64, nu: f64) -> f64 {
    let log_tail = gammaln_nonnegative_scalar((nu + 1.0) / 2.0)
        - 0.5 * std::f64::consts::PI.ln()
        - gammaln_nonnegative_scalar(nu / 2.0)
        + (nu / 2.0 - 1.0) * nu.ln()
        - nu * t.ln();
    log_tail.exp()
}

pub(crate) fn regularized_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let log_bt = gammaln_nonnegative_scalar(a + b)
        - gammaln_nonnegative_scalar(a)
        - gammaln_nonnegative_scalar(b)
        + a * x.ln()
        + b * (1.0 - x).ln();
    let bt = log_bt.exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * beta_continued_fraction(a, b, x) / a
    } else {
        1.0 - bt * beta_continued_fraction(b, a, 1.0 - x) / b
    }
}

/// Lower regularized beta with a logarithmic argument. Preserve a finite
/// small probability when x, or the beta prefactor, is subnormal/underflows.
/// Ordinary arguments still use the existing continued-fraction owner.
pub(crate) fn regularized_beta_from_log_x(log_x: f64, a: f64, b: f64) -> f64 {
    if log_x.is_nan() || log_x > 0.0 || !a.is_finite() || !b.is_finite() || a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }
    if log_x == f64::NEG_INFINITY {
        return 0.0;
    }
    if log_x == 0.0 {
        return 1.0;
    }
    // B_x(a,1)/B(a,1)=x^a, without a gamma-difference cancellation.
    if b == 1.0 {
        return (a * log_x).exp();
    }
    let x = log_x.exp();
    if x >= f64::MIN_POSITIVE {
        return regularized_beta(x, a, b);
    }
    // DLMF8.17.7: I_x=x^a/(a B(a,b))*2F1(a,1-b;a+1;x).
    // Retain log(x) rather than taking log of rounded/zero subnormal x.
    // The existing fraction carries the higher-order terms whenever x is
    // representable. If exp(log_x)==0, finite b times the lost x is below
    // 5e-16, beneath the fraction's existing3e-14 tolerance.
    if a == 1.0 {
        // For subnormal x, -log(1-x)=x to much better than binary64
        // precision, but b*x need not be small. Preserve that product in log.
        let exponent = (log_x + b.ln()).exp();
        return -(-exponent).exp_m1();
    }
    let normalization = gammaln_nonnegative_scalar(a + b)
        - gammaln_nonnegative_scalar(a)
        - gammaln_nonnegative_scalar(b);
    let log_result =
        normalization + a * log_x + b * (-x).ln_1p() + beta_continued_fraction(a, b, x).ln()
            - a.ln();
    log_result.exp().clamp(0.0, 1.0)
}

pub(crate) fn regularized_gamma_p(a: f64, x: f64) -> f64 {
    if a.is_nan() || x.is_nan() || a <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    if x == f64::INFINITY {
        return 1.0;
    }
    if x < a + 1.0 {
        gamma_series_p(a, x)
    } else {
        1.0 - gamma_continued_fraction_q(a, x)
    }
}

pub(crate) fn regularized_gamma_q(a: f64, x: f64) -> f64 {
    if a.is_nan() || x.is_nan() || a <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 1.0;
    }
    if x == f64::INFINITY {
        return 0.0;
    }
    if x < a + 1.0 {
        1.0 - gamma_series_p(a, x)
    } else {
        gamma_continued_fraction_q(a, x)
    }
}

fn gamma_series_p(a: f64, x: f64) -> f64 {
    const MAX_ITER: usize = 512;
    const EPS: f64 = 3.0e-14;

    let gln = gammaln_nonnegative_scalar(a);
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    for _ in 0..MAX_ITER {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() <= sum.abs() * EPS {
            break;
        }
    }
    sum * (-x + a * x.ln() - gln).exp()
}

fn gamma_continued_fraction_q(a: f64, x: f64) -> f64 {
    const MAX_ITER: usize = 512;
    const EPS: f64 = 3.0e-14;
    const FP_MIN: f64 = 1.0e-300;

    let gln = gammaln_nonnegative_scalar(a);
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / FP_MIN;
    let mut d = 1.0 / b.max(FP_MIN);
    let mut h = d;
    for i in 1..=MAX_ITER {
        let i_f = i as f64;
        let an = -i_f * (i_f - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < FP_MIN {
            d = FP_MIN;
        }
        c = b + an / c;
        if c.abs() < FP_MIN {
            c = FP_MIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() <= EPS {
            break;
        }
    }
    (-x + a * x.ln() - gln).exp() * h
}

fn beta_continued_fraction(a: f64, b: f64, x: f64) -> f64 {
    const MAX_ITER: usize = 200;
    const EPS: f64 = 3.0e-14;
    const FP_MIN: f64 = 1.0e-300;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FP_MIN {
        d = FP_MIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAX_ITER {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;
        let mut aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FP_MIN {
            d = FP_MIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FP_MIN {
            c = FP_MIN;
        }
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FP_MIN {
            d = FP_MIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FP_MIN {
            c = FP_MIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

#[cfg(test)]
mod log_argument_beta_tests {
    use super::*;
    #[test]
    fn logarithmic_argument_preserves_closed_form_tails_below_the_x_frontier() {
        for log_x in [-710.0, -1000.0, -1400.0] {
            let actual = regularized_beta_from_log_x(log_x, 0.5, 1.0);
            assert!((actual / (log_x * 0.5).exp() - 1.0).abs() < 1e-13);
            let actual = regularized_beta_from_log_x(log_x, 0.5, 0.5);
            let expected = 2.0 / std::f64::consts::PI * (log_x * 0.5).exp();
            assert!((actual / expected - 1.0).abs() < 1e-12);
        }
        let actual = regularized_beta_from_log_x(-746.0, 1.0, 1e10);
        assert!((actual / (-746.0 + 1e10f64.ln()).exp() - 1.0).abs() < 1e-12);
        let log_x = f64::MIN_POSITIVE.ln() - 0.5;
        let b = 1e308_f64;
        let expected = -(-(log_x + b.ln()).exp()).exp_m1();
        assert!((regularized_beta_from_log_x(log_x, 1.0, b) - expected).abs() < 1e-13);
        assert_eq!(
            regularized_beta_from_log_x(f64::NEG_INFINITY, 0.5, 1.0),
            0.0
        );
        assert_eq!(regularized_beta_from_log_x(0.0, 0.5, 1.0), 1.0);
        assert!(regularized_beta_from_log_x(f64::NAN, 0.5, 1.0).is_nan());
        assert!(
            (regularized_beta_from_log_x(0.25f64.ln(), 2.0, 3.0)
                - regularized_beta(0.25, 2.0, 3.0))
            .abs()
                < 1e-14
        );
    }
}
