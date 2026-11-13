use super::*;

pub(super) fn pdf_scalar(fit: &FittedDistribution, x: f64) -> f64 {
    pdf_scalar_raw(fit.kind, &fit.parameters, x)
}

pub(super) fn pdf_scalar_raw(kind: DistributionKind, params: &[f64], x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    match kind {
        DistributionKind::Normal => {
            let [mu, sigma] = two(params);
            if sigma <= 0.0 {
                return if x == mu { f64::INFINITY } else { 0.0 };
            }
            distribution_math::standard_normal_pdf((x - mu) / sigma) / sigma
        }
        DistributionKind::Exponential => {
            let mu = params[0];
            if x < 0.0 || mu <= 0.0 {
                0.0
            } else {
                (-x / mu).exp() / mu
            }
        }
        DistributionKind::Lognormal => {
            let [mu, sigma] = two(params);
            if x <= 0.0 || sigma <= 0.0 {
                0.0
            } else {
                distribution_math::standard_normal_pdf((x.ln() - mu) / sigma) / (x * sigma)
            }
        }
        DistributionKind::Gamma => {
            let [shape, scale] = two(params);
            if x < 0.0 || shape <= 0.0 || scale <= 0.0 {
                0.0
            } else if x == 0.0 && shape < 1.0 {
                f64::INFINITY
            } else if x == 0.0 && shape > 1.0 {
                0.0
            } else {
                ((shape - 1.0) * x.ln()
                    - x / scale
                    - gammaln_nonnegative_scalar(shape)
                    - shape * scale.ln())
                .exp()
            }
        }
        DistributionKind::Weibull => {
            let [scale, shape] = two(params);
            if x < 0.0 || scale <= 0.0 || shape <= 0.0 {
                0.0
            } else if x == 0.0 && shape < 1.0 {
                f64::INFINITY
            } else if x == 0.0 && shape > 1.0 {
                0.0
            } else {
                (shape / scale) * (x / scale).powf(shape - 1.0) * (-(x / scale).powf(shape)).exp()
            }
        }
        DistributionKind::Poisson => {
            let lambda = params[0];
            if x < 0.0 || x.fract() != 0.0 || lambda < 0.0 {
                0.0
            } else if lambda == 0.0 {
                if x == 0.0 {
                    1.0
                } else {
                    0.0
                }
            } else {
                (x * lambda.ln() - lambda - gammaln_nonnegative_scalar(x + 1.0)).exp()
            }
        }
    }
}

pub(super) fn cdf_scalar_with_tail(fit: &FittedDistribution, x: f64, tail: CdfTail) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let upper = matches!(tail, CdfTail::Upper);
    match fit.kind {
        DistributionKind::Normal => {
            let [mu, sigma] = two(&fit.parameters);
            if sigma <= 0.0 {
                let lower = if x < mu { 0.0 } else { 1.0 };
                if upper {
                    1.0 - lower
                } else {
                    lower
                }
            } else if upper {
                distribution_math::standard_normal_cdf((mu - x) / sigma)
            } else {
                distribution_math::standard_normal_cdf((x - mu) / sigma)
            }
        }
        DistributionKind::Exponential => {
            let mu = fit.parameters[0];
            if mu <= 0.0 || x < 0.0 {
                if upper {
                    1.0
                } else {
                    0.0
                }
            } else if upper {
                (-x / mu).exp()
            } else {
                -(-x / mu).exp_m1()
            }
        }
        DistributionKind::Lognormal => {
            let [mu, sigma] = two(&fit.parameters);
            if x <= 0.0 || sigma <= 0.0 {
                if upper {
                    1.0
                } else {
                    0.0
                }
            } else if upper {
                distribution_math::standard_normal_cdf((mu - x.ln()) / sigma)
            } else {
                distribution_math::standard_normal_cdf((x.ln() - mu) / sigma)
            }
        }
        DistributionKind::Gamma => {
            let [shape, scale] = two(&fit.parameters);
            if x <= 0.0 || shape <= 0.0 || scale <= 0.0 {
                if upper {
                    1.0
                } else {
                    0.0
                }
            } else if upper {
                distribution_math::regularized_gamma_q(shape, x / scale)
            } else {
                distribution_math::regularized_gamma_p(shape, x / scale)
            }
        }
        DistributionKind::Weibull => {
            let [scale, shape] = two(&fit.parameters);
            if x < 0.0 || scale <= 0.0 || shape <= 0.0 {
                if upper {
                    1.0
                } else {
                    0.0
                }
            } else if upper {
                (-(x / scale).powf(shape)).exp()
            } else {
                -(-(x / scale).powf(shape)).exp_m1()
            }
        }
        DistributionKind::Poisson => {
            let lambda = fit.parameters[0];
            if x < 0.0 {
                if upper {
                    1.0
                } else {
                    0.0
                }
            } else if lambda == 0.0 {
                if upper {
                    0.0
                } else {
                    1.0
                }
            } else if upper {
                distribution_math::regularized_gamma_p(x.floor() + 1.0, lambda)
            } else {
                distribution_math::regularized_gamma_q(x.floor() + 1.0, lambda)
            }
        }
    }
}

pub(super) fn icdf_scalar(
    fit: &FittedDistribution,
    p: f64,
    poll: impl Fn() -> BuiltinResult<()>,
) -> BuiltinResult<f64> {
    poll()?;
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return Ok(f64::NAN);
    }
    let value = match fit.kind {
        DistributionKind::Normal => {
            let [mu, sigma] = two(&fit.parameters);
            mu + sigma * distribution_math::standard_normal_inv(p)
        }
        DistributionKind::Exponential => {
            let mu = fit.parameters[0];
            if p == 1.0 {
                f64::INFINITY
            } else {
                -mu * (1.0 - p).ln()
            }
        }
        DistributionKind::Lognormal => {
            let [mu, sigma] = two(&fit.parameters);
            (mu + sigma * distribution_math::standard_normal_inv(p)).exp()
        }
        DistributionKind::Gamma => {
            let [shape, scale] = two(&fit.parameters);
            return invert_positive(
                p,
                shape * scale,
                |x| distribution_math::regularized_gamma_p(shape, x / scale),
                poll,
            );
        }
        DistributionKind::Weibull => {
            let [scale, shape] = two(&fit.parameters);
            if p == 1.0 {
                f64::INFINITY
            } else {
                scale * (-(1.0 - p).ln()).powf(1.0 / shape)
            }
        }
        DistributionKind::Poisson => {
            return distribution_math::poisson_inverse(p, fit.parameters[0], poll)
        }
    };
    Ok(value)
}
