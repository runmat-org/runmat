use super::*;

pub(super) fn random_samples(fit: &FittedDistribution, len: usize) -> BuiltinResult<Vec<f64>> {
    match fit.kind {
        DistributionKind::Normal => {
            let [mu, sigma] = two(&fit.parameters);
            random::generate_normal_scaled(mu, sigma, len, RANDOM_NAME)
        }
        DistributionKind::Exponential => {
            random::generate_exponential(fit.parameters[0].max(MIN_POSITIVE), len, RANDOM_NAME)
        }
        DistributionKind::Lognormal => {
            let [mu, sigma] = two(&fit.parameters);
            random::generate_normal_scaled(mu, sigma, len, RANDOM_NAME)
                .map(|values| values.into_iter().map(f64::exp).collect())
        }
        DistributionKind::Gamma => {
            let [shape, scale] = two(&fit.parameters);
            random::generate_gamma_shape_scale(&[shape], &[scale], len, RANDOM_NAME)
        }
        DistributionKind::Weibull => {
            let [scale, shape] = two(&fit.parameters);
            random::generate_weibull(&[scale], &[shape], len, RANDOM_NAME)
        }
        DistributionKind::Poisson => {
            let uniforms = random::generate_uniform(len, RANDOM_NAME)?;
            uniforms
                .into_iter()
                .map(|u| distribution_math::poisson_inverse(u, fit.parameters[0], || Ok(())))
                .collect()
        }
    }
}
