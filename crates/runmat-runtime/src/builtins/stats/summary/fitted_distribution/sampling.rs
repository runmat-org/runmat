use super::*;

pub(super) fn random_samples(
    turn: &mut random::RandomTurn,
    fit: &FittedDistribution,
    len: usize,
    poll: impl Fn() -> BuiltinResult<()>,
) -> BuiltinResult<Vec<f64>> {
    poll()?;
    let samples = match fit.kind {
        DistributionKind::Normal => {
            let [mu, sigma] = two(&fit.parameters);
            random::generate_normal_scaled(turn, mu, sigma, len, RANDOM_NAME)
        }
        DistributionKind::Exponential => random::generate_exponential(
            turn,
            fit.parameters[0].max(MIN_POSITIVE),
            len,
            RANDOM_NAME,
        ),
        DistributionKind::Lognormal => {
            let [mu, sigma] = two(&fit.parameters);
            random::generate_normal_scaled(turn, mu, sigma, len, RANDOM_NAME)
                .map(|values| values.into_iter().map(f64::exp).collect())
        }
        DistributionKind::Gamma => {
            let [shape, scale] = two(&fit.parameters);
            random::generate_gamma_shape_scale(turn, &[shape], &[scale], len, RANDOM_NAME)
        }
        DistributionKind::Weibull => {
            let [scale, shape] = two(&fit.parameters);
            random::generate_weibull(turn, &[scale], &[shape], len, RANDOM_NAME)
        }
        DistributionKind::Poisson => {
            let uniforms = random::generate_uniform(turn, len, RANDOM_NAME)?;
            uniforms
                .into_iter()
                .map(|u| distribution_math::poisson_inverse(u, fit.parameters[0], &poll))
                .collect()
        }
    }?;
    poll()?;
    Ok(samples)
}
