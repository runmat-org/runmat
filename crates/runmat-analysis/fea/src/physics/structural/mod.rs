pub fn displacement_increment_norm(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    for (av, bv) in a.iter().zip(b.iter()) {
        let d = av - bv;
        sum += d * d;
    }
    sum.sqrt()
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransientAdaptivityInput {
    pub step_dt: f64,
    pub residual_norm: f64,
    pub residual_target: f64,
    pub min_dt: f64,
    pub max_dt: f64,
    pub converged: bool,
    pub retries: usize,
    pub adapt_nonconverged_shrink: f64,
    pub adapt_growth_exponent: f64,
    pub adapt_min_scale: f64,
    pub adapt_max_scale: f64,
    pub adapt_retry_growth_cap: f64,
    pub thermo_growth_limit: f64,
    pub thermo_nonconverged_shrink: f64,
}

pub fn recommend_next_time_step(input: TransientAdaptivityInput) -> f64 {
    if !input.converged {
        return (input.step_dt
            * input.adapt_nonconverged_shrink.clamp(0.2, 1.0)
            * input.thermo_nonconverged_shrink)
            .clamp(input.min_dt, input.max_dt);
    }

    let target = input.residual_target.max(1.0e-12);
    let ratio = (target / input.residual_norm.max(1.0e-12)).clamp(0.25, 4.0);
    let mut factor = ratio.powf(input.adapt_growth_exponent.clamp(0.1, 1.0));
    factor = factor.clamp(
        input.adapt_min_scale.clamp(0.2, 1.0),
        input.adapt_max_scale.clamp(1.0, 2.0),
    );
    if input.retries > 0 {
        factor = factor.min(input.adapt_retry_growth_cap.clamp(1.0, 1.5));
    } else if input.residual_norm <= target * 0.1 {
        factor = factor.max((1.0 + (input.adapt_max_scale - 1.0) * 0.6).clamp(1.0, 1.5));
    }
    factor = factor.min(input.thermo_growth_limit.max(0.5));
    (input.step_dt * factor).clamp(input.min_dt, input.max_dt)
}

pub fn nonlinear_iteration_damping(
    line_search_enabled: bool,
    refresh_tangent: bool,
    thermo_severity: f64,
) -> f64 {
    let mut damping = if line_search_enabled { 0.62 } else { 0.72 };
    if refresh_tangent {
        damping *= 0.85;
    }
    damping * (1.0 - 0.08 * thermo_severity).clamp(0.65, 1.0)
}

pub fn nonlinear_trial_residual(residual: f64, trial_scale: f64) -> f64 {
    residual * (0.85 * trial_scale + 0.1)
}

pub fn nonlinear_spike_threshold(max_newton_iters: usize, complexity_scale: f64) -> usize {
    ((max_newton_iters as f64) * 0.7 / complexity_scale.sqrt()).ceil() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn displacement_increment_norm_matches_expected_delta() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 0.0, 3.0];
        let norm = displacement_increment_norm(&a, &b);
        assert!((norm - 2.0).abs() < 1.0e-12);
    }

    #[test]
    fn transient_adaptivity_recommendation_stays_bounded() {
        let next = recommend_next_time_step(TransientAdaptivityInput {
            step_dt: 1.0e-3,
            residual_norm: 1.0e-7,
            residual_target: 1.0e-6,
            min_dt: 1.0e-6,
            max_dt: 2.0e-2,
            converged: true,
            retries: 0,
            adapt_nonconverged_shrink: 0.75,
            adapt_growth_exponent: 0.35,
            adapt_min_scale: 0.8,
            adapt_max_scale: 1.25,
            adapt_retry_growth_cap: 1.05,
            thermo_growth_limit: 1.0,
            thermo_nonconverged_shrink: 1.0,
        });
        assert!((1.0e-6..=2.0e-2).contains(&next));
    }

    #[test]
    fn nonlinear_policy_helpers_produce_finite_values() {
        let damping = nonlinear_iteration_damping(true, true, 0.8);
        let trial = nonlinear_trial_residual(0.1, 0.5);
        let spike = nonlinear_spike_threshold(24, 4.0);
        assert!(damping.is_finite() && damping > 0.0);
        assert!(trial.is_finite() && trial > 0.0);
        assert!(spike >= 1);
    }
}
