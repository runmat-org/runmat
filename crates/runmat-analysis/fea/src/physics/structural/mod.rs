pub(crate) mod dynamics;
pub(crate) mod frame;
pub mod material;
pub(crate) mod topology;

pub fn displacement_increment_norm(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    for (av, bv) in a.iter().zip(b.iter()) {
        let d = av - bv;
        sum += d * d;
    }
    sum.sqrt()
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
    fn nonlinear_policy_helpers_produce_finite_values() {
        let damping = nonlinear_iteration_damping(true, true, 0.8);
        let trial = nonlinear_trial_residual(0.1, 0.5);
        let spike = nonlinear_spike_threshold(24, 4.0);
        assert!(damping.is_finite() && damping > 0.0);
        assert!(trial.is_finite() && trial > 0.0);
        assert!(spike >= 1);
    }
}
