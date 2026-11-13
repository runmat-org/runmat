//! Closed-form continuum mode, derived without finite-element matrices or quadrature.
use super::*;
pub(super) fn shape(x: f64) -> f64 {
    (std::f64::consts::FRAC_PI_2 * x).sin()
}

pub(super) fn state(time: f64, damped: bool) -> [f64; 3] {
    let omega_squared = YOUNGS * std::f64::consts::FRAC_PI_2.powi(2);
    let gamma = if damped {
        (ALPHA + BETA * omega_squared) / 2.0
    } else {
        0.0
    };
    let omega_d = (omega_squared - gamma * gamma).sqrt();
    let decay = (-gamma * time).exp();
    let q = AMPLITUDE * decay * ((omega_d * time).cos() + gamma / omega_d * (omega_d * time).sin());
    let v = -AMPLITUDE * decay * omega_squared / omega_d * (omega_d * time).sin();
    let a = -2.0 * gamma * v - omega_squared * q;
    [q, v, a]
}

pub(super) fn energies(time: f64, damped: bool) -> [f64; 3] {
    let [q, v, _] = state(time, damped);
    // int_0^1 sin²(lambda*x) dx = int_0^1 cos²(lambda*x) dx = 1/2.
    let elastic = YOUNGS * std::f64::consts::FRAC_PI_2.powi(2) * q * q / 4.0;
    let kinetic = v * v / 4.0;
    let initial = YOUNGS * std::f64::consts::FRAC_PI_2.powi(2) * AMPLITUDE.powi(2) / 4.0;
    [elastic, kinetic, initial - elastic - kinetic]
}

#[test]
fn analytic_mode_satisfies_initial_conditions_and_continuum_energy() {
    for damped in [false, true] {
        let [q, v, a] = state(0.0, damped);
        assert_eq!(q, AMPLITUDE);
        assert_eq!(v, 0.0);
        assert!((a + YOUNGS * std::f64::consts::FRAC_PI_2.powi(2) * AMPLITUDE).abs() < 1.0e-14);
        assert_eq!(energies(0.0, damped)[2], 0.0);
    }
    for time in [0.025, 0.05, 0.1, 0.2] {
        assert!(energies(time, false)[2].abs() < 1.0e-16);
        assert!(energies(time, true)[2] > 0.0);
    }
}
