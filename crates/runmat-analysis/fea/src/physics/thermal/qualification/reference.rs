//! Independent SI conduction/gray-radiation scalar reference; no FE operator.
use super::{AMBIENT, EMISSIVITY, K, SIGMA};
pub fn steady_surface_and_power() -> (f64, f64) {
    let residual = |surface: f64| {
        K * (400.0 - surface) - EMISSIVITY * SIGMA * (surface.powi(4) - AMBIENT.powi(4))
    };
    let (mut low, mut high) = (AMBIENT, 400.0);
    for _ in 0..64 {
        let middle = 0.5 * (low + high);
        if residual(middle) > 0.0 {
            low = middle;
        } else {
            high = middle;
        }
    }
    let surface = 0.5 * (low + high);
    let power = K * (400.0 - surface);
    (surface, power)
}

pub fn cooling_temperature(x: f64, time: f64) -> f64 {
    400.0 - 10.0 * x - 20.0 * time * time
}
