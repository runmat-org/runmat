//! Continuous separated solution of -Δa=μJ on the unit square, zero perimeter
//! a, extruded in z with A=(0,0,a), B=(∂y a,-∂x a,0). No RunMat operator
//! or numerical solution enters the reference. The odd Fourier/hyperbolic series
//! has tails |a|≤2μ|J|/(π³N²), |grad a|≤4μ|J|/(π²N).
pub(super) fn fields(x: f64, y: f64, mu_j: f64, maximum: usize) -> ([f64; 3], [f64; 3]) {
    let mut a = 0.0;
    let mut dx = 0.0;
    let mut dy = 0.0;
    for n in (1..=maximum).step_by(2) {
        let k = n as f64 * std::f64::consts::PI;
        let left = (-k * x).exp();
        let right = (-k * (1.0 - x)).exp();
        let denominator = 1.0 + (-k).exp();
        a += 4.0 / k.powi(3) * (1.0 - (left + right) / denominator) * (k * y).sin();
        dx -= 4.0 / k.powi(2) * (right - left) / denominator * (k * y).sin();
        dy += 4.0 / k.powi(2) * (1.0 - (left + right) / denominator) * (k * y).cos();
    }
    ([0.0, 0.0, mu_j * a], [mu_j * dy, -mu_j * dx, 0.0])
}
pub(super) fn energy(mu_j2: f64, maximum: usize) -> f64 {
    let mut sum = 0.0;
    for m in (1..=maximum).step_by(2) {
        for n in (1..=maximum).step_by(2) {
            let mm = (m as f64).powi(2);
            let nn = (n as f64).powi(2);
            sum += 1.0 / (mm * nn * (mm + nn));
        }
    }
    32.0 * mu_j2 / std::f64::consts::PI.powi(6) * sum
}
