//! Analytic fields defined independently of all discrete operators.
use super::fixture::ReferenceCase;
use std::f64::consts::PI;

pub fn rotate(v: [f64; 3]) -> [f64; 3] {
    [v[2], v[0], v[1]]
}
fn inverse_rotate(v: [f64; 3]) -> [f64; 3] {
    [v[1], v[2], v[0]]
}
pub fn velocity(case: ReferenceCase, x: [f64; 3], rotated: bool) -> [f64; 3] {
    let x = if rotated { inverse_rotate(x) } else { x };
    let u = match case {
        ReferenceCase::Couette => [x[1], 0.0, 0.0],
        // div u=0, rho(u.grad)u=(0.2,0,0), grad p=(-0.2,0,0),
        // full viscous stress constant; original body force is identically zero.
        ReferenceCase::Convective => [1.0 + x[1], 0.2, 0.0],
        ReferenceCase::Duct => [duct(x[1], x[2], 255), 0.0, 0.0],
    };
    if rotated {
        rotate(u)
    } else {
        u
    }
}
pub fn pressure(case: ReferenceCase, x: [f64; 3], rotated: bool) -> f64 {
    let x = if rotated { inverse_rotate(x) } else { x };
    if matches!(case, ReferenceCase::Convective) {
        -0.2 * x[0]
    } else {
        0.0
    }
}
pub fn outlet(x: [f64; 3], rotated: bool) -> bool {
    (if rotated { x[1] } else { x[0] } - 1.0).abs() < 1e-12
}

// Unit square duct: -mu(uyy+uzz)=G, G=mu=1; sine expansion of the constant
// forcing gives 4/(pi^3 n^3), not a fitted profile or pressure-drop correlation.
pub fn duct(y: f64, z: f64, last: usize) -> f64 {
    (1..=last)
        .step_by(2)
        .map(|n| {
            let k = n as f64 * PI;
            let a = k * (z - 0.5).abs();
            let b = k * 0.5;
            let cosh_ratio = ((a - b).exp() + (-a - b).exp()) / (1.0 + (-2.0 * b).exp());
            4.0 / PI.powi(3) * (k * y).sin() * (1.0 - cosh_ratio) / (n as f64).powi(3)
        })
        .sum()
}
pub fn duct_tail(last: usize) -> f64 {
    2.0 / (PI.powi(3) * (last as f64).powi(2))
}

pub fn duct_flow(last: usize) -> f64 {
    (1..=last)
        .step_by(2)
        .map(|n| {
            let k = n as f64 * PI;
            8.0 / PI.powi(4) / (n as f64).powi(4) * (1.0 - 2.0 / k * (k / 2.0).tanh())
        })
        .sum()
}
pub fn velocity_gradient(case: ReferenceCase, x: [f64; 3]) -> [[f64; 3]; 3] {
    match case {
        ReferenceCase::Couette | ReferenceCase::Convective => [[0.0, 1.0, 0.0], [0.0; 3], [0.0; 3]],
        ReferenceCase::Duct => {
            let mut gy = 0.0;
            let mut gz = 0.0;
            for n in (1..=255).step_by(2) {
                let k = n as f64 * PI;
                let a = k * (x[2] - 0.5).abs();
                let b = k / 2.0;
                let denom = 1.0 + (-2.0 * b).exp();
                let c = ((a - b).exp() + (-a - b).exp()) / denom;
                let s = ((a - b).exp() - (-a - b).exp()) / denom * (x[2] - 0.5).signum();
                gy += 4.0 / PI.powi(2) / (n as f64).powi(2) * (k * x[1]).cos() * (1.0 - c);
                gz -= 4.0 / PI.powi(2) / (n as f64).powi(2) * (k * x[1]).sin() * s;
            }
            [[0.0, gy, gz], [0.0; 3], [0.0; 3]]
        }
    }
}
