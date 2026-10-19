//! Independent affine-simplex moments for the traveling-wave volume mean.
//!
//! For uniformly distributed tetrahedral barycentric coordinates,
//! E[x^n] = 6 n! h_n(x0,x1,x2,x3)/(n+3)!, where h_n is the complete
//! homogeneous polynomial. Integrating the exponential series yields the
//! exact continuum mean up to its bounded Taylor remainder. This fixture has
//! |kx| <= 0.2 pi; 32 terms leave a remainder below 1e-40.
//! No production quadrature, discrete pressure, gradients, or recovery enter.

pub(super) fn mean_velocity(xs: [f64; 4], amplitude: [f64; 2]) -> [f64; 2] {
    let k = std::f64::consts::PI * 0.2;
    let mut h = [0.0; 32];
    h[0] = 1.0;
    for x in xs {
        for n in 1..h.len() {
            h[n] += x * h[n - 1];
        }
    }
    let mut coefficient = [1.0, 0.0];
    let mut mean = [0.0; 2];
    for (n, hn) in h.into_iter().enumerate() {
        if n > 0 {
            coefficient = [
                coefficient[1] * k / n as f64,
                -coefficient[0] * k / n as f64,
            ];
        }
        let moment = 6.0 * hn / ((n + 1) * (n + 2) * (n + 3)) as f64;
        for i in 0..2 {
            mean[i] += coefficient[i] * moment;
        }
    }
    [
        (amplitude[0] * mean[0] - amplitude[1] * mean[1]) / 6.0,
        (amplitude[0] * mean[1] + amplitude[1] * mean[0]) / 6.0,
    ]
}

#[test]
fn simplex_moments_have_the_constant_coordinate_limit_and_vertex_symmetry() {
    let amplitude = [1.0, 0.4];
    for x in [0.0, 0.5, 1.0] {
        let actual = mean_velocity([x; 4], amplitude);
        let angle = std::f64::consts::PI * 0.2 * x;
        let expected = [
            (amplitude[0] * angle.cos() + amplitude[1] * angle.sin()) / 6.0,
            (amplitude[1] * angle.cos() - amplitude[0] * angle.sin()) / 6.0,
        ];
        for i in 0..2 {
            assert!((actual[i] - expected[i]).abs() < 2e-15);
        }
    }
    let actual = mean_velocity([0.0, 0.2, 0.7, 1.0], amplitude);
    let permuted = mean_velocity([1.0, 0.7, 0.2, 0.0], amplitude);
    for i in 0..2 {
        assert!((actual[i] - permuted[i]).abs() < 2e-15);
    }
}
