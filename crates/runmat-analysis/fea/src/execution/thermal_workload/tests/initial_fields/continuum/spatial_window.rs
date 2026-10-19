//! Independent one-dimensional consistent-P1 calculation; no RunMat matrix/solver.
//! It diagnoses the coarse fixture, not an oracle for the three-dimensional solve.
use super::{FINAL_TIME, GENERATION};
use std::f64::consts::PI;

fn continuum_heat(time: f64) -> f64 {
    (1..128)
        .step_by(2)
        .map(|mode| {
            let mode = mode as f64;
            8.0 * GENERATION / (PI * mode).powi(4) * (-(PI * mode).powi(2) * time).exp()
        })
        .sum()
}

fn consistent_p1_heat(cells: usize, time: f64) -> f64 {
    // Steady -T''=q gives exact P1 nodal values q*x*(1-x)/2. The consistent
    // tridiagonal mass and stiffness matrices share sine eigenvectors, with
    // lambda_j=6(1-cos(j*pi*h))/(h²*(2+cos(j*pi*h))). Discrete sine projection
    // of the ACTUAL steady nodal field supplies each amplitude. Its P1 volume
    // integral is h times the sum of interior nodal values.
    let h = 1.0 / cells as f64;
    (1..cells)
        .map(|mode| {
            let angle = mode as f64 * PI / cells as f64;
            let amplitude = 2.0
                * h
                * (1..cells)
                    .map(|node| {
                        let x = node as f64 * h;
                        0.5 * GENERATION * x * (1.0 - x) * (angle * node as f64).sin()
                    })
                    .sum::<f64>();
            let eigenvalue = 6.0 * (1.0 - angle.cos()) / (h * h * (2.0 + angle.cos()));
            let integral = h
                * (1..cells)
                    .map(|node| (angle * node as f64).sin())
                    .sum::<f64>();
            amplitude * (-eigenvalue * time).exp() * integral
        })
        .sum()
}

#[test]
fn independent_p1_reference_explains_the_failed_coarse_heat_gate() {
    let exact = continuum_heat(FINAL_TIME);
    let mut errors = Vec::new();
    for (cells, expected) in [
        (1, 1.0),
        (2, 0.385_013_45),
        (4, 0.101_216_78),
        (8, 0.025_488_58),
    ] {
        let heat = consistent_p1_heat(cells, FINAL_TIME);
        let error = (heat - exact).abs() / exact;
        assert!((error - expected).abs() < 1e-8, "cells={cells}: {error}");
        // At t=0 exact integration of the solved P1 interpolation gives this
        // independently derived trapezoidal heat content, including n=1 zero.
        let initial = GENERATION / 12.0 * (1.0 - 1.0 / (cells * cells) as f64);
        assert!((consistent_p1_heat(cells, 0.0) - initial).abs() < 1e-12);
        errors.push(error);
    }
    assert_eq!(consistent_p1_heat(1, FINAL_TIME), 0.0);
    assert!(
        errors[2] > 0.08,
        "original four-cell heat gate is unattainable here"
    );
    assert!(errors[3] < 0.08, "same accuracy limit on resolved mesh");
    assert!(errors[2..]
        .windows(2)
        .all(|pair| (pair[0] / pair[1]).log2() > 1.7));
}
