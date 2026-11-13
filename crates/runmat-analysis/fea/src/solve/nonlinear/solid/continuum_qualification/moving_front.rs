//! Independent monotonic uniaxial-strain history from equilibrium and J2 consistency.
//! No constitutive integration or finite-element solve supplies the reference.
use super::*;

mod captured_path;
mod cell_norm;
mod constitutive;
mod representability;
mod verification;

#[derive(Debug, Clone, Copy)]
struct Reference {
    displacement: f64,
    strain: f64,
    alpha: f64,
    stress: [f64; 6],
}

fn elastic_tangent() -> f64 {
    bulk() + 4.0 * mu() / 3.0
}

fn front(factor: f64) -> f64 {
    assert!(factor.is_finite() && (0.0..=1.0).contains(&factor));
    if factor == 0.0 {
        return 1.0;
    }
    ((elastic_tangent() * YIELD / (2.0 * mu() * factor) - stress(0.0)[0]) / (tangent() * B))
        .clamp(0.0, 1.0)
}

fn reference(x: f64, factor: f64) -> Reference {
    assert!(x.is_finite() && (0.0..=1.0).contains(&x));
    let r = front(factor);
    let s0 = stress(0.0)[0];
    let slope = tangent() * B;
    let axial_stress = factor * (s0 + slope * x);
    // A clamped front at x=0 means the entire domain is plastic, including the
    // endpoint. Classify by stress, not x <= clamped_front.
    let strain = if axial_stress <= elastic_tangent() * YIELD / (2.0 * mu()) {
        axial_stress / elastic_tangent()
    } else {
        (axial_stress - stress_offset()) / tangent()
    };
    let alpha = ((2.0 * mu() * strain - YIELD) / (3.0 * mu() + HARDENING)).max(0.0);
    let elastic_end = x.min(r);
    let elastic_integral =
        factor * (s0 * elastic_end + 0.5 * slope * elastic_end.powi(2)) / elastic_tangent();
    let plastic_integral = if x > r {
        ((factor * s0 - stress_offset()) * (x - r) + 0.5 * factor * slope * (x.powi(2) - r.powi(2)))
            / tangent()
    } else {
        0.0
    };
    let lateral = bulk() * strain - 2.0 * mu() * strain / 3.0 + mu() * alpha;
    Reference {
        displacement: elastic_integral + plastic_integral,
        strain,
        alpha,
        stress: [axial_stress, lateral, lateral, 0.0, 0.0, 0.0],
    }
}

fn integrals(factor: f64) -> ([f64; 3], f64) {
    fn square(c: f64, d: f64, a: f64, b: f64) -> f64 {
        c * c * (b - a) + c * d * (b * b - a * a) + d * d * (b.powi(3) - a.powi(3)) / 3.0
    }
    fn linear(c: f64, d: f64, a: f64, b: f64) -> f64 {
        c * (b - a) + 0.5 * d * (b * b - a * a)
    }
    let r = front(factor);
    let s0 = stress(0.0)[0];
    let slope = tangent() * B;
    let elastic = (
        factor * s0 / elastic_tangent(),
        factor * slope / elastic_tangent(),
    );
    let plastic = (
        (factor * s0 - stress_offset()) / tangent(),
        factor * slope / tangent(),
    );
    let plastic_alpha = (
        (2.0 * mu() * plastic.0 - YIELD) / (3.0 * mu() + HARDENING),
        2.0 * mu() * plastic.1 / (3.0 * mu() + HARDENING),
    );
    let mut energy = [0.0; 3];
    let mut lateral = 0.0;
    for (a, b, e, p) in [
        (0.0, r, elastic, (0.0, 0.0)),
        (r, 1.0, plastic, plastic_alpha),
    ] {
        energy[0] += 0.5 * bulk() * square(e.0, e.1, a, b)
            + 2.0 * mu() / 3.0 * square(e.0 - 1.5 * p.0, e.1 - 1.5 * p.1, a, b);
        energy[1] += 0.5 * HARDENING * square(p.0, p.1, a, b);
        energy[2] += YIELD * linear(p.0, p.1, a, b);
        lateral += 2.0
            * ((bulk() - 2.0 * mu() / 3.0) * linear(e.0, e.1, a, b)
                + mu() * linear(p.0, p.1, a, b));
    }
    (energy, lateral)
}

#[test]
fn independent_front_satisfies_equilibrium_consistency_and_compatibility() {
    let first_yield = elastic_tangent() * YIELD / (2.0 * mu() * stress(1.0)[0]);
    let full_yield = elastic_tangent() * YIELD / (2.0 * mu() * stress(0.0)[0]);
    assert!(first_yield > 0.125 && first_yield < 0.25);
    assert!(full_yield > 0.375 && full_yield < 0.5);
    assert_eq!(front(0.125), 1.0);
    assert!(front(0.25) > 0.0 && front(0.25) < 1.0);
    assert!(front(0.375) > 0.0 && front(0.375) < front(0.25));
    assert_eq!(front(0.5), 0.0);
    for factor in [0.0, 0.125, first_yield, 0.25, 0.375, full_yield, 0.5, 1.0] {
        let (energies, moment) = integrals(factor);
        assert!(energies.iter().all(|v| *v >= -1e-14));
        assert!(moment >= 0.0);
        if factor <= first_yield {
            assert!(energies[1].abs() < 1e-14 && energies[2].abs() < 1e-14);
        }
        if factor == 1.0 {
            for (actual, final_value) in energies.iter().zip(energy_references()) {
                assert!((actual - final_value).abs() < 1e-14);
            }
            assert!((moment - 2.0 * stress(0.5)[1]).abs() < 1e-12);
        }
        assert_eq!(reference(0.0, factor).displacement, 0.0);
        for i in 0..=100 {
            let x = i as f64 / 100.0;
            let value = reference(x, factor);
            let equivalent = (value.stress[0] - value.stress[1]).abs();
            let yield_stress = YIELD + HARDENING * value.alpha;
            assert!(equivalent <= yield_stress + 1e-12);
            if value.alpha > 0.0 {
                assert!((equivalent - yield_stress).abs() < 1e-12);
            }
            // Reconstruct stress directly from elastic strain, not the piecewise inversion.
            let reconstructed = elastic_tangent() * value.strain - 2.0 * mu() * value.alpha;
            assert!((reconstructed - value.stress[0]).abs() < 1e-12);
            if x > 0.0 && x < 1.0 {
                let dx = 1e-6;
                let left = reference(x - dx, factor);
                let right = reference(x + dx, factor);
                let derivative = (right.displacement - left.displacement) / (2.0 * dx);
                assert!((derivative - value.strain).abs() < 1e-8);
                // dσxx/dx + f_x = 0 for the exact fixed dead-load fixture at this factor.
                let balance =
                    (right.stress[0] - left.stress[0]) / (2.0 * dx) - factor * tangent() * B;
                assert!(balance.abs() < 1e-8);
            }
            if factor == 1.0 {
                assert!((value.displacement - displacement(x)).abs() < 1e-14);
                assert!((value.alpha - alpha(x)).abs() < 1e-14);
                for (actual, final_value) in value.stress.iter().zip(stress(x)) {
                    assert!((actual - final_value).abs() < 1e-12);
                }
            }
        }
        let r = front(factor);
        if r > 1e-7 && r < 1.0 - 1e-7 {
            let left = reference(r - 1e-8, factor);
            let right = reference(r + 1e-8, factor);
            assert!((left.strain - right.strain).abs() < 1e-8);
            assert!((left.displacement - right.displacement).abs() < 1e-8);
        }
    }
}
