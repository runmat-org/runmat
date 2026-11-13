//! Additional finer-mesh evidence, not a replacement for the failed coarse gate.
use super::*;

#[test]
fn low_wave_number_tet10_finer_pressure_velocity_and_power_converge() {
    let mut failures = Vec::new();
    let mut inventory = std::collections::BTreeSet::new();
    for rotated in [false, true] {
        let (rho, c, direction) = if rotated {
            (
                5.0,
                4.0,
                [
                    std::f64::consts::FRAC_1_SQRT_2,
                    std::f64::consts::FRAC_1_SQRT_2,
                    0.0,
                ],
            )
        } else {
            (3.0, 2.0, [1.0, 0.0, 0.0])
        };
        let frequency = 0.2 * c / 2.0;
        let k = std::f64::consts::TAU * frequency / c;
        let impedance = rho * c;
        let axial = |position: [f64; 3]| {
            position
                .into_iter()
                .zip(direction)
                .map(|(x, d)| x * d)
                .sum::<f64>()
        };
        for (phase, amplitude) in [[1.0, 0.4], [-0.4, 1.0]].into_iter().enumerate() {
            let norm = amplitude[0] * amplitude[0] + amplitude[1] * amplitude[1];
            let mut pressure_errors = Vec::new();
            let mut velocity_errors = Vec::new();
            for cells in [2, 4, 8] {
                assert!(inventory.insert((rotated, phase, cells)));
                let result = solve(cells, true, frequency, amplitude, rotated);
                let pressure = (result
                    .projection
                    .nodes
                    .iter()
                    .zip(result.pressure_pa.canonical_values())
                    .map(|(node, p)| {
                        let x = axial(node.coordinates_m.canonical_value());
                        let reference = multiply(amplitude, [(k * x).cos(), -(k * x).sin()]);
                        (p.real - reference[0]).powi(2) + (p.imaginary - reference[1]).powi(2)
                    })
                    .sum::<f64>()
                    / result.pressure_pa.canonical_values().len() as f64)
                    .sqrt();
                let velocity = (result
                    .projection
                    .elements
                    .iter()
                    .zip(result.element_velocity_m_per_s.canonical_values())
                    .map(|(element, v)| {
                        let xs = std::array::from_fn(|i| {
                            axial(
                                result.projection.nodes[element.node_indices[i]]
                                    .coordinates_m
                                    .canonical_value(),
                            )
                        });
                        let p = multiply(amplitude, mean_wave(xs, k));
                        (0..3)
                            .map(|axis| {
                                (v.real[axis] - direction[axis] * p[0] / impedance).powi(2)
                                    + (v.imaginary[axis] - direction[axis] * p[1] / impedance)
                                        .powi(2)
                            })
                            .sum::<f64>()
                    })
                    .sum::<f64>()
                    / result.projection.elements.len() as f64)
                    .sqrt();
                pressure_errors.push(pressure);
                velocity_errors.push(velocity);
                let inlet = outward_recovered_power(&result, "slab_left");
                let outlet = outward_recovered_power(&result, "slab_right");
                let exact_power = norm / (2.0 * impedance);
                eprintln!("radiation finer rotated={rotated} phase={phase} cells={cells} pressure={pressure} velocity={velocity} inlet={inlet} outlet={outlet} exact_power={exact_power} residual={} power_closure={}",result.relative_residual,result.power_relative_error);
                if !(result.relative_residual < 1e-10
                    && result.power_relative_error < 1e-8
                    && inlet < 0.0
                    && outlet > 0.0
                    && result.mean_impedance_loss_w.canonical_value() > 0.0
                    && (inlet + exact_power).abs() < 0.005
                    && (outlet - exact_power).abs() < 0.005
                    && (result.mean_impedance_loss_w.canonical_value() - exact_power).abs() < 0.005)
                {
                    failures.push(format!("physical power/algebraic gate rotated={rotated} phase={phase} cells={cells}"));
                }
            }
            if !(pressure_errors
                .windows(2)
                .all(|pair| pair[0] / pair[1] > 6.0)
                && velocity_errors
                    .windows(2)
                    .all(|pair| pair[0] / pair[1] > 3.0)
                && pressure_errors[2] < 0.004 * norm.sqrt()
                && velocity_errors[2] < 0.01)
            {
                failures.push(format!("finer refinement gate rotated={rotated} phase={phase}: pressure={pressure_errors:?} velocity={velocity_errors:?}"));
            }
        }
    }
    let expected = [false, true]
        .into_iter()
        .flat_map(|rotated| {
            (0..2).flat_map(move |phase| {
                [2, 4, 8]
                    .into_iter()
                    .map(move |cells| (rotated, phase, cells))
            })
        })
        .collect();
    assert_eq!(
        inventory, expected,
        "finer radiation solve inventory differs"
    );
    assert!(
        failures.is_empty(),
        "additional finer gates failed: {}",
        failures.join("\n")
    );
}
