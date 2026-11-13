//! Independent continuum references, not agreement between two discrete solves.
use super::*;
use crate::topology::test_meshes::quadratic_slab_mesh;

fn solve_duct(
    cells: usize,
    quadratic: bool,
    frequency: f64,
    matched: bool,
) -> CanonicalAcousticResponse {
    let mesh = if quadratic {
        quadratic_slab_mesh(cells)
    } else {
        slab_mesh(cells)
    };
    let mut model = model();
    model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("slab"));
    model.sequence.steps[0].independent_domain = IndependentDomain::FrequencyHertz(vec![frequency]);
    velocity_source(&mut model, -1.0 / 6.0, -0.4 / 6.0);
    // The shared source is constant over this extended frequency schedule.
    let PhysicalSignal::Velocity(signal) = &mut model.signals[0] else {
        unreachable!()
    };
    let SignalRepresentation::Tabulated(table) = &mut signal.representation else {
        unreachable!()
    };
    for knot in &mut table.knots {
        knot.abscissa *= 2.0;
    }
    model.boundaries.push(AnalysisBoundary {
        boundary_id: "termination".into(),
        target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_right")),
        operator: if matched {
            BoundaryOperator::AcousticImpedance {
                impedance: SpecificAcousticImpedance::new(6.0),
            }
        } else {
            BoundaryOperator::AcousticRigidWall
        },
    });
    solve_canonical_harmonic(
        &model,
        &mesh,
        "acoustic",
        Frequency::new(frequency),
        AcousticHarmonicPolicy {
            relative_tolerance: 1e-11,
            ..Default::default()
        },
        1e-8,
    )
    .unwrap()
}

fn pressure_error(
    result: &CanonicalAcousticResponse,
    reference: impl Fn(f64) -> (f64, f64),
) -> f64 {
    let mut error = 0.0;
    let mut scale = 0.0;
    for (node, pressure) in result
        .projection
        .nodes
        .iter()
        .zip(result.pressure_pa.canonical_values())
    {
        let (real, imaginary) = reference(node.coordinates_m.canonical_value()[0]);
        error += (pressure.real - real).powi(2) + (pressure.imaginary - imaginary).powi(2);
        scale += real.powi(2) + imaginary.powi(2);
    }
    (error / scale).sqrt()
}

// witness: FEA-ACOUSTIC-NUMERICS-001/duct @ acoustic_harmonic/tet10/numerical_kernel;
//   check: traveling-wave; boundary: Tet10 slab at two frequencies, three refinements; exact traveling-wave references,
//   not all acoustic geometries.
#[test]
fn tet10_traveling_wave_phase_pressure_velocity_and_power_converge() {
    for frequency in [0.2, 0.5] {
        let k = std::f64::consts::TAU * frequency / 2.0;
        let mut errors = Vec::new();
        for cells in [1, 2, 4] {
            let result = solve_duct(cells, true, frequency, true);
            let reference = |x: f64| {
                let phase = k * x;
                (
                    phase.cos() + 0.4 * phase.sin(),
                    0.4 * phase.cos() - phase.sin(),
                )
            };
            let error = pressure_error(&result, reference);
            errors.push(error);
            assert!(result.relative_residual <= 1e-10);
            assert!(result.power_relative_error < 1e-8);
            assert_eq!(result.source_evidence.len(), 1);
            assert!(result.source_evidence[0].accepted);
            assert!((result.mean_impedance_loss_w.canonical_value() - 1.16 / 12.0).abs() < 0.01);
            if cells == 4 {
                assert!(error < 0.004, "f={frequency}, errors={errors:?}");
                let velocity_error = result
                    .projection
                    .elements
                    .iter()
                    .zip(result.element_velocity_m_per_s.canonical_values())
                    .map(|(element, velocity)| {
                        let (real, imaginary) = reference(element.centroid_m[0]);
                        (velocity.real[0] - real / 6.0).powi(2)
                            + (velocity.imaginary[0] - imaginary / 6.0).powi(2)
                            + velocity.real[1].powi(2)
                            + velocity.real[2].powi(2)
                            + velocity.imaginary[1].powi(2)
                            + velocity.imaginary[2].powi(2)
                    })
                    .sum::<f64>();
                assert!((velocity_error / result.projection.elements.len() as f64).sqrt() < 0.01);
                let intensity_error = result
                    .element_mean_intensity_w_per_m2
                    .canonical_values()
                    .iter()
                    .map(|i| (i[0] - 1.16 / 12.0).powi(2) + i[1].powi(2) + i[2].powi(2))
                    .sum::<f64>();
                assert!((intensity_error / result.projection.elements.len() as f64).sqrt() < 0.005);
                let energy_error = result
                    .element_mean_energy_density_j_per_m3
                    .canonical_values()
                    .iter()
                    .map(|e| (e - 1.16 / 24.0).powi(2))
                    .sum::<f64>();
                assert!((energy_error / result.projection.elements.len() as f64).sqrt() < 0.005);
            }
        }
        eprintln!("Tet10 matched f={frequency}: {errors:?}");
        assert!(
            errors.windows(2).all(|p| p[0] / p[1] > 6.0),
            "f={frequency}: {errors:?}"
        );
    }
}

// witness: FEA-ACOUSTIC-NUMERICS-001/cavity @ acoustic_harmonic/tet4|tet10/numerical_kernel;
//   check: forced-cavity; boundary: Forced axial cavity at 0.9/1.1 times first pole, phase/pressure/energy convergence;
//   no complete multidimensional modal-spectrum credit.
#[test]
fn tet4_and_tet10_forced_rigid_cavity_converge_on_both_sides_of_axial_resonance() {
    // Unit cube, rigid side/right walls, prescribed outward v_n=-(1+.4i)/6 at x=0.
    // p=i*rho*c*v_n*cos(k(1-x))/sin(k), v_x=-v_n*sin(k(1-x))/sin(k).
    // f_n=n*c/(2L)=n Hz. Do not solve or claim uniqueness at the pole itself.
    for quadratic in [false, true] {
        for frequency in [0.9, 1.1] {
            let k = std::f64::consts::PI * frequency;
            let mut errors = Vec::new();
            let refinements = if quadratic { [1, 2, 4] } else { [4, 8, 16] };
            for cells in refinements {
                let result = solve_duct(cells, quadratic, frequency, false);
                let reference = |x: f64| {
                    let shape = (k * (1.0 - x)).cos() / k.sin();
                    (0.4 * shape, -shape)
                };
                errors.push(pressure_error(&result, reference));
                assert!(result.relative_residual <= 1e-10);
                assert!(result.mean_impedance_loss_w.canonical_value().abs() < 1e-14);
                assert!(result.power_relative_error < 1e-8);
                if cells == refinements[2] {
                    assert!(
                        errors[2] < 0.10,
                        "quadratic={quadratic}, f={frequency}: {errors:?}"
                    );
                    // Boundary-average phase must flip across the first axial pole.
                    let boundary = result
                        .projection
                        .nodes
                        .iter()
                        .zip(result.pressure_pa.canonical_values())
                        .filter(|(n, _)| n.coordinates_m.canonical_value()[0] == 0.0)
                        .map(|(_, p)| p.imaginary)
                        .sum::<f64>();
                    assert_eq!(boundary.is_sign_positive(), frequency < 1.0);
                    assert!(result
                        .element_mean_intensity_w_per_m2
                        .canonical_values()
                        .iter()
                        .flatten()
                        .all(|i| i.abs() < 1e-8));
                    let reference_energy = 1.16 / (48.0 * k.sin().powi(2));
                    let energy_error = result
                        .element_mean_energy_density_j_per_m3
                        .canonical_values()
                        .iter()
                        .map(|e| (e / reference_energy - 1.0).powi(2))
                        .sum::<f64>();
                    let energy_error =
                        (energy_error / result.projection.elements.len() as f64).sqrt();
                    assert!(energy_error < 0.15, "quadratic={quadratic}, f={frequency}, energy error={energy_error}, pressure={errors:?}");
                }
            }
            eprintln!("rigid quadratic={quadratic}, f={frequency}: {errors:?}");
            assert!(
                errors.windows(2).all(|p| p[0] > p[1]),
                "quadratic={quadratic}, f={frequency}: {errors:?}"
            );
        }
    }
}
