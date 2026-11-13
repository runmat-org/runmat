//! Incoming-only injection, tested against continuum references rather than prescribed total p.
use super::*;

fn source(model: &mut AnalysisModel, amplitude: [f64; 2]) {
    model.constraints.clear();
    model.signals.clear();
    // Reuse the shared metadata fixture, replacing only its typed physical action/signal.
    velocity_source(model, 0.0, 0.0);
    model.signals[0] = PhysicalSignal::Pressure(complex_signal(
        Pressure::new(amplitude[0]),
        Pressure::new(amplitude[1]),
    ));
    let action = model.excitations.pop().unwrap().excitation().clone();
    model
        .excitations
        .push(ExcitationAction::AcousticIncidentPressure(action));
}

fn solve(model: &AnalysisModel, mesh: &SolverMeshArtifact) -> CanonicalAcousticResponse {
    solve_canonical_harmonic(
        model,
        mesh,
        "acoustic",
        Frequency::new(0.2),
        AcousticHarmonicPolicy {
            relative_tolerance: 1e-11,
            ..Default::default()
        },
        1e-8,
    )
    .unwrap()
}

// witness: FEA-ACOUSTIC-SOURCE-001/incident @ acoustic_harmonic/tet4|tet10/numerical_kernel;
//   check: incident-impedance; boundary: Planar incident/impedance fixture, phase and net-power refinement only.
#[test]
fn incoming_pressure_reflection_transmission_and_net_power_refine() {
    let mut inventory = std::collections::BTreeSet::new();
    for quadratic in [false, true] {
        for rotated in [false, true] {
            let direction = if rotated {
                [
                    std::f64::consts::FRAC_1_SQRT_2,
                    std::f64::consts::FRAC_1_SQRT_2,
                    0.0,
                ]
            } else {
                [1.0, 0.0, 0.0]
            };
            for (phase, amplitude) in [[1.0, 0.4], [-0.4, 1.0]].into_iter().enumerate() {
                let expected_power = 1.16 / 12.0 * (1.0 - (7.0_f64 / 13.0).powi(2));
                let mut pressure_errors = Vec::new();
                let mut velocity_errors = Vec::new();
                for cells in [2, 4, 8] {
                    inventory.insert((quadratic, rotated, phase, cells));
                    let (mut model, mesh) =
                        interface::two_medium(cells, quadratic, amplitude, rotated);
                    source(&mut model, amplitude);
                    let result = solve(&model, &mesh);
                    let axial = |position: [f64; 3]| {
                        (0..3).map(|i| position[i] * direction[i]).sum::<f64>()
                    };
                    let pressure_error = (result
                        .projection
                        .nodes
                        .iter()
                        .zip(result.pressure_pa.canonical_values())
                        .map(|(node, value)| {
                            let reference = interface::exact_pressure(
                                axial(node.coordinates_m.canonical_value()),
                                amplitude,
                            );
                            (value.real - reference[0]).powi(2)
                                + (value.imaginary - reference[1]).powi(2)
                        })
                        .sum::<f64>()
                        / result.pressure_pa.canonical_values().len() as f64)
                        .sqrt();
                    let velocity_error = (result
                        .projection
                        .elements
                        .iter()
                        .zip(result.element_velocity_m_per_s.canonical_values())
                        .map(|(element, value)| {
                            let xs = std::array::from_fn(|i| {
                                axial(
                                    result.projection.nodes[element.node_indices[i]]
                                        .coordinates_m
                                        .canonical_value(),
                                )
                            });
                            let reference = interface::exact_mean_velocity(
                                xs,
                                xs.into_iter().sum::<f64>() / 4.0 < 0.5,
                                amplitude,
                            );
                            (0..3)
                                .map(|i| {
                                    (value.real[i] - direction[i] * reference[0]).powi(2)
                                        + (value.imaginary[i] - direction[i] * reference[1]).powi(2)
                                })
                                .sum::<f64>()
                        })
                        .sum::<f64>()
                        / result.projection.elements.len() as f64)
                        .sqrt();
                    pressure_errors.push(pressure_error);
                    velocity_errors.push(velocity_error);
                    assert_eq!(result.source_evidence.len(), 1);
                    let evidence = &result.source_evidence[0];
                    evidence.validate().unwrap();
                    assert!(evidence.accepted);
                    assert_eq!(
                        evidence.quantity,
                        runmat_analysis_core::PhysicalQuantityKind::Pressure
                    );
                    assert_eq!(evidence.signal_id, "drive");
                    assert!(result.relative_residual < 1e-10 && result.power_relative_error < 1e-8);
                    eprintln!("incident q={quadratic} rotated={rotated} phase={phase} cells={cells} p={pressure_error:e} v={velocity_error:e} net={:e} loss={:e}", result.mean_input_power_w.canonical_value(), result.mean_impedance_loss_w.canonical_value());
                    if cells == 8 {
                        assert!(pressure_error < 0.004 * 1.16_f64.sqrt() && velocity_error < 0.01);
                        assert!(
                            (result.mean_input_power_w.canonical_value() - expected_power).abs()
                                < 0.005
                        );
                        assert!(
                            (result.mean_impedance_loss_w.canonical_value() - expected_power).abs()
                                < 0.005
                        );
                    }
                }
                assert!(
                    pressure_errors
                        .windows(2)
                        .all(|p| p[0] / p[1] > if quadratic { 6.0 } else { 3.0 }),
                    "{pressure_errors:?}"
                );
                assert!(
                    velocity_errors
                        .windows(2)
                        .all(|v| v[0] / v[1] > if quadratic { 3.0 } else { 1.5 }),
                    "{velocity_errors:?}"
                );
            }
        }
    }
    assert_eq!(inventory.len(), 24);
}

#[test]
fn matched_incident_power_is_not_double_counted_at_small_amplitude() {
    for amplitude in [[1.0, 0.4], [1e-12, 4e-13]] {
        let mut model = model();
        model.material_assignments[0].target =
            PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("slab"));
        model.boundaries.push(AnalysisBoundary {
            boundary_id: "outlet".into(),
            target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                "slab_right",
            )),
            operator: BoundaryOperator::AcousticImpedance {
                impedance: SpecificAcousticImpedance::new(6.0),
            },
        });
        source(&mut model, amplitude);
        let result = solve(
            &model,
            &crate::topology::test_meshes::quadratic_slab_mesh(4),
        );
        let expected = amplitude.into_iter().map(|x| x * x).sum::<f64>() / 12.0;
        assert!((result.mean_input_power_w.canonical_value() / expected - 1.0).abs() < 1e-4);
        assert!((result.mean_impedance_loss_w.canonical_value() / expected - 1.0).abs() < 1e-4);
        assert!(result.power_relative_error < 1e-8);
    }
}

#[test]
fn incident_and_pressure_overlap_rejects_both_source_orders_and_duplicates() {
    for duplicate_incident in [false, true] {
        for incident_first in [false, true] {
            let (mut model, mesh) = interface::two_medium(2, false, [1.0, 0.4], false);
            source(&mut model, [1.0, 0.4]);
            let mut other = model.excitations[0].excitation().clone();
            model.excitations[0].excitation_mut().excitation_id =
                ExcitationId(if incident_first { "a" } else { "z" }.into());
            other.excitation_id = ExcitationId(if incident_first { "z" } else { "a" }.into());
            model.excitations.push(if duplicate_incident {
                ExcitationAction::AcousticIncidentPressure(other)
            } else {
                ExcitationAction::AcousticPressure(other)
            });
            model.excitation_cases[0].action_ids =
                vec![ExcitationId("a".into()), ExcitationId("z".into())];
            assert!(matches!(
                solve_canonical_harmonic(
                    &model,
                    &mesh,
                    "acoustic",
                    Frequency::new(0.2),
                    AcousticHarmonicPolicy::default(),
                    1e-8
                ),
                Err(AcousticHarmonicError::Boundary(_))
            ));
        }
    }
}

#[test]
fn incident_rejects_nonplanar_target_and_boundary_overlap() {
    let (mut model, mesh) = interface::two_medium(2, false, [1.0, 0.4], false);
    source(&mut model, [1.0, 0.4]);
    model.boundaries.push(AnalysisBoundary {
        boundary_id: "conflict".into(),
        target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_left")),
        operator: BoundaryOperator::AcousticRigidWall,
    });
    assert!(matches!(
        solve_canonical_harmonic(
            &model,
            &mesh,
            "acoustic",
            Frequency::new(0.2),
            AcousticHarmonicPolicy::default(),
            1e-8
        ),
        Err(AcousticHarmonicError::Boundary(_))
    ));
    model.boundaries.pop();
    model.excitations[0].excitation_mut().target = ExcitationTarget::Region(
        RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab")),
    );
    assert!(solve_canonical_harmonic(
        &model,
        &mesh,
        "acoustic",
        Frequency::new(0.2),
        AcousticHarmonicPolicy::default(),
        1e-8
    )
    .is_err());
}
