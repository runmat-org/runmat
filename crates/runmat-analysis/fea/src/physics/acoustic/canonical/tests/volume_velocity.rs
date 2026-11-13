//! Independent traveling wave with total incoming flow Q and inlet area A:
//! p=Z Q/A exp(-ikx), v=p/Z, P=Z |Q|²/(2A). Z=6, k=2pi f/c.
use super::*;
use crate::topology::test_meshes::quadratic_slab_mesh;
use runmat_analysis_core::{
    ComplexPhase, PairedConstraint, PairingPolicy, PeriodicConstraint, PeriodicTransform,
    PhaseShiftedPeriodicConstraint, PhysicalQuantityKind, Translation, VolumeFlowRate,
};

mod reference;

fn duct(
    cells: usize,
    quadratic: bool,
    area: f64,
    phase: bool,
) -> (AnalysisModel, SolverMeshArtifact) {
    let mut mesh = if quadratic {
        quadratic_slab_mesh(cells)
    } else {
        slab_mesh(cells)
    };
    for node in &mut mesh.topology.nodes {
        {
            let quantity_change = area;
            let quantity_index = 1;
            let quantity_slot = &mut node.coordinates_m;
            let mut quantity_coordinates = quantity_slot.canonical_value();
            quantity_coordinates[quantity_index] *= quantity_change;
            *quantity_slot = runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(quantity_coordinates);
        };
        if area != 1.0 {
            let [x, y, z] = node.coordinates_m.canonical_value();
            let a = std::f64::consts::FRAC_1_SQRT_2;
            node.coordinates_m = runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new([a * (x - y), a * (x + y), z]);
        }
    }
    mesh.seal_canonical_digest().unwrap();
    let mut model = model();
    model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("slab"));
    // Positive physical incoming flow is negative outward-normal flow.
    let q = if phase {
        [0.4 / 6.0, -1.0 / 6.0]
    } else {
        [-1.0 / 6.0, -0.4 / 6.0]
    };
    velocity_source(&mut model, 0.0, 0.0);
    model.signals[0] = PhysicalSignal::VolumeFlowRate(complex_signal(
        VolumeFlowRate::new(q[0]),
        VolumeFlowRate::new(q[1]),
    ));
    let mut action = model.excitations.pop().unwrap().excitation().clone();
    action.distribution = ExcitationDistribution::PreserveResultant;
    model
        .excitations
        .push(ExcitationAction::VolumeVelocity(action));
    model.boundaries.push(AnalysisBoundary {
        boundary_id: "matched".into(),
        target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_right")),
        operator: BoundaryOperator::AcousticImpedance {
            impedance: SpecificAcousticImpedance::new(6.0),
        },
    });
    (model, mesh)
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

// witness: FEA-ACOUSTIC-SOURCE-001/resultant @ acoustic_harmonic/tet4|tet10/numerical_kernel;
//   check: volume-velocity-source; boundary: Area scaling, complex phase and independent flow/power refinement on slab fixtures.
#[test]
fn total_volume_flow_area_phase_rotation_and_power_refine_against_continuum() {
    let mut count = 0;
    for quadratic in [false, true] {
        for area in [1.0, 2.0] {
            let direction = if area == 1.0 {
                [1.0, 0.0, 0.0]
            } else {
                [
                    std::f64::consts::FRAC_1_SQRT_2,
                    std::f64::consts::FRAC_1_SQRT_2,
                    0.0,
                ]
            };
            for phase in [false, true] {
                let amplitude = if phase {
                    [-0.4 / area, 1.0 / area]
                } else {
                    [1.0 / area, 0.4 / area]
                };
                let reference = |position: [f64; 3]| {
                    let x = position
                        .into_iter()
                        .zip(direction)
                        .map(|(x, d)| x * d)
                        .sum::<f64>();
                    let angle = std::f64::consts::PI * 0.2 * x;
                    [
                        amplitude[0] * angle.cos() + amplitude[1] * angle.sin(),
                        amplitude[1] * angle.cos() - amplitude[0] * angle.sin(),
                    ]
                };
                let power = 1.16 / (12.0 * area);
                let mut errors = Vec::new();
                for cells in [2, 4, 8] {
                    count += 1;
                    let (model, mesh) = duct(cells, quadratic, area, phase);
                    let result = solve(&model, &mesh);
                    let error = (result
                        .projection
                        .nodes
                        .iter()
                        .zip(result.pressure_pa.canonical_values())
                        .map(|(node, p)| {
                            let exact = reference(node.coordinates_m.canonical_value());
                            (p.real - exact[0]).powi(2) + (p.imaginary - exact[1]).powi(2)
                        })
                        .sum::<f64>()
                        / result.pressure_pa.canonical_values().len() as f64)
                        .sqrt();
                    errors.push(error);
                    assert!(result.relative_residual < 1e-10 && result.power_relative_error < 1e-8);
                    assert_eq!(result.source_evidence.len(), 1);
                    assert_eq!(
                        result.source_evidence[0].quantity,
                        PhysicalQuantityKind::VolumeFlowRate
                    );
                    assert!(result.source_evidence[0].accepted);
                    if cells == 8 {
                        assert!(error < 0.004 * 1.16_f64.sqrt() / area, "{errors:?}");
                        assert!(
                            (result.mean_input_power_w.canonical_value() - power).abs()
                                < 0.005 / area
                        );
                        assert!(
                            (result.mean_impedance_loss_w.canonical_value() - power).abs()
                                < 0.005 / area
                        );
                        let velocity_error = (result
                            .projection
                            .elements
                            .iter()
                            .zip(result.element_velocity_m_per_s.canonical_values())
                            .map(|(element, v)| {
                                let xs = std::array::from_fn(|i| {
                                    result.projection.nodes[element.node_indices[i]]
                                        .coordinates_m
                                        .canonical_value()
                                        .into_iter()
                                        .zip(direction)
                                        .map(|(x, d)| x * d)
                                        .sum::<f64>()
                                });
                                // Compare the retained volume mean with the exact
                                // continuum tetrahedron mean, not its centroid value.
                                let v_exact = reference::mean_velocity(xs, amplitude);
                                (0..3)
                                    .map(|i| {
                                        (v.real[i] - direction[i] * v_exact[0]).powi(2)
                                            + (v.imaginary[i] - direction[i] * v_exact[1]).powi(2)
                                    })
                                    .sum::<f64>()
                            })
                            .sum::<f64>()
                            / result.projection.elements.len() as f64)
                            .sqrt();
                        assert!(velocity_error < 0.01 / area, "v={velocity_error:e}");
                    }
                    eprintln!("volume q={quadratic} area={area} phase={phase} cells={cells} p={error:e} power={:e}", result.mean_input_power_w.canonical_value());
                }
                assert!(
                    errors
                        .windows(2)
                        .all(|e| e[0] / e[1] > if quadratic { 6.0 } else { 3.0 }),
                    "{errors:?}"
                );
            }
        }
    }
    assert_eq!(count, 24);
}

#[test]
fn total_volume_flow_and_normal_velocity_agree_without_duplicating_patch_resultant() {
    for quadratic in [false, true] {
        for area in [1.0, 2.0] {
            let (mut model, mesh) = duct(2, quadratic, area, false);
            let flow = solve(&model, &mesh);
            model.signals[0] = PhysicalSignal::Velocity(complex_signal(
                Velocity::new(-1.0 / (6.0 * area)),
                Velocity::new(-0.4 / (6.0 * area)),
            ));
            let mut action = model.excitations.pop().unwrap().excitation().clone();
            action.distribution = ExcitationDistribution::UniformKinematic;
            model
                .excitations
                .push(ExcitationAction::NormalVelocity(action));
            let velocity = solve(&model, &mesh);
            for (a, b) in flow
                .pressure_pa
                .canonical_values()
                .iter()
                .zip(velocity.pressure_pa.canonical_values())
            {
                assert!((a.real - b.real).abs() < 1e-9 && (a.imaginary - b.imaginary).abs() < 1e-9);
            }
            assert!(
                (flow.mean_input_power_w.canonical_value()
                    - velocity.mean_input_power_w.canonical_value())
                .abs()
                    < 1e-10
            );
        }
    }
}

#[test]
fn total_volume_flow_rejects_overlapping_sources_in_both_action_orders() {
    for other_kind in [0, 1, 2, 3] {
        for volume_first in [false, true] {
            let (mut model, mesh) = duct(2, false, 1.0, false);
            let mut action = model.excitations[0].excitation().clone();
            action.excitation_id = ExcitationId("other".into());
            action.signal_id = SignalId("other".into());
            let other = match other_kind {
                0 => {
                    let mut signal =
                        complex_signal(VolumeFlowRate::new(1.0), VolumeFlowRate::new(0.0));
                    signal.signal_id = action.signal_id.clone();
                    model.signals.push(PhysicalSignal::VolumeFlowRate(signal));
                    ExcitationAction::VolumeVelocity(action)
                }
                1 => {
                    let mut signal = complex_signal(Velocity::new(1.0), Velocity::new(0.0));
                    signal.signal_id = action.signal_id.clone();
                    model.signals.push(PhysicalSignal::Velocity(signal));
                    action.distribution = ExcitationDistribution::UniformKinematic;
                    ExcitationAction::NormalVelocity(action)
                }
                _ => {
                    let mut signal = complex_signal(Pressure::new(1.0), Pressure::new(0.0));
                    signal.signal_id = action.signal_id.clone();
                    model.signals.push(PhysicalSignal::Pressure(signal));
                    action.distribution = ExcitationDistribution::UniformKinematic;
                    if other_kind == 2 {
                        ExcitationAction::AcousticPressure(action)
                    } else {
                        ExcitationAction::AcousticIncidentPressure(action)
                    }
                }
            };
            model.excitation_cases[0]
                .action_ids
                .push(ExcitationId("other".into()));
            model.excitations.push(other);
            if !volume_first {
                model.excitations.reverse();
            }
            let result = solve_canonical_harmonic(
                &model,
                &mesh,
                "acoustic",
                Frequency::new(0.2),
                AcousticHarmonicPolicy::default(),
                1e-8,
            );
            assert!(
                matches!(result, Err(AcousticHarmonicError::Boundary(_))),
                "{result:?}"
            );
        }
    }
    for operator in [
        BoundaryOperator::AcousticRigidWall,
        BoundaryOperator::AcousticImpedance {
            impedance: SpecificAcousticImpedance::new(6.0),
        },
    ] {
        let (mut model, mesh) = duct(2, false, 1.0, false);
        model.boundaries.push(AnalysisBoundary {
            boundary_id: "conflict".into(),
            target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                "slab_left",
            )),
            operator,
        });
        let result = solve_canonical_harmonic(
            &model,
            &mesh,
            "acoustic",
            Frequency::new(0.2),
            AcousticHarmonicPolicy::default(),
            1e-8,
        );
        assert!(
            matches!(result, Err(AcousticHarmonicError::Boundary(_))),
            "{result:?}"
        );
    }
}

#[test]
fn total_volume_flow_rejects_prescribed_pressure_on_its_patch() {
    for quadratic in [false, true] {
        let (mut model, mesh) = duct(2, quadratic, 1.0, false);
        model
            .constraints
            .push(AnalysisConstraint::Acoustic(AcousticConstraint {
                constraint_id: "prescribed_source_patch".into(),
                kind: AcousticConstraintKind::PrescribedPressure {
                    target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                        "slab_left",
                    )),
                    value: PrescribedValue::Constant(Pressure::new(0.0)),
                },
            }));
        let result = solve_canonical_harmonic(
            &model,
            &mesh,
            "acoustic",
            Frequency::new(0.2),
            AcousticHarmonicPolicy::default(),
            1e-8,
        );
        assert!(
            matches!(&result, Err(AcousticHarmonicError::Boundary(message))
                if message.contains("volume velocity overlaps")),
            "prescribed pressure must conflict with total flow: {result:?}",
        );
    }
}

#[test]
fn total_volume_flow_rejects_periodic_and_bloch_primary_or_secondary_patches() {
    for quadratic in [false, true] {
        for phase in [None, Some(-std::f64::consts::PI * 0.2)] {
            for source_side in ["slab_left", "slab_right"] {
                let (mut model, mesh) = duct(2, quadratic, 1.0, false);
                // Remove the matched termination so a boundary/periodic conflict
                // cannot mask the specific total-flow/periodic source conflict.
                model.boundaries.clear();
                let ExcitationAction::VolumeVelocity(action) = &mut model.excitations[0] else {
                    unreachable!()
                };
                action.target = ExcitationTarget::Region(RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_boundary_selection(&(source_side)),
                ));
                let periodic = PeriodicConstraint {
                    pair: PairedConstraint {
                        primary: RegionTarget::boundary(
                            runmat_meshing_core::fixtures::fixture_face("slab_left"),
                        ),
                        secondary: RegionTarget::boundary(
                            runmat_meshing_core::fixtures::fixture_face("slab_right"),
                        ),
                        pairing: PairingPolicy::Matching,
                    },
                    transform: PeriodicTransform::Translation(Translation::new([1.0, 0.0, 0.0])),
                };
                let kind = if let Some(phase) = phase {
                    AcousticConstraintKind::PhaseShiftedPeriodic(PhaseShiftedPeriodicConstraint {
                        periodic,
                        phase: ComplexPhase::new(phase),
                    })
                } else {
                    AcousticConstraintKind::Periodic(periodic)
                };
                model
                    .constraints
                    .push(AnalysisConstraint::Acoustic(AcousticConstraint {
                        constraint_id: "paired_source_patch".into(),
                        kind,
                    }));
                let result = solve_canonical_harmonic(
                    &model,
                    &mesh,
                    "acoustic",
                    Frequency::new(0.2),
                    AcousticHarmonicPolicy::default(),
                    1e-8,
                );
                assert!(
                    matches!(&result, Err(AcousticHarmonicError::Boundary(message))
                        if message.contains("periodic pressure overlaps natural acoustic source")),
                    "q={quadratic} phase={phase:?} side={source_side}: {result:?}",
                );
            }
        }
    }
}
