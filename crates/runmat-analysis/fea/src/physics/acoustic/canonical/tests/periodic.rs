//! Independent anchored plane-wave Bloch evidence, not general periodic qualification.
use super::*;
use crate::topology::test_meshes::quadratic_slab_mesh;
use runmat_analysis_core::{
    ComplexPhase, PairedConstraint, PairingPolicy, PeriodicConstraint, PeriodicTransform,
    PhaseShiftedPeriodicConstraint, Translation,
};

mod reference;

fn fixture(
    cells: usize,
    quadratic: bool,
    amplitude: [f64; 2],
) -> (AnalysisModel, SolverMeshArtifact) {
    let mesh = if quadratic {
        quadratic_slab_mesh(cells)
    } else {
        slab_mesh(cells)
    };
    let mut model = model();
    model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("slab"));
    model.sequence.steps[0].independent_domain = IndependentDomain::FrequencyHertz(vec![0.2]);
    model.signals.push(PhysicalSignal::Pressure(complex_signal(
        Pressure::new(amplitude[0]),
        Pressure::new(amplitude[1]),
    )));
    model
        .constraints
        .push(AnalysisConstraint::Acoustic(AcousticConstraint {
            constraint_id: "anchor".into(),
            kind: AcousticConstraintKind::PrescribedPressure {
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "slab_left",
                )),
                value: PrescribedValue::Signal(SignalId("drive".into())),
            },
        }));
    model
        .constraints
        .push(AnalysisConstraint::Acoustic(AcousticConstraint {
            constraint_id: "bloch".into(),
            kind: AcousticConstraintKind::PhaseShiftedPeriodic(PhaseShiftedPeriodicConstraint {
                periodic: PeriodicConstraint {
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
                },
                phase: ComplexPhase::new(-std::f64::consts::TAU * 0.2 / 2.0),
            }),
        }));
    (model, mesh)
}

fn solve(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
) -> Result<CanonicalAcousticResponse, AcousticHarmonicError> {
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
}

// witness: FEA-ACOUSTIC-BOUNDARY-001/phase @ acoustic_harmonic/tet4|tet10/numerical_kernel;
//   check: periodic-plane-wave; boundary: Anchored slab plane wave and declared paired planes; no unanchored pole/global
//   conditioning credit.
#[test]
fn anchored_bloch_plane_wave_pressure_mean_velocity_and_signed_intensity_refine() {
    // Anchoring all primary-plane DOFs propagates their phase; interior DOFs still solve.
    // This does not certify elimination of unanchored/free phased classes.
    let mut executed = 0;
    for quadratic in [false, true] {
        for amplitude in [[1.0, 0.4], [-0.4, 1.0]] {
            let cells = if quadratic { [1, 2, 4] } else { [2, 4, 8] };
            let mut pressures = Vec::new();
            let mut velocities = Vec::new();
            for cells in cells {
                let (model, mesh) = fixture(cells, quadratic, amplitude);
                let result = solve(&model, &mesh).unwrap();
                executed += 1;
                let (pressure, velocity, intensity) = reference::errors(&result, amplitude);
                pressures.push(pressure);
                velocities.push(velocity);
                assert!(result.relative_residual < 1e-10);
                assert!(result.mean_impedance_loss_w.canonical_value().abs() < 1e-12);
                assert!(result.mean_input_power_w.canonical_value().abs() < 1e-8);
                if cells == if quadratic { 4 } else { 8 } {
                    assert!(
                        pressure < 0.004,
                        "quadratic={quadratic} pressure={pressures:?}"
                    );
                    assert!(
                        velocity < 0.01,
                        "quadratic={quadratic} velocity={velocities:?}"
                    );
                    assert!(
                        intensity < 0.005,
                        "quadratic={quadratic} intensity RMS={intensity}"
                    );
                }
                reference::check_paired_planes(&result, amplitude);
            }
            eprintln!("Bloch quadratic={quadratic} amplitude={amplitude:?} pressures={pressures:?} meanvelocities={velocities:?}");
            assert!(pressures
                .windows(2)
                .all(|p| p[0] / p[1] > if quadratic { 6.0 } else { 3.0 }));
            assert!(velocities.windows(2).all(|p| p[1] < p[0]));
        }
    }
    assert_eq!(executed, 12);
}

// witness: FEA-ACOUSTIC-BOUNDARY-001/reject @ acoustic_harmonic/tet4|tet10/numerical_kernel;
//   boundary: Periodic pairing, overlap and conflicting pressure rejection only.
#[test]
fn periodic_rejects_unmatched_natural_overlap_and_conflicting_pressure() {
    for quadratic in [false, true] {
        let (model, mesh) = fixture(1, quadratic, [1.0, 0.4]);
        let mut unmatched = model.clone();
        let AnalysisConstraint::Acoustic(constraint) = &mut unmatched.constraints[1] else {
            panic!()
        };
        let AcousticConstraintKind::PhaseShiftedPeriodic(periodic) = &mut constraint.kind else {
            panic!()
        };
        periodic.periodic.transform =
            PeriodicTransform::Translation(Translation::new([0.9, 0.0, 0.0]));
        assert!(solve(&unmatched, &mesh).is_err());
        let mut overlap = model.clone();
        overlap.boundaries.push(AnalysisBoundary {
            boundary_id: "overlap".into(),
            target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                "slab_right",
            )),
            operator: BoundaryOperator::AcousticRadiation,
        });
        assert!(solve(&overlap, &mesh).is_err());
        let mut conflict = model.clone();
        conflict
            .constraints
            .push(AnalysisConstraint::Acoustic(AcousticConstraint {
                constraint_id: "wrong_secondary_pressure".into(),
                kind: AcousticConstraintKind::PrescribedPressure {
                    target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                        "slab_right",
                    )),
                    value: PrescribedValue::Signal(SignalId("drive".into())),
                },
            }));
        assert!(solve(&conflict, &mesh).is_err());
    }
}

#[test]
fn unphased_periodic_anchored_trace_matches_independent_cosine_solution() {
    for quadratic in [false, true] {
        let amplitude = [1.0, 0.4];
        let (mut model, mesh) = fixture(4, quadratic, amplitude);
        let AnalysisConstraint::Acoustic(constraint) = &mut model.constraints[1] else {
            panic!()
        };
        let AcousticConstraintKind::PhaseShiftedPeriodic(phase) = &constraint.kind else {
            panic!()
        };
        constraint.kind = AcousticConstraintKind::Periodic(phase.periodic.clone());
        let result = solve(&model, &mesh).unwrap();
        let k = std::f64::consts::TAU * 0.2 / 2.0;
        let mut error = 0.0;
        let mut norm = 0.0;
        for (node, pressure) in result
            .projection
            .nodes
            .iter()
            .zip(result.pressure_pa.canonical_values())
        {
            let ratio =
                (k * (node.coordinates_m.canonical_value()[0] - 0.5)).cos() / (0.5 * k).cos();
            let reference = amplitude.map(|value| value * ratio);
            error += (pressure.real - reference[0]).powi(2)
                + (pressure.imaginary - reference[1]).powi(2);
            norm += reference[0].powi(2) + reference[1].powi(2);
        }
        // Distributed primary-plane anchor supplies the standing-wave reaction.
        // This is not an unforced periodic eigenmode or free phased-class test.
        assert!((error / norm).sqrt() < 0.004);
        assert!(result.relative_residual < 1e-10);
    }
}
