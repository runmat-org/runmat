//! Cold portable periodic pressure retains the authored phase and full physical field.
use super::*;
use runmat_analysis_core::{
    ComplexPhase, PairedConstraint, PairingPolicy, PeriodicConstraint, PeriodicTransform,
    PhaseShiftedPeriodicConstraint, Translation,
};
use runmat_meshing_core::CanonicalMeshingContract;

#[test]
fn cold_bloch_workload_and_result_preserve_full_pressure_and_phase_lineage() {
    for quadratic in [false, true] {
        let mesh = if quadratic {
            quadratic_slab_mesh(2)
        } else {
            slab_mesh(4)
        };
        let mut authored = workload();
        authored.model.boundaries.clear();
        let phase = -std::f64::consts::TAU * authored.frequency.canonical_value() / 2.0;
        authored
            .model
            .constraints
            .push(AnalysisConstraint::Acoustic(AcousticConstraint {
                constraint_id: "bloch".into(),
                kind: AcousticConstraintKind::PhaseShiftedPeriodic(
                    PhaseShiftedPeriodicConstraint {
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
                            transform: PeriodicTransform::Translation(Translation::new([
                                1.0, 0.0, 0.0,
                            ])),
                        },
                        phase: ComplexPhase::new(phase),
                    },
                ),
            }));
        let cold =
            AcousticHarmonicWorkload::from_canonical_bytes(&authored.canonical_bytes().unwrap())
                .unwrap();
        let cold_mesh =
            SolverMeshArtifact::canonical_decode(&mesh.canonical_encode().unwrap()).unwrap();
        let result = cold.execute(&cold_mesh).unwrap();
        let imported =
            AcousticHarmonicSolverResult::from_canonical_bytes(&result.canonical_bytes().unwrap())
                .unwrap();
        imported.validate_for(&authored, &mesh).unwrap();
        assert_eq!(result, imported);
        for (node, pressure) in mesh.topology.nodes.iter().zip(imported.pressure_pa()) {
            let argument = phase * node.coordinates_m.canonical_value()[0];
            assert!(
                (pressure.real - argument.cos()).hypot(pressure.imaginary - argument.sin()) < 0.002
            );
        }
        let profiles = imported
            .state_publication_profiles(Digest::sha256(result.canonical_bytes().unwrap()))
            .unwrap();
        profiles.canonical_bytes().unwrap();
        let mut wrong_phase = authored.clone();
        let AnalysisConstraint::Acoustic(constraint) = &mut wrong_phase.model.constraints[1] else {
            panic!()
        };
        let AcousticConstraintKind::PhaseShiftedPeriodic(bloch) = &mut constraint.kind else {
            panic!()
        };
        bloch.phase = runmat_types::physical::Quantity::new(bloch.phase.canonical_value() + (0.1));
        assert_ne!(
            authored.identity().unwrap(),
            wrong_phase.identity().unwrap()
        );
        assert!(imported.validate_for(&wrong_phase, &mesh).is_err());
    }
}
