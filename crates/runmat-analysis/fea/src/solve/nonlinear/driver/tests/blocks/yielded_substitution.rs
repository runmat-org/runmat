//! Rehashing checks byte integrity, not original provenance or physical history.
use super::*;

// witness: FEA-NONLINEAR-STATE-001/cold_validation @
//   small_strain_j2/tet4|tet10/prepared_controller;
//   boundary: Rehashed yielded history substitution requires physical validation, beyond content
//   hashes.
#[test]
fn rehashed_cold_yielded_history_requires_physical_validation() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (model, summary, mut options) = uniaxial(order);
        options.increment_count = 2;
        options.max_accepted_increments = 3;
        let schedule = LoadIncrementSchedule::canonical(&[], &[0.95], 0.0, 1.0).unwrap();
        let first = block(
            &model,
            &summary,
            &options,
            &schedule,
            NonlinearLoadBlockStart::Fresh,
            2,
        )
        .unwrap();
        let original = first.continuation;
        assert!(!original.complete());
        assert!(original
            .accepted_state
            .constitutive
            .points
            .iter()
            .all(|p| p.equivalent_plastic_strain > 0.0));
        let original_bytes = original.canonical_bytes().unwrap();
        let basis = crate::assembly::nonlinear_solid::CanonicalNonlinearSolidBasis::prepare(
            &model,
            options.solver_mesh.as_ref().unwrap(),
            &summary,
        )
        .unwrap();
        let problem = crate::solve::nonlinear::solid::SolidNewtonProblem::prepare(
            &basis,
            &summary,
            options.newton_policy(),
        )
        .unwrap();
        problem
            .validate_accepted(original.accepted_state())
            .unwrap();
        for substitution in 0..5 {
            let mut changed = original.clone();
            match substitution {
                0 => {
                    changed.accepted_state.constitutive.points[0].engineering_plastic_strain[3] +=
                        0.01
                }
                1 => {
                    changed.accepted_state.constitutive.points[0].equivalent_plastic_strain += 0.01
                }
                2 => {
                    for point in &mut changed.accepted_state.constitutive.points {
                        point.engineering_plastic_strain = [0.0; 6];
                        point.equivalent_plastic_strain = 0.0;
                    }
                    changed.accepted_state.accumulated_plastic_dissipation_j = 0.0;
                }
                3 => changed
                    .accepted_state
                    .constitutive
                    .point_identities
                    .swap(0, 1),
                _ => changed.accepted_state.accumulated_plastic_dissipation_j *= 0.5,
            }
            changed.content_digest = changed.derive_digest().unwrap();
            let bytes = changed.canonical_bytes().unwrap();
            // Explicitly pass integrity transport: rejection must happen downstream.
            let cold =
                NonlinearLoadContinuation::from_canonical_bytes(&bytes, changed.content_digest())
                    .unwrap();
            assert_eq!(cold, changed);
            // Isolate the physical owner so controller work exhaustion cannot
            // accidentally earn rejection credit for a substituted history.
            assert!(problem.validate_accepted(cold.accepted_state()).is_err());
            assert!(
                block(
                    &model,
                    &summary,
                    &options,
                    &schedule,
                    NonlinearLoadBlockStart::Continue {
                        continuation: &cold,
                        expected_digest: cold.content_digest(),
                    },
                    1,
                )
                .is_err(),
                "order={order:?}, substitution={substitution}"
            );
            assert_eq!(original.canonical_bytes().unwrap(), original_bytes);
        }
        let cold = NonlinearLoadContinuation::from_canonical_bytes(
            &original_bytes,
            original.content_digest(),
        )
        .unwrap();
        let resumed = block(
            &model,
            &summary,
            &options,
            &schedule,
            NonlinearLoadBlockStart::Continue {
                continuation: &cold,
                expected_digest: cold.content_digest(),
            },
            1,
        )
        .unwrap();
        assert!(resumed.complete);
        assert!((resumed.result.equivalent_plastic_strain_snapshots[0][0] - 0.02).abs() < 2e-9);
    }
}
