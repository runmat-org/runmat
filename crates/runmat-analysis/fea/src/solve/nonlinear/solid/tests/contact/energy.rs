use super::*;

// witness: FEA-NONLINEAR-STATE-001/cold_validation @
//   small_strain_j2_penalty_coulomb_contact/tet4|tet10/prepared_controller;
//   boundary: Accepted contact energy/current point work validation; not proof of past
//   numerical-loss trajectory.
#[test]
fn accepted_energy_rejects_missing_negative_foreign_and_changed_point_work() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let f = CoupledFixture::new(order, true);
        let operator = f.operator(true);
        let basis =
            CanonicalNonlinearSolidBasis::prepare(&f.model, &f.surface.mesh, &f.summary).unwrap();
        let problem = SolidNewtonProblem::prepare_with_contact(
            &basis,
            &f.summary,
            policy(),
            Some(CoupledFixture::coupling(&operator)),
        )
        .unwrap();
        let original = problem.initial_state().unwrap();
        let state = problem.advance_next(&original, 1.).unwrap().state;
        let encoded = serde_json::to_value(&state).unwrap();
        for mutation in 0..7 {
            let mut wire = encoded.clone();
            match mutation {
                0 => wire["contact_energy"] = serde_json::Value::Null,
                1 => wire["contact_energy"]["frictional_dissipation_j"] = (-1.).into(),
                2 => wire["contact_energy"]["frictional_dissipation_j"] = 1.0.into(),
                3 => wire["contact_energy"]["algorithmic_dissipation_j"] = (-1.).into(),
                4 => wire["contact_energy"]["separation_release_j"] = (-1.).into(),
                5 => {
                    wire["contact_energy"]["last_increment"]["normal_elastic_energy_j"] = 1.0.into()
                }
                _ => wire["contact_energy"]["last_increment"]["maximum_penetration_m"] = 1.0.into(),
            }
            let cold: AcceptedSolidState = serde_json::from_value(wire).unwrap();
            assert!(
                problem.validate_accepted(&cold).is_err(),
                "mutation={mutation}"
            );
        }
        let mut initial = original.clone();
        initial.contact_energy = state.contact_energy.clone();
        assert!(problem.validate_accepted(&initial).is_err());
        assert_eq!(
            original
                .contact_energy()
                .unwrap()
                .frictional_dissipation_j(),
            0.
        );
    }
}

// witness: FEA-CONTACT-OPERATOR-001/normal_friction_equilibrium @
//   penalty_coulomb_contact/tet4|tet10/numerical_kernel; coverage: partial;
//   boundary: Opening releases stored tangential energy without inventing friction heat in a
//   coupled solid solve for both orders.
// witness: FEA-NONLINEAR-GOVERNANCE-001/operator_constitutive_contact @
//   nonlinear_structural/tet4|tet10/numerical_kernel; coverage: partial;
//   boundary: A coupled solid contact solve releases stored tangential energy on opening for both
//   orders; constitutive plasticity is covered in the operator requirement.
#[test]
fn opening_releases_stored_tangential_energy_without_inventing_friction_heat() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let f = CoupledFixture::new(order, true);
        let operator = f.operator(true);
        let basis =
            CanonicalNonlinearSolidBasis::prepare(&f.model, &f.surface.mesh, &f.summary).unwrap();
        let problem = SolidNewtonProblem::prepare_with_contact(
            &basis,
            &f.summary,
            policy(),
            Some(CoupledFixture::coupling(&operator)),
        )
        .unwrap();
        let initial = problem.initial_state().unwrap();
        let state = problem.advance_next(&initial, 1.).unwrap().state;
        let before = serde_json::to_vec(&state).unwrap();
        // Frozen unloading configuration is a kinematic point-history reference,
        // not a newly solved decreasing-load trajectory or driver qualification.
        let positions = basis
            .contact_positions(&vec![0.; basis.dof_count()])
            .unwrap();
        let accepted = state.contact_configuration().unwrap();
        let trial = operator
            .assemble_trial(accepted, accepted.ordinal(), &positions)
            .unwrap();
        let previous = state.contact_energy().unwrap();
        let next = previous.advance(trial.diagnostics).unwrap();
        near(
            next.frictional_dissipation_j(),
            previous.frictional_dissipation_j(),
            1e-12,
        );
        near(
            next.separation_release_j(),
            previous.last_increment().tangential_elastic_energy_j,
            1e-12,
        );
        near(
            next.algorithmic_dissipation_j(),
            previous.algorithmic_dissipation_j(),
            1e-12,
        );
        near(next.last_increment().tangential_elastic_energy_j, 0., 1e-12);
        assert_eq!(next.last_increment().open_points, operator.point_count());
        assert_eq!(before, serde_json::to_vec(&state).unwrap());
        assert_eq!(next, previous.advance(trial.diagnostics).unwrap());
    }
}
