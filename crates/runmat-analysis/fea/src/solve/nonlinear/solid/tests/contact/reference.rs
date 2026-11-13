use super::*;
#[test]
fn deformable_penalty_and_coulomb_equilibrium_match_closed_affine_body_reference() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        for friction in [false, true] {
            let f = CoupledFixture::new(order, friction);
            let operator = f.operator(friction);
            let basis =
                CanonicalNonlinearSolidBasis::prepare(&f.model, &f.surface.mesh, &f.summary)
                    .unwrap();
            let problem = SolidNewtonProblem::prepare_with_contact(
                &basis,
                &f.summary,
                policy(),
                Some(CoupledFixture::coupling(&operator)),
            )
            .unwrap();
            let mut state = problem.initial_state().unwrap();
            let initial = state.clone();
            let mut old_s = 0.;
            let mut old_q = 0.;
            let mut work = 0.;
            let mut numerical_loss = 0.;
            for (ordinal, factor) in [0.1, 0.5, 1.].into_iter().enumerate() {
                let before = state.clone();
                let increment = problem.advance_next(&state, factor).unwrap();
                assert_eq!(state, before);
                assert_eq!(increment.state.load_factor, factor);
                assert_eq!(increment.cutbacks, 0);
                assert!(increment.relative_residual <= policy().relative_residual_tolerance);
                assert!(increment.relative_increment <= policy().relative_residual_tolerance);
                // Affine epsilon_zz=d/H, gamma_xz=s/H, V=H^3/6,
                // H=.2, E=1200, nu=0: k_z=40N/m, k_x=20N/m.
                // Reference area .02; normal/tangent penalty stiffness=200N/m.
                let normal_load = 0.016 * factor;
                let d = if normal_load <= 40. * 0.0001 {
                    normal_load / 40.
                } else {
                    (normal_load + 200. * 0.0001) / 240.
                };
                let normal = 200. * (d - 0.0001).max(0.);
                let (s, q) = if !friction {
                    (0., 0.)
                } else if normal == 0. {
                    (0.01 * factor / 20., 0.)
                } else {
                    let stick = (0.01 * factor - old_q + 200. * old_s) / 220.;
                    let q = old_q + 200. * (stick - old_s);
                    let cap = 0.3 * normal;
                    if q.abs() <= cap {
                        (stick, q)
                    } else {
                        let q = q.signum() * cap;
                        let s = (0.01 * factor - q) / 20.;
                        work += q.abs() * ((s - old_s) - (q - old_q) / 200.).abs();
                        (s, q)
                    }
                };
                if normal > 0. && friction {
                    numerical_loss += (q - old_q).powi(2) / (2. * 200.);
                }
                let energy = increment.state.contact_energy().unwrap();
                near(energy.frictional_dissipation_j(), work, 1e-12);
                near(energy.algorithmic_dissipation_j(), numerical_loss, 1e-12);
                near(energy.separation_release_j(), 0., 1e-12);
                near(
                    energy.last_increment().normal_elastic_energy_j,
                    normal.powi(2) / (2. * 200.),
                    1e-12,
                );
                near(
                    energy.last_increment().tangential_elastic_energy_j,
                    q.powi(2) / (2. * 200.),
                    1e-12,
                );
                assert_eq!(
                    energy.last_increment().active_points + energy.last_increment().open_points,
                    operator.point_count()
                );
                let master_z = *problem.reduction.free_dof_indices.last().unwrap();
                near(increment.state.displacement[master_z], d, 1e-12);
                if friction {
                    near(
                        increment.state.displacement[problem.reduction.free_dof_indices[0]],
                        s,
                        1e-12,
                    );
                }
                for point in &increment.evaluation.point_evidence {
                    let element = f
                        .surface
                        .projection
                        .elements
                        .iter()
                        .find(|e| e.stable_identity == point.identity.element_identity)
                        .unwrap();
                    let secondary = element
                        .node_indices
                        .iter()
                        .all(|&i| (100..200).contains(&f.surface.projection.nodes[i].node_id));
                    let strain = if secondary {
                        [0., 0., d / 0.2, 0., s / 0.2, 0.]
                    } else {
                        [0.; 6]
                    };
                    let stress = if secondary {
                        [0., 0., 1200. * d / 0.2, 0., 600. * s / 0.2, 0.]
                    } else {
                        [0.; 6]
                    };
                    for i in 0..6 {
                        near(point.total_engineering_strain[i], strain[i], 1e-10);
                        near(point.stress_pa[i], stress[i], 1e-7);
                    }
                }
                let contact = increment.state.contact_configuration().unwrap();
                assert_eq!(contact.ordinal(), ordinal as u64 + 1);
                for point in contact.point_history() {
                    near(point.point().gap_m().canonical_value(), 0.0001 - d, 1e-12);
                }
                let original_work = contact
                    .point_history()
                    .iter()
                    .map(|p| p.point().frictional_work_j_per_m2().canonical_value())
                    .sum::<f64>()
                    / contact.point_history().len() as f64
                    * 0.02;
                near(original_work, work, 1e-12);
                near(operator.frictional_work_j(contact).unwrap(), work, 1e-12);
                problem.validate_accepted(&increment.state).unwrap();
                old_s = s;
                old_q = q;
                state = increment.state;
            }
            assert_eq!(initial.contact_configuration().unwrap().ordinal(), 0);
            let cold: AcceptedSolidState =
                serde_json::from_slice(&serde_json::to_vec(&state).unwrap()).unwrap();
            assert_eq!(cold, state);
            problem.validate_accepted(&cold).unwrap();
        }
    }
}
