use super::*;
// witness: FEA-CONTACT-OPERATOR-001/references @
//   penalty_coulomb_contact/tet4|tet10/numerical_kernel; coverage: partial;
//   boundary: Two disjoint canonical bodies under prescribed compression match the closed penalty
//   force, energy and tangent for both orders; Hertz-type curved contact is not covered.
#[test]
fn disjoint_canonical_bodies_under_prescribed_compression_match_closed_penalty_force_energy_and_tangent(
) {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let f = Fixture::new(order, false);
        let operator = f.operator(false);
        let initial = operator.initial_configuration(&f.positions).unwrap();
        assert_eq!(initial.ordinal(), 0);
        assert_eq!(initial.point_history().len(), 7);
        assert!(initial
            .point_history()
            .iter()
            .all(|p| p.point().gap_m().canonical_value() > 0.));
        let current = f.moved(0.06, 0.);
        let trial = operator.assemble_trial(&initial, 0, &current).unwrap();
        assert_eq!(initial.ordinal(), 0);
        assert_eq!(trial.configuration.ordinal(), 1);
        operator
            .validate_configuration(&trial.configuration, 1)
            .unwrap();
        let force = f.secondary_force(&trial.residual);
        near(force[0], 0., 1e-12);
        near(force[1], 0., 1e-12);
        near(force[2], 2., 1e-10);
        near(trial.diagnostics.normal_elastic_energy_j, 0.01, 1e-12);
        near(trial.diagnostics.maximum_penetration_m, 0.01, 1e-12);
        assert_eq!(trial.diagnostics.active_points, 7);
        assert_eq!(trial.diagnostics.open_points, 0);
        for k in 0..3 {
            near(f.rows.iter().map(|r| trial.residual[r[k]]).sum(), 0., 1e-10);
        }
        let velocity: Vec<_> = current
            .iter()
            .map(|p| [0.003 * p[1], -0.002 * p[0], 0.004 * p[2] + 0.001])
            .collect();
        let flat: Vec<_> = velocity.iter().flatten().copied().collect();
        let action = crate::algebra::spmv(&trial.tangent, &flat).unwrap();
        for step in [1e-5, 1e-6] {
            let eval = |sign: f64| {
                let positions: Vec<_> = current
                    .iter()
                    .zip(&velocity)
                    .map(|(p, v)| std::array::from_fn(|k| p[k] + sign * step * v[k]))
                    .collect();
                operator
                    .assemble_trial(&initial, 0, &positions)
                    .unwrap()
                    .residual
            };
            let plus = eval(1.);
            let minus = eval(-1.);
            for ((actual, a), b) in action.iter().zip(plus).zip(minus) {
                near(*actual, (a - b) / (2. * step), 1e-6);
            }
        }
        assert_eq!(
            initial,
            operator.initial_configuration(&f.positions).unwrap()
        );
    }
}
#[test]
fn original_global_coulomb_history_retains_stick_slip_work_and_no_rigid_motion_defect() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let f = Fixture::new(order, true);
        let operator = f.operator(true);
        let initial = operator.initial_configuration(&f.positions).unwrap();
        let stick = operator
            .assemble_trial(&initial, 0, &f.moved(0.06, 0.001))
            .unwrap();
        near(f.secondary_force(&stick.residual)[0], 0.2, 1e-10);
        assert_eq!(stick.diagnostics.stick_points, 7);
        near(stick.diagnostics.tangential_elastic_energy_j, 0.0001, 1e-12);
        let slip = operator
            .assemble_trial(&stick.configuration, 1, &f.moved(0.06, 0.02))
            .unwrap();
        for (committed, ordinal, current) in [
            (&initial, 0, f.moved(0.06, 0.001)),
            (&stick.configuration, 1, f.moved(0.06, 0.02)),
        ] {
            let trial = operator
                .assemble_trial(committed, ordinal, &current)
                .unwrap();
            let velocity: Vec<_> = current
                .iter()
                .map(|p| [0.003 * p[1], -0.002 * p[0], 0.004 * p[2] + 0.001])
                .collect();
            let flat: Vec<_> = velocity.iter().flatten().copied().collect();
            let action = crate::algebra::spmv(&trial.tangent, &flat).unwrap();
            for step in [1e-5, 1e-6] {
                let eval = |sign: f64| {
                    let positions: Vec<_> = current
                        .iter()
                        .zip(&velocity)
                        .map(|(p, v)| std::array::from_fn(|k| p[k] + sign * step * v[k]))
                        .collect();
                    operator
                        .assemble_trial(committed, ordinal, &positions)
                        .unwrap()
                        .residual
                };
                let plus = eval(1.);
                let minus = eval(-1.);
                for ((actual, a), b) in action.iter().zip(plus).zip(minus) {
                    near(*actual, (a - b) / (2. * step), 1e-6);
                }
            }
            let mut moment = [0.; 3];
            let mut force = [0.; 3];
            for (p, rows) in current.iter().zip(&f.rows) {
                let r = rows.map(|row| trial.residual[row]);
                for k in 0..3 {
                    force[k] += r[k];
                }
                moment[0] += p[1] * r[2] - p[2] * r[1];
                moment[1] += p[2] * r[0] - p[0] * r[2];
                moment[2] += p[0] * r[1] - p[1] * r[0];
            }
            for value in force.into_iter().chain(moment) {
                near(value, 0., 1e-10);
            }
        }

        assert_eq!(slip.diagnostics.slip_points, 7);
        near(f.secondary_force(&slip.residual)[0], 0.6, 1e-10);
        near(
            slip.diagnostics.frictional_dissipation_increment_j,
            0.0102,
            1e-10,
        );
        assert!(slip.configuration.point_history().iter().all(|p| p
            .point()
            .accumulated_sliding_distance_m()
            .canonical_value()
            > 0.));
        let rigid: Vec<_> = slip
            .configuration
            .positions_m()
            .iter()
            .map(|p| [2. - p[1], 3. + p[0], 4. + p[2]])
            .collect();
        let rotated = operator
            .assemble_trial(&slip.configuration, 2, &rigid)
            .unwrap();
        near(
            rotated.diagnostics.frictional_dissipation_increment_j,
            0.,
            1e-12,
        );
        for (a, b) in slip
            .configuration
            .point_history()
            .iter()
            .zip(rotated.configuration.point_history())
        {
            near(
                a.point().accumulated_sliding_distance_m().canonical_value(),
                b.point().accumulated_sliding_distance_m().canonical_value(),
                1e-12,
            );
            near(
                a.point().frictional_work_j_per_m2().canonical_value(),
                b.point().frictional_work_j_per_m2().canonical_value(),
                1e-12,
            );
        }
    }
}
