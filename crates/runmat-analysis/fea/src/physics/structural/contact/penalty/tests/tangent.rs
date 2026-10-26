use super::*;
// witness: FEA-CONTACT-OPERATOR-001/friction_tangent_history @
//   penalty_coulomb_contact/tet4|tet10/numerical_kernel; coverage: partial;
//   boundary: The local consistent tangent matches two finite-difference step sizes in all smooth
//   regimes of the point law.
#[test]
fn local_consistent_tangent_matches_two_step_sizes_in_all_smooth_regimes() {
    let law = law(0.3);
    let accepted = law
        .update(
            runmat_types::physical::Length::new(-0.01),
            runmat_types::physical::Quantity::new([0.01, -0.005]),
            &law.initial_state(),
        )
        .unwrap()
        .trial_state;
    for (gap, increment, mode) in [
        (0.01, [0.001, -0.002], ContactPointMode::Open),
        (-0.02, [0.001, -0.002], ContactPointMode::Stick),
        (-0.01, [0.08, 0.06], ContactPointMode::Slip),
    ] {
        let response = law
            .update(
                runmat_types::physical::Length::new(gap),
                runmat_types::physical::Quantity::new(increment),
                &accepted,
            )
            .unwrap();
        assert_eq!(response.mode, mode);
        for h in [1.0e-6, 1.0e-7] {
            for column in 0..3 {
                let mut plus = increment;
                let mut minus = increment;
                let mut pg = gap;
                let mut mg = gap;
                if column < 2 {
                    plus[column] += h;
                    minus[column] -= h;
                } else {
                    pg += h;
                    mg -= h;
                }
                let a = law
                    .update(
                        runmat_types::physical::Length::new(pg),
                        runmat_types::physical::Quantity::new(plus),
                        &accepted,
                    )
                    .unwrap();
                let b = law
                    .update(
                        runmat_types::physical::Length::new(mg),
                        runmat_types::physical::Quantity::new(minus),
                        &accepted,
                    )
                    .unwrap();
                for row in 0..3 {
                    let difference = (a.internal_traction_local_pa.canonical_value()[row]
                        - b.internal_traction_local_pa.canonical_value()[row])
                        / (2.0 * h);
                    assert!(
                        (difference - response.tangent_pa_per_m.canonical_value()[row][column])
                            .abs()
                            < 1.0e-6 * 10_000.0,
                        "mode={mode:?} row={row} column={column} h={h}"
                    );
                }
            }
        }
        if mode == ContactPointMode::Slip {
            assert!(response.tangent_pa_per_m.canonical_value()[0][2].abs() > 1.0);
            assert_eq!(response.tangent_pa_per_m.canonical_value()[2][0], 0.0);
        }
    }
}
