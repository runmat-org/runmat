use super::*;
use runmat_analysis_core::FrictionCoefficient;
fn law(mu: f64) -> PenaltyContactLaw {
    PenaltyContactLaw::derive(
        &ContactInterfaceModel {
            friction: ContactFriction::Coulomb {
                coefficient: FrictionCoefficient::new(mu),
            },
        },
        ContactPenaltyPolicy {
            normal_stiffness_pa_per_m: runmat_types::physical::TractionDisplacementStiffness::new(
                10_000.0,
            ),
            tangential_stiffness_pa_per_m: Some(
                runmat_types::physical::TractionDisplacementStiffness::new(1_000.0),
            ),
        },
    )
    .unwrap()
}
fn near(a: f64, b: f64) {
    assert!(
        (a - b).abs() <= 1.0e-12 * a.abs().max(b.abs()).max(1.0),
        "{a} != {b}"
    );
}
// witness: FEA-CONTACT-OPERATOR-001/references @
//   penalty_coulomb_contact/tet4|tet10/numerical_kernel; coverage: partial;
//   boundary: Analytic sliding, unloading, reversal and separation keep the original history at the
//   point law.
#[test]
fn analytic_sliding_unloading_reversal_and_separation_keep_original_history() {
    let law = law(0.3);
    let mut accepted = law.initial_state();
    for (increment, q, net_slip, distance, work, mode) in [
        (0.01, 10.0, 0.0, 0.0, 0.0, ContactPointMode::Stick),
        (0.05, 30.0, 0.03, 0.03, 0.9, ContactPointMode::Slip),
        (-0.01, 20.0, 0.03, 0.03, 0.9, ContactPointMode::Stick),
        (-0.08, -30.0, 0.0, 0.06, 1.8, ContactPointMode::Slip),
    ] {
        let before = accepted.clone();
        let response = law
            .update(
                runmat_types::physical::Length::new(-0.01),
                runmat_types::physical::Quantity::new([increment, 0.0]),
                &accepted,
            )
            .unwrap();
        assert_eq!(accepted, before);
        assert_eq!(
            response,
            law.update(
                runmat_types::physical::Length::new(-0.01),
                runmat_types::physical::Quantity::new([increment, 0.0]),
                &accepted
            )
            .unwrap()
        );
        assert_eq!(response.mode, mode);
        near(response.normal_pressure_pa.canonical_value(), 100.0);
        near(response.internal_traction_local_pa.canonical_value()[0], q);
        near(
            response.trial_state.inelastic_slip_m().canonical_value()[0],
            net_slip,
        );
        near(
            response
                .trial_state
                .accumulated_sliding_distance_m()
                .canonical_value(),
            distance,
        );
        near(
            response
                .trial_state
                .frictional_work_j_per_m2()
                .canonical_value(),
            work,
        );
        let old_q = accepted.tangential_traction_pa().canonical_value()[0];
        let delta_elastic_energy = 0.5 * (q * q - old_q * old_q) / 1_000.0;
        near(
            q * increment,
            delta_elastic_energy
                + response
                    .frictional_dissipation_increment_j_per_m2
                    .canonical_value()
                + response
                    .tangential_algorithmic_dissipation_increment_j_per_m2
                    .canonical_value(),
        );
        accepted = response.trial_state;
    }
    let response = law
        .update(
            runmat_types::physical::Length::new(0.02),
            runmat_types::physical::Quantity::new([100.0, -100.0]),
            &accepted,
        )
        .unwrap();
    assert_eq!(response.mode, ContactPointMode::Open);
    assert_eq!(
        response.internal_traction_local_pa.canonical_value(),
        [0.0; 3]
    );
    near(
        response
            .separation_release_energy_j_per_m2
            .canonical_value(),
        0.45,
    );
    near(
        response
            .frictional_dissipation_increment_j_per_m2
            .canonical_value(),
        0.0,
    );
    near(
        response
            .trial_state
            .frictional_work_j_per_m2()
            .canonical_value(),
        1.8,
    );
    near(
        response
            .trial_state
            .accumulated_sliding_distance_m()
            .canonical_value(),
        0.06,
    );
    let reclosed = law
        .update(
            runmat_types::physical::Length::new(-0.01),
            runmat_types::physical::Quantity::new([0.001, 0.0]),
            &response.trial_state,
        )
        .unwrap();
    near(
        reclosed.internal_traction_local_pa.canonical_value()[0],
        1.0,
    );
}
// witness: FEA-CONTACT-OPERATOR-001/normal_friction_equilibrium @
//   penalty_coulomb_contact/tet4|tet10/numerical_kernel; coverage: partial;
//   boundary: At the point law, pressure unloading changes the friction radius and retains energy
//   accounting; no assembled body.
#[test]
fn pressure_unloading_changes_the_friction_radius_and_retains_energy_accounting() {
    let law = law(0.3);
    let accepted = law
        .update(
            runmat_types::physical::Length::new(-0.01),
            runmat_types::physical::Quantity::new([0.1, 0.0]),
            &law.initial_state(),
        )
        .unwrap()
        .trial_state;
    let response = law
        .update(
            runmat_types::physical::Length::new(-0.005),
            runmat_types::physical::Quantity::new([0.0; 2]),
            &accepted,
        )
        .unwrap();
    assert_eq!(response.mode, ContactPointMode::Slip);
    near(
        response.internal_traction_local_pa.canonical_value()[0],
        15.0,
    );
    near(
        response
            .frictional_dissipation_increment_j_per_m2
            .canonical_value(),
        0.225,
    );
    near(
        response
            .tangential_algorithmic_dissipation_increment_j_per_m2
            .canonical_value(),
        0.1125,
    );
    near(
        response
            .tangential_elastic_energy_j_per_m2
            .canonical_value(),
        0.1125,
    );
}
// witness: FEA-CONTACT-OPERATOR-001/normal_friction_equilibrium @
//   penalty_coulomb_contact/tet4|tet10/numerical_kernel; coverage: partial;
//   boundary: Frictionless and zero-coefficient pairs have no tangential resistance at the point
//   law.
#[test]
fn frictionless_and_zero_coefficient_have_no_tangential_resistance() {
    let frictionless = PenaltyContactLaw::derive(
        &ContactInterfaceModel {
            friction: ContactFriction::Frictionless,
        },
        ContactPenaltyPolicy {
            normal_stiffness_pa_per_m: runmat_types::physical::TractionDisplacementStiffness::new(
                10_000.0,
            ),
            tangential_stiffness_pa_per_m: None,
        },
    )
    .unwrap();
    for law in [frictionless, law(0.0)] {
        for increment in [[0.0; 2], [1.0, -2.0]] {
            let response = law
                .update(
                    runmat_types::physical::Length::new(-0.01),
                    runmat_types::physical::Quantity::new(increment),
                    &law.initial_state(),
                )
                .unwrap();
            assert_eq!(
                &response.internal_traction_local_pa.canonical_value()[..2],
                &[0.0; 2]
            );
            for row in &response.tangent_pa_per_m.canonical_value()[..2] {
                assert_eq!(*row, [0.0; 3]);
            }
            near(
                response.normal_elastic_energy_j_per_m2.canonical_value(),
                0.5,
            );
        }
    }
}
mod objectivity;

#[test]
fn changing_tangential_direction_returns_to_a_circular_coulomb_limit() {
    let law = law(0.3);
    let accepted = law
        .update(
            runmat_types::physical::Length::new(-0.01),
            runmat_types::physical::Quantity::new([0.01, 0.0]),
            &law.initial_state(),
        )
        .unwrap()
        .trial_state;
    let response = law
        .update(
            runmat_types::physical::Length::new(-0.01),
            runmat_types::physical::Quantity::new([0.0, 0.04]),
            &accepted,
        )
        .unwrap();
    let trial_length = 1700.0_f64.sqrt();
    let expected = [300.0 / trial_length, 1200.0 / trial_length];
    for (actual, expected) in response
        .internal_traction_local_pa
        .canonical_value()
        .iter()
        .zip(expected)
    {
        near(*actual, expected);
    }
    near(
        response
            .trial_state
            .tangential_traction_pa()
            .canonical_value()[0]
            .hypot(
                response
                    .trial_state
                    .tangential_traction_pa()
                    .canonical_value()[1],
            ),
        30.0,
    );
    near(
        response
            .trial_state
            .accumulated_sliding_distance_m()
            .canonical_value(),
        (trial_length - 30.0) / 1000.0,
    );
    near(
        response
            .frictional_dissipation_increment_j_per_m2
            .canonical_value(),
        30.0 * (trial_length - 30.0) / 1000.0,
    );
    near(
        expected[1] * 0.04,
        response
            .tangential_elastic_energy_j_per_m2
            .canonical_value()
            - 0.05
            + response
                .frictional_dissipation_increment_j_per_m2
                .canonical_value()
            + response
                .tangential_algorithmic_dissipation_increment_j_per_m2
                .canonical_value(),
    );
}

mod tangent;
mod validation;

#[test]
fn typed_penalty_policy_and_point_history_keep_canonical_numeric_wire() {
    let policy = ContactPenaltyPolicy {
        normal_stiffness_pa_per_m: runmat_types::physical::TractionDisplacementStiffness::new(
            10_000.0,
        ),
        tangential_stiffness_pa_per_m: Some(
            runmat_types::physical::TractionDisplacementStiffness::new(1_000.0),
        ),
    };
    let wire = serde_json::to_value(policy).unwrap();
    assert_eq!(
        wire,
        serde_json::json!({
            "normal_stiffness_pa_per_m": 10_000.0,
            "tangential_stiffness_pa_per_m": 1_000.0
        })
    );
    assert_eq!(
        serde_json::from_value::<ContactPenaltyPolicy>(wire).unwrap(),
        policy
    );
    let frictionless: ContactPenaltyPolicy =
        serde_json::from_value(serde_json::json!({ "normal_stiffness_pa_per_m": 10_000.0 }))
            .unwrap();
    assert_eq!(frictionless.tangential_stiffness_pa_per_m, None);
    let law = law(0.3);
    let state = law
        .update(
            runmat_types::physical::Length::new(-0.01),
            runmat_types::physical::Quantity::new([0.5, 0.0]),
            &law.initial_state(),
        )
        .unwrap()
        .trial_state;
    let encoded = serde_json::to_value(&state).unwrap();
    assert_eq!(encoded["gap_m"], serde_json::json!(-0.01));
    assert!(encoded["tangential_traction_pa"].is_array());
    assert!(encoded["accumulated_sliding_distance_m"].is_number());
    assert_eq!(
        serde_json::from_value::<ContactPointState>(encoded).unwrap(),
        state
    );
}
