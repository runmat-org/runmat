use super::*;
#[test]
fn cold_local_history_revalidates_and_rejects_law_substitution_and_tampering() {
    let law = law(0.3);
    let accepted = law
        .update(
            runmat_types::physical::Length::new(-0.01),
            runmat_types::physical::Quantity::new([0.1, 0.03]),
            &law.initial_state(),
        )
        .unwrap()
        .trial_state;
    let wire = serde_json::to_value(&accepted).unwrap();
    let cold: ContactPointState = serde_json::from_value(wire.clone()).unwrap();
    law.validate_state(&cold).unwrap();
    assert_eq!(
        law.update(
            runmat_types::physical::Length::new(-0.01),
            runmat_types::physical::Quantity::new([0.001, 0.0]),
            &accepted
        ),
        law.update(
            runmat_types::physical::Length::new(-0.01),
            runmat_types::physical::Quantity::new([0.001, 0.0]),
            &cold
        )
    );
    assert_eq!(
        super::law(0.4).update(
            runmat_types::physical::Length::new(-0.01),
            runmat_types::physical::Quantity::new([0.0; 2]),
            &cold
        ),
        Err(ContactPointError::InvalidHistory)
    );
    for (key, value) in [
        ("gap_m", serde_json::json!(0.1)),
        ("tangential_traction_pa", serde_json::json!([1000.0, 0.0])),
        ("accumulated_sliding_distance_m", serde_json::json!(-1.0)),
        ("frictional_work_j_per_m2", serde_json::json!(-1.0)),
    ] {
        let mut bad = wire.clone();
        bad[key] = value;
        let bad: ContactPointState = serde_json::from_value(bad).unwrap();
        assert_eq!(
            law.update(
                runmat_types::physical::Length::new(-0.01),
                runmat_types::physical::Quantity::new([0.0; 2]),
                &bad
            ),
            Err(ContactPointError::InvalidHistory)
        );
    }
}
#[test]
fn nonfinite_inputs_overflow_and_cancellation_cannot_publish_trial_history() {
    let law = law(0.3);
    let accepted = law.initial_state();
    assert_eq!(
        law.update(
            runmat_types::physical::Length::new(f64::NAN),
            runmat_types::physical::Quantity::new([0.0; 2]),
            &accepted
        ),
        Err(ContactPointError::InvalidInput)
    );
    assert_eq!(
        law.update(
            runmat_types::physical::Length::new(-f64::MAX),
            runmat_types::physical::Quantity::new([0.0; 2]),
            &accepted
        ),
        Err(ContactPointError::NonFiniteResponse)
    );
    assert_eq!(
        law.update(
            runmat_types::physical::Length::new(-0.01),
            runmat_types::physical::Quantity::new([f64::MAX, 0.0]),
            &accepted
        ),
        Err(ContactPointError::NonFiniteResponse)
    );
    let token = crate::progress::FeaCancellationToken::new();
    token.cancel();
    let _guard =
        crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
            cancellation: token,
            ..Default::default()
        });
    let (result, retained, peak) = crate::topology::measure_test_allocations(|| {
        law.update(
            runmat_types::physical::Length::new(-0.01),
            runmat_types::physical::Quantity::new([0.001, 0.0]),
            &accepted,
        )
    });
    assert_eq!(result, Err(ContactPointError::Cancelled));
    assert_eq!((retained, peak), (0, 0));
}

#[test]
fn impossible_zero_friction_work_and_invalid_controls_reject() {
    let zero = super::law(0.0);
    let mut state = serde_json::to_value(zero.initial_state()).unwrap();
    state["frictional_work_j_per_m2"] = serde_json::json!(1.0);
    let state: ContactPointState = serde_json::from_value(state).unwrap();
    assert_eq!(
        zero.validate_state(&state),
        Err(ContactPointError::InvalidHistory)
    );
    let frictionless = ContactInterfaceModel {
        friction: ContactFriction::Frictionless,
    };
    for normal in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            PenaltyContactLaw::derive(
                &frictionless,
                ContactPenaltyPolicy {
                    normal_stiffness_pa_per_m:
                        runmat_types::physical::TractionDisplacementStiffness::new(normal),
                    tangential_stiffness_pa_per_m: None
                }
            ),
            Err(ContactPointError::InvalidPolicy)
        ));
    }
    assert!(matches!(
        PenaltyContactLaw::derive(
            &frictionless,
            ContactPenaltyPolicy {
                normal_stiffness_pa_per_m:
                    runmat_types::physical::TractionDisplacementStiffness::new(1.0),
                tangential_stiffness_pa_per_m: Some(
                    runmat_types::physical::TractionDisplacementStiffness::new(1.0)
                )
            }
        ),
        Err(ContactPointError::InvalidPolicy)
    ));
}
