use super::*;
use crate::builtins::deep_learning::arguments::DeepLearningObjectEdit as _;

fn options(solver: &str, extra: &[(&str, Value)]) -> Value {
    let identity = Solver::from_name(solver)
        .expect("test solver is supported")
        .class_identity();
    let mut object = ObjectInstance::new(identity);
    object
        .replace_dl_property("SolverName", Value::String(solver.into()))
        .expect("fixture property");
    for (name, value) in extra {
        object
            .replace_dl_property(name, value.clone())
            .expect("fixture property");
    }
    Value::Object(object)
}

#[test]
fn omitted_solver_properties_use_canonical_defaults_and_preserve_overrides() {
    for (name, rate, decay) in [
        ("sgdm", 0.01, 0.999),
        ("rmsprop", 0.001, 0.9),
        ("adam", 0.001, 0.999),
    ] {
        let config = TrainingConfig::from_value(&options(name, &[]), "trainnet").unwrap();
        assert_eq!(config.learn_rate, rate);
        assert_eq!(config.squared_gradient_decay_factor, decay);
    }
    for name in ["rmsprop", "adam"] {
        let config = TrainingConfig::from_value(
            &options(
                name,
                &[
                    ("InitialLearnRate", Value::Num(0.025)),
                    ("SquaredGradientDecayFactor", Value::Num(0.8)),
                ],
            ),
            "trainnet",
        )
        .unwrap();
        assert_eq!(config.learn_rate, 0.025);
        assert_eq!(config.squared_gradient_decay_factor, 0.8);
    }
}

#[test]
fn constructor_and_training_consumer_share_solver_defaults() {
    for (name, solver, rate, decay) in [
        ("sgdm", Solver::Sgdm, 0.01, None),
        ("rmsprop", Solver::Rmsprop, 0.001, Some(0.9)),
        ("adam", Solver::Adam, 0.001, Some(0.999)),
    ] {
        let value = futures::executor::block_on(
            crate::builtins::deep_learning::training::training_options_builtin(
                Value::String(name.into()),
                crate::builtins::common::test_support::builtin_arguments(vec![]),
            ),
        )
        .unwrap();
        let Value::Object(object) = &value else {
            panic!("trainingOptions must return an object")
        };
        assert!(object.class_name().is(solver.class_identity()));
        assert_eq!(
            object.properties().get("InitialLearnRate"),
            Some(&Value::Num(rate))
        );
        assert_eq!(
            object
                .properties()
                .get("SquaredGradientDecayFactor")
                .cloned(),
            decay.map(Value::Num)
        );
        let config = TrainingConfig::from_value(&value, "trainnet").unwrap();
        assert_eq!(config.solver, solver);
        assert_eq!(config.learn_rate, rate);
        assert_eq!(config.squared_gradient_decay_factor, decay.unwrap_or(0.999));
    }
}

#[test]
fn solver_class_and_solver_name_must_agree_without_generic_class_fallback() {
    let mut mismatch = options("adam", &[]);
    let Value::Object(object) = &mut mismatch else {
        unreachable!()
    };
    object
        .replace_dl_property("SolverName", Value::String("sgdm".into()))
        .expect("fixture property");
    assert!(TrainingConfig::from_value(&mismatch, "trainnet")
        .err()
        .unwrap()
        .message()
        .contains("disagrees"));

    let mut missing_name = options("rmsprop", &[]);
    let Value::Object(object) = &mut missing_name else {
        unreachable!()
    };
    object
        .remove_dl_property("SolverName")
        .expect("fixture removal");
    assert_eq!(
        TrainingConfig::from_value(&missing_name, "trainnet")
            .unwrap()
            .solver,
        Solver::Rmsprop
    );

    let mut legacy = options("adam", &[]);
    let Value::Object(object) = &mut legacy else {
        unreachable!()
    };
    let properties = object
        .properties()
        .iter()
        .map(|(name, value)| (name, value.clone()))
        .collect::<Vec<_>>();
    let Value::Object(reclassified) =
        crate::builtins::deep_learning::object("nnet.cnn.TrainingOptions", properties)
            .expect("fixture class replacement")
    else {
        unreachable!()
    };
    *object = reclassified;
    assert!(TrainingConfig::from_value(&legacy, "trainnet").is_err());
}

#[test]
fn positional_solver_cannot_be_overridden_by_a_named_property() {
    let error = futures::executor::block_on(
        crate::builtins::deep_learning::training::training_options_builtin(
            Value::String("sgdm".into()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("SolverName".into()),
                Value::String("adam".into()),
            ]),
        ),
    )
    .err()
    .unwrap();
    assert!(error.message().contains("SolverName"), "{error}");
}

#[test]
fn training_controls_have_one_typed_admission_authority() {
    let valid = options(
        "adam",
        &[
            ("L2Regularization", Value::Num(0.0)),
            ("Plots", Value::String("none".into())),
        ],
    );
    assert!(TrainingConfig::from_value(&valid, "trainnet").is_ok());
    for extra in [
        ("ValidationData", Value::Num(1.0)),
        ("LearnRateSchedule", Value::String("piecewise".into())),
        ("GradientThreshold", Value::Num(1.0)),
        ("Plots", Value::String("training-progress".into())),
        ("Mystery", Value::Num(1.0)),
    ] {
        let value = options("adam", std::slice::from_ref(&extra));
        let error = TrainingConfig::from_value(&value, "trainnet")
            .err()
            .unwrap();
        assert!(error.message().contains(extra.0), "{error}");
    }
    let nonzero_l2 = options("sgdm", &[("L2Regularization", Value::Num(0.01))]);
    assert_eq!(
        TrainingConfig::from_value(&nonzero_l2, "trainnet")
            .unwrap()
            .l2_regularization,
        0.01
    );
}

#[test]
fn solver_specific_and_duplicate_options_fail_before_training() {
    let adam = options("adam", &[("Momentum", Value::Num(0.5))]);
    assert!(TrainingConfig::from_value(&adam, "trainnet")
        .err()
        .unwrap()
        .message()
        .contains("Momentum"));
    let sgdm = options("sgdm", &[("GradientDecayFactor", Value::Num(0.8))]);
    assert!(TrainingConfig::from_value(&sgdm, "trainNetwork")
        .err()
        .unwrap()
        .message()
        .contains("GradientDecayFactor"));
    let rmsprop = options(
        "rmsprop",
        &[("SquaredGradientDecayFactor", Value::Num(0.8))],
    );
    assert_eq!(
        TrainingConfig::from_value(&rmsprop, "trainnet")
            .unwrap()
            .solver,
        Solver::Rmsprop
    );
    let duplicate = options(
        "sgdm",
        &[
            ("MaxEpochs", Value::Num(3.0)),
            ("maxepochs", Value::Num(5.0)),
        ],
    );
    assert!(TrainingConfig::from_value(&duplicate, "trainNetwork")
        .err()
        .unwrap()
        .message()
        .contains("duplicate"));
}
