use super::*;

fn options(solver: &str, extra: &[(&str, Value)]) -> Value {
    let mut object = ObjectInstance::new(super::super::super::TRAINING_OPTIONS_CLASS.owned());
    object
        .properties
        .insert("SolverName".into(), Value::String(solver.into()));
    for (name, value) in extra {
        object.properties.insert((*name).into(), value.clone());
    }
    Value::Object(object)
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
        ("L2Regularization", Value::Num(0.01)),
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
        .contains("Adam controls"));
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
