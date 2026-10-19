use super::*;
use runmat_types::{
    CallRequest, LiteralContext, LiteralValue, OutputSelection, RequestedOutputCount, ValueFact,
    ValueKindFact,
};

fn infer(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallInference {
    crate::infer_catalog_call(
        &TRAINING_OPTIONS_CATALOG_ENTRY,
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
    )
}

#[test]
fn literal_option_names_preserve_value_facts_in_host_object() {
    let integer = ValueFact::scalar(ValueKindFact::Numeric(runmat_types::NumericFact {
        class: runmat_types::NumericClass::UInt64,
        domain: runmat_types::NumericDomain::Real,
    }));
    let result = infer(
        vec![
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
            integer.clone(),
        ],
        vec![
            LiteralValue::String("adam".into()),
            LiteralValue::String("MaxEpochs".into()),
            LiteralValue::Unknown,
        ],
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    let ValueKindFact::Object(object) = &result.outputs[0].kind else {
        panic!("expected options object")
    };
    assert_eq!(
        object.runtime_class,
        Some(super::TrainingSolver::Adam.class_identity().owned())
    );
    assert_eq!(object.properties["MaxEpochs"], integer);
    assert!(object.properties_complete);
}

#[test]
fn dynamic_name_cannot_prove_default_properties_and_bad_solver_diagnoses() {
    let dynamic = infer(
        vec![
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::Logical),
        ],
        vec![
            LiteralValue::String("adam".into()),
            LiteralValue::Unknown,
            LiteralValue::Unknown,
        ],
    );
    let ValueKindFact::Object(object) = &dynamic.outputs[0].kind else {
        panic!("expected options object")
    };
    assert!(!object.properties_complete);
    assert!(object.properties.is_empty());
    let invalid = infer(
        vec![ValueFact::scalar(ValueKindFact::String)],
        vec![LiteralValue::String("unsupported".into())],
    );
    assert!(!invalid.diagnostics.is_empty());
    assert_eq!(
        TRAINING_OPTIONS_DESCRIPTOR.errors[0].identifier,
        Some("RunMat:deepLearning:InvalidInput")
    );
}

#[test]
fn solver_literal_controls_closed_property_set_but_unknown_solver_keeps_common_facts() {
    for (name, expected_squared) in [("sgdm", false), ("adam", true), ("rmsprop", true)] {
        let result = infer(
            vec![ValueFact::scalar(ValueKindFact::String)],
            vec![LiteralValue::String(name.into())],
        );
        let ValueKindFact::Object(object) = &result.outputs[0].kind else {
            panic!("expected options object")
        };
        assert!(object.properties_complete, "{name}");
        assert_eq!(
            object.runtime_class,
            Some(
                super::TrainingSolver::from_name(name)
                    .unwrap()
                    .class_identity()
                    .owned()
            )
        );
        assert_eq!(
            object.properties.contains_key("SquaredGradientDecayFactor"),
            expected_squared,
            "{name}"
        );
    }
    let unknown = infer(
        vec![ValueFact::scalar(ValueKindFact::String)],
        vec![LiteralValue::Unknown],
    );
    let ValueKindFact::Object(object) = &unknown.outputs[0].kind else {
        panic!("expected options object")
    };
    assert!(!object.properties_complete);
    assert!(object.runtime_class.is_none());
    assert!(object.properties.contains_key("InitialLearnRate"));
    assert!(!object.properties.contains_key("SquaredGradientDecayFactor"));
}

#[test]
fn solver_name_cannot_be_overridden_as_a_named_option() {
    let result = infer(
        vec![
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
        ],
        vec![
            LiteralValue::String("sgdm".into()),
            LiteralValue::String("SolverName".into()),
            LiteralValue::String("adam".into()),
        ],
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-TRAININGOPTIONS-SOLVER"));
}
