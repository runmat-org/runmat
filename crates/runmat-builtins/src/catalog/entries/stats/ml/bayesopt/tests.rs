use super::*;
use runmat_types::{
    standard, CallRequest, CallableFact, DynamicReason, LiteralContext, LiteralValue, ObjectFact,
    OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

fn call(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> runmat_types::CallInference {
    inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
        &BAYESOPT_CATALOG_ENTRY,
    )
}

fn objective() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity: None,
        capabilities: Default::default(),
        parameters: vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
        parameters_complete: true,
        outputs: vec![ValueFact::scalar(ValueKindFact::Numeric(
            runmat_types::NumericFact {
                class: runmat_types::NumericClass::Double,
                domain: runmat_types::NumericDomain::Real,
            },
        ))],
        outputs_complete: true,
        variadic_inputs: false,
        variadic_outputs: false,
        captures: Vec::new(),
        captures_complete: true,
    }))
}

fn variable() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(standard::OPTIMIZABLE_VARIABLE.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

#[test]
fn discarded_result_keeps_search_effects_without_inventing_an_output() {
    let result = inference::infer(
        &CallRequest {
            arguments: vec![objective(), variable()],
            literals: LiteralContext::new(vec![]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(0)),
        },
        &BAYESOPT_CATALOG_ENTRY,
    );
    assert!(result.diagnostics.is_empty());
    assert!(result.outputs.is_empty());
    assert_eq!(result.effects, BAYESOPT_CATALOG_ENTRY.contract.effect_set());
    for label in ["bayesopt(fun, vars)", "bayesopt(fun, vars, Name, Value)"] {
        assert!(BAYESOPT_DESCRIPTOR
            .signatures
            .iter()
            .any(|signature| signature.label == label && signature.outputs.is_empty()));
    }
}

#[test]
fn public_bayesian_option_identity_is_closed() {
    assert_eq!(
        BayesoptOption::from_public_name("MAXOBJECTIVEEVALUATIONS"),
        Some(BayesoptOption::MaxObjectiveEvaluations)
    );
    assert_eq!(BayesoptOption::from_public_name("unknown"), None);
    assert_eq!(
        BayesoptAcquisition::from_public_name("expected_improvement_plus"),
        Some(BayesoptAcquisition::ExpectedImprovementPlus)
    );
    assert_eq!(
        standard::BAYESIAN_OPTIMIZATION.display_name(),
        "BayesianOptimization"
    );
}

#[test]
fn known_valid_unknown_and_invalid_calls_have_sound_result_facts() {
    let valid = call(vec![objective(), variable()], vec![]);
    assert!(valid.diagnostics.is_empty());
    assert!(matches!(valid.outputs[0].kind, ValueKindFact::Object(_)));

    let unknown = call(
        vec![objective(), ValueFact::unknown(DynamicReason::RuntimeValue)],
        vec![],
    );
    assert!(unknown.diagnostics.is_empty());
    assert_eq!(unknown.outputs[0].kind, ValueKindFact::Unknown);

    let unknown_option = call(
        vec![
            objective(),
            variable(),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::Logical),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Bool(false),
        ],
    );
    assert_eq!(unknown_option.outputs[0].kind, ValueKindFact::Unknown);

    let invalid = call(
        vec![
            objective(),
            variable(),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("unknown".into()),
            LiteralValue::String("value".into()),
        ],
    );
    assert!(!invalid.diagnostics.is_empty());
    assert_eq!(invalid.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn callable_capabilities_flow_through_optimization_inference() {
    let mut objective = objective();
    let ValueKindFact::Callable(callable) = &mut objective.kind else {
        panic!("fixture callable")
    };
    callable
        .capabilities
        .0
        .insert(runmat_types::CapabilityRequirement::Network);
    let result = call(vec![objective, variable()], vec![]);
    assert!(result
        .capabilities
        .0
        .contains(&runmat_types::CapabilityRequirement::Network));
    assert!(result
        .effects
        .0
        .contains(&runmat_types::EffectKind::HostCallback));
}
