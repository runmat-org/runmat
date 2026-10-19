use super::*;
use runmat_types::{
    BuiltinId, CallRequest, CallableFact, CallableIdentity, CapabilitySet, EffectKind,
    LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact, OutputSelection,
    RequestedOutputCount, ValueFact, ValueKindFact,
};

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

#[test]
fn timeit_returns_scalar_double_for_known_and_unknown_callbacks() {
    for callback in [
        ValueFact::scalar(ValueKindFact::String),
        ValueFact::unknown(runmat_types::DynamicReason::UnresolvedCallable),
    ] {
        let result = inference::infer(&request(vec![callback], Vec::new()), &TIMEIT_CATALOG_ENTRY);
        assert_eq!(result.outputs.len(), 1);
        assert_eq!(
            result.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })
        );
    }
}

#[test]
fn timeit_rejects_invalid_count_and_handle_before_claiming_runtime_facts() {
    let invalid_count = inference::infer(
        &request(
            vec![
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })),
            ],
            vec![
                LiteralValue::String("@sin".into()),
                LiteralValue::Number(-1.0),
            ],
        ),
        &TIMEIT_CATALOG_ENTRY,
    );
    assert!(invalid_count
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-TIMEIT-OUTPUT-COUNT"));
    let invalid_handle = inference::infer(
        &request(
            vec![ValueFact::scalar(ValueKindFact::String)],
            vec![LiteralValue::String("sin".into())],
        ),
        &TIMEIT_CATALOG_ENTRY,
    );
    assert!(invalid_handle
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-TIMEIT-HANDLE-TEXT"));
}

#[test]
fn timeit_catalog_preserves_contract_and_examples() {
    assert_eq!(TIMEIT_CATALOG_ENTRY.descriptor.signatures.len(), 2);
    assert_eq!(TIMEIT_CATALOG_ENTRY.extensions.len(), 2);
    assert_eq!(TIMEIT_CATALOG_ENTRY.integer_capabilities.len(), 1);
    assert_eq!(TIMEIT_CATALOG_ENTRY.documentation.examples.len(), 6);
}

#[test]
fn known_callback_effects_propagate_and_unknown_callbacks_remain_conservative() {
    let callback = CallableFact {
        identity: Some(CallableIdentity::Builtin(BuiltinId("tic".into()))),
        capabilities: CapabilitySet::default(),
        parameters: Vec::new(),
        parameters_complete: true,
        outputs: Vec::new(),
        outputs_complete: false,
        variadic_inputs: false,
        variadic_outputs: true,
        captures: Vec::new(),
        captures_complete: true,
    };
    let known = inference::infer(
        &request(
            vec![ValueFact::scalar(ValueKindFact::Callable(callback))],
            Vec::new(),
        ),
        &TIMEIT_CATALOG_ENTRY,
    );
    assert!(known.effects.0.contains(&EffectKind::EnvironmentWrite));
    assert!(!known.effects.0.contains(&EffectKind::Unknown));
    let unknown = inference::infer(
        &request(
            vec![ValueFact::unknown(
                runmat_types::DynamicReason::UnresolvedCallable,
            )],
            Vec::new(),
        ),
        &TIMEIT_CATALOG_ENTRY,
    );
    assert!(unknown.effects.0.contains(&EffectKind::Unknown));
}
