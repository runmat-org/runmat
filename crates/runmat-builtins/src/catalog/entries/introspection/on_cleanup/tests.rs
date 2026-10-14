use super::*;
use runmat_types::*;

fn request(arguments: Vec<ValueFact>, outputs: usize) -> CallRequest {
    CallRequest {
        literals: LiteralContext::new(vec![LiteralValue::Unknown; arguments.len()]),
        arguments,
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

#[test]
fn retained_callback_lifecycle_declares_suspension() {
    assert_eq!(
        ON_CLEANUP_CATALOG_ENTRY.contract.async_behavior,
        BuiltinAsyncBehavior::MaySuspend
    );
    let inferred = inference::infer(
        &request(vec![ValueFact::unknown(DynamicReason::RuntimeValue)], 1),
        &ON_CLEANUP_CATALOG_ENTRY,
    );
    assert!(inferred.effects.0.contains(&EffectKind::HostCallback));
    assert!(inferred.effects.0.contains(&EffectKind::MaySuspend));
    assert!(inferred.effects.0.contains(&EffectKind::Unknown));
}

#[test]
fn callable_unknown_invalid_and_discarded_output() {
    let mut call = request(vec![ValueFact::unknown(DynamicReason::RuntimeValue)], 1);
    let unknown = inference::infer(&call, &ON_CLEANUP_CATALOG_ENTRY);
    assert!(unknown.diagnostics.is_empty());
    assert_eq!(unknown.outputs.len(), 1);
    assert_eq!(unknown.outputs[0].shape, ShapeFact::Scalar);
    assert_eq!(unknown.outputs[0].storage, StorageFact::Opaque);
    assert!(matches!(
        &unknown.outputs[0].kind,
        ValueKindFact::Object(ObjectFact {
            runtime_class: Some(class),
            handle_semantics: Some(true),
            ..
        }) if class.display_name() == "onCleanup"
    ));
    call.arguments[0] = ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity: None,
        capabilities: CapabilitySet::default(),
        parameters: vec![],
        parameters_complete: true,
        outputs: vec![],
        outputs_complete: true,
        variadic_inputs: false,
        variadic_outputs: false,
        captures: vec![],
        captures_complete: true,
    }));
    assert!(inference::infer(&call, &ON_CLEANUP_CATALOG_ENTRY)
        .diagnostics
        .is_empty());

    call.outputs = OutputSelection::new(RequestedOutputCount::Exactly(0));
    assert!(inference::infer(&call, &ON_CLEANUP_CATALOG_ENTRY)
        .outputs
        .is_empty());
    call.outputs = OutputSelection::new(RequestedOutputCount::Exactly(2));
    assert!(!inference::infer(&call, &ON_CLEANUP_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    call.outputs = OutputSelection::new(RequestedOutputCount::Exactly(1));
    call.arguments[0] = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Real,
    }));
    assert!(!inference::infer(&call, &ON_CLEANUP_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    call.arguments.clear();
    assert!(!inference::infer(&call, &ON_CLEANUP_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}
