use super::*;
use runmat_types::*;
fn call(outputs: usize) -> CallRequest {
    CallRequest {
        arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}
#[test]
fn output_and_bounds_do_not_fabricate_callable_authority() {
    let entry = &FUNC2STR_CATALOG_ENTRY;
    assert_eq!(entry.bindings, &REQUIRED_DEFAULT_BINDING);
    for outputs in [0, 1] {
        let result = inference::infer(&call(outputs), entry);
        assert!(result.diagnostics.is_empty(), "{:#?}", result.diagnostics);
        if outputs == 1 {
            assert_eq!(result.outputs[0].kind, ValueKindFact::String);
        }
    }
    assert!(!inference::infer(&call(2), entry).diagnostics.is_empty());
    let mut renamed = *entry;
    renamed.identity = BuiltinCatalogIdentity {
        name: "display-only-renamed",
    };
    assert_eq!(
        inference::infer(&call(1), entry).outputs,
        inference::infer(&call(1), &renamed).outputs
    );
}
#[test]
fn known_nonhandles_and_wrong_arity_reject() {
    let entry = &FUNC2STR_CATALOG_ENTRY;
    let mut request = call(1);
    for kind in [
        ValueKindFact::String,
        ValueKindFact::Logical,
        ValueKindFact::Character,
    ] {
        request.arguments = vec![ValueFact::scalar(kind)];
        assert!(!inference::infer(&request, entry).diagnostics.is_empty());
    }
    for count in [0, 2] {
        request.arguments = vec![ValueFact::unknown(DynamicReason::RuntimeValue); count];
        assert!(!inference::infer(&request, entry).diagnostics.is_empty());
    }
}

#[test]
fn callable_metadata_does_not_infer_callback_execution() {
    let mut request = call(1);
    request.arguments = vec![ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity: None,
        capabilities: Default::default(),
        parameters: vec![],
        parameters_complete: false,
        outputs: vec![],
        outputs_complete: false,
        variadic_inputs: true,
        variadic_outputs: true,
        captures: vec![],
        captures_complete: false,
    }))];
    let result = inference::infer(&request, &FUNC2STR_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].kind, ValueKindFact::String);
    assert_eq!(
        FUNC2STR_CATALOG_ENTRY.contract.effects,
        &[EffectKind::MayThrow]
    );
    assert!(FUNC2STR_CATALOG_ENTRY.contract.capabilities.is_empty());
}
