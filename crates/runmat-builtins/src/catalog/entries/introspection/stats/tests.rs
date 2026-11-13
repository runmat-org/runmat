use super::*;
use runmat_types::*;
fn request(outputs: usize) -> CallRequest {
    CallRequest {
        arguments: vec![ValueFact::scalar(ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(crate::MEMOIZED_FUNCTION_CLASS.owned()),
            properties: Default::default(),
            properties_complete: false,
            handle_semantics: Some(true),
        }))],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}
#[test]
fn default_binding_output_facts_and_bounds_are_typed() {
    let entry = &STATS_CATALOG_ENTRY;
    assert_eq!(entry.bindings.len(), 1);
    let result = inference::infer(&request(1), entry);
    assert!(result.diagnostics.is_empty(), "{:#?}", result.diagnostics);
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Struct(_)));
    let result = inference::infer(&request(2), entry);
    assert!(!result.diagnostics.is_empty());
    let mut renamed = *entry;
    renamed.identity = crate::BuiltinCatalogIdentity {
        name: "display-only-renamed",
    };
    assert_eq!(
        inference::infer(&request(1), &renamed).outputs,
        inference::infer(&request(1), entry).outputs
    );
}
#[test]
fn excess_inputs_reject_when_the_form_is_bounded() {
    let entry = &STATS_CATALOG_ENTRY;
    let mut call = request(1);
    call.arguments = vec![ValueFact::unknown(DynamicReason::RuntimeValue); 5];
    assert!(!inference::infer(&call, entry).diagnostics.is_empty());
}
