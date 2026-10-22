use super::*;
use runmat_types::*;
#[test]
fn known_host_classes_are_retained_without_guessing_device_wrapper_identity() {
    let mut call = request(1);
    call.arguments[0] = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Real,
    }));
    let result = inference::infer(&call, &METACLASS_CATALOG_ENTRY);
    assert!(
        matches!(&result.outputs[0].kind,ValueKindFact::ClassReference(c) if c.runtime_class.as_ref().is_some_and(|c|c.is(standard::UINT64)))
    );
    call.arguments[0].residency = ResidencyFact::Device { provider: None };
    let result = inference::infer(&call, &METACLASS_CATALOG_ENTRY);
    assert!(
        matches!(&result.outputs[0].kind,ValueKindFact::ClassReference(c) if c.runtime_class.is_none())
    );
}
fn request(outputs: usize) -> CallRequest {
    CallRequest {
        arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}
#[test]
fn default_binding_output_facts_and_bounds_are_typed() {
    let entry = &METACLASS_CATALOG_ENTRY;
    assert_eq!(entry.bindings.len(), 1);
    let result = inference::infer(&request(1), entry);
    assert!(result.diagnostics.is_empty(), "{:#?}", result.diagnostics);
    assert!(matches!(
        result.outputs[0].kind,
        ValueKindFact::ClassReference(_)
    ));
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
    let entry = &METACLASS_CATALOG_ENTRY;
    let mut call = request(1);
    call.arguments = vec![ValueFact::unknown(DynamicReason::RuntimeValue); 5];
    assert!(!inference::infer(&call, entry).diagnostics.is_empty());
}
