use super::*;
use runmat_types::*;

fn request(name: ValueFact, literal: LiteralValue, outputs: usize) -> CallRequest {
    CallRequest {
        arguments: vec![name, ValueFact::unknown(DynamicReason::RuntimeValue)],
        literals: LiteralContext::new(vec![literal, LiteralValue::Unknown]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

#[test]
fn dynamic_target_does_not_fabricate_output_class_or_residency() {
    let name = ValueFact::scalar(ValueKindFact::String);
    let call = request(name.clone(), LiteralValue::String("plus".into()), 2);
    let inferred = inference::infer(&call, &BUILTIN_CATALOG_ENTRY);
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(inferred.outputs.len(), 2);
    assert!(inferred.outputs.iter().all(|output| {
        output.kind == ValueKindFact::Unknown && output.residency == ResidencyFact::Unknown
    }));

    let unknown = request(name, LiteralValue::Unknown, 0);
    assert!(inference::infer(&unknown, &BUILTIN_CATALOG_ENTRY)
        .outputs
        .is_empty());
    let invalid = request(
        ValueFact::scalar(ValueKindFact::Logical),
        LiteralValue::Bool(true),
        1,
    );
    assert!(!inference::infer(&invalid, &BUILTIN_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}

#[test]
fn descriptor_preserves_variadic_integer_delegation() {
    assert_eq!(
        BUILTIN_DESCRIPTOR.output_mode,
        BuiltinOutputMode::ByRequestedOutputCount
    );
    assert_eq!(
        BUILTIN_DESCRIPTOR.completion_policy,
        BuiltinCompletionPolicy::Public
    );
    assert_eq!(
        BUILTIN_DESCRIPTOR.signatures[0].outputs[0].arity,
        BuiltinParamArity::Variadic
    );
    assert_eq!(BUILTIN_INTEGER_CAPABILITIES.len(), 1);
    assert_eq!(
        BUILTIN_INTEGER_CAPABILITIES[0].backend,
        BuiltinIntegerBackendRule::FunctionSpecific
    );
}
