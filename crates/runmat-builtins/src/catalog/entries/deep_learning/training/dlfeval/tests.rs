use super::*;
use runmat_types::{
    CallableFact, CapabilityRequirement, CapabilitySet, LiteralContext, NumericClass,
    NumericDomain, NumericFact, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

fn request(arguments: Vec<ValueFact>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

fn callable() -> ValueFact {
    let output = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::Scalar,
        StorageFact::Scalar,
    );
    ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity: None,
        capabilities: CapabilitySet([CapabilityRequirement::Filesystem].into_iter().collect()),
        parameters: vec![],
        parameters_complete: false,
        outputs: vec![output.clone(), output],
        outputs_complete: true,
        variadic_inputs: true,
        variadic_outputs: false,
        captures: vec![],
        captures_complete: false,
    }))
}

#[test]
fn callback_outputs_and_capabilities_flow_through_requested_arity() {
    let inferred = crate::infer_catalog_call(&DLFEVAL_CATALOG_ENTRY, &request(vec![callable()], 2));
    assert!(
        inferred.diagnostics.is_empty(),
        "{:?}",
        inferred.diagnostics
    );
    assert_eq!(inferred.outputs.len(), 2);
    assert_eq!(
        inferred.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );
    assert!(inferred
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));
    assert!(inferred.effects.0.contains(&EffectKind::HostCallback));
    assert!(inferred.effects.0.contains(&EffectKind::EnvironmentWrite));
    let too_many = crate::infer_catalog_call(&DLFEVAL_CATALOG_ENTRY, &request(vec![callable()], 3));
    assert!(!too_many.diagnostics.is_empty());
}

#[test]
fn known_nonhandle_rejects_and_unknown_callback_does_not_invent_outputs() {
    let text = ValueFact::scalar(ValueKindFact::String);
    let rejected = crate::infer_catalog_call(&DLFEVAL_CATALOG_ENTRY, &request(vec![text], 1));
    assert!(rejected
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-DLFEVAL-FUNCTION"));
    let unknown = crate::infer_catalog_call(
        &DLFEVAL_CATALOG_ENTRY,
        &request(
            vec![ValueFact::unknown(
                runmat_types::DynamicReason::RuntimeValue,
            )],
            2,
        ),
    );
    assert_eq!(unknown.outputs.len(), 2);
    assert!(unknown
        .outputs
        .iter()
        .all(|output| matches!(&output.kind, ValueKindFact::Unknown)));
    let missing = crate::infer_catalog_call(&DLFEVAL_CATALOG_ENTRY, &request(vec![], 1));
    assert!(missing
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DL-DLFEVAL-ARITY"));
}

#[test]
fn catalog_declares_process_stack_and_runtime_owned_callback_boundary() {
    assert_eq!(
        DLFEVAL_CATALOG_ENTRY.link.execution_stack,
        ExecutionStackRequirement::Process
    );
    assert_eq!(
        DLFEVAL_CATALOG_ENTRY.placement.preparation,
        BuiltinOperandPreparation::RuntimeOwned
    );
    assert_eq!(
        DLFEVAL_DESCRIPTOR.output_mode,
        BuiltinOutputMode::ByRequestedOutputCount
    );
}
