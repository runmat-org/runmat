use super::*;
use runmat_types::*;

fn call(arguments: Vec<ValueFact>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

fn unknown() -> ValueFact {
    ValueFact::unknown(DynamicReason::RuntimeValue)
}

#[test]
fn pivot_has_canonical_host_contract_and_asserted_example() {
    assert_eq!(
        ENTRY.documentation.authority,
        BuiltinDocumentationAuthority::Catalog
    );
    assert_eq!(ENTRY.bindings, &REQUIRED_DEFAULT_BINDING);
    assert_eq!(
        ENTRY.placement.accelerator,
        BuiltinAcceleratorPolicy::Forbidden
    );
    assert_eq!(ENTRY.placement.fusion, BuiltinFusionPolicy::Boundary);
    assert_eq!(ENTRY.integer_capabilities.len(), 2);
    assert!(ENTRY.documentation.examples.iter().all(|example| matches!(
        example.verification,
        BuiltinExampleVerification::Assertions { .. }
    )));
}

#[test]
fn pivot_infers_only_table_identity_not_runtime_schema() {
    let result = infer_catalog_call(
        &ENTRY,
        &call(vec![unknown(), unknown(), unknown(), unknown()], 1),
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    let ValueKindFact::Object(object) = &result.outputs[0].kind else {
        panic!("pivot output must be a table")
    };
    assert_eq!(object.runtime_class, Some(standard::TABLE.owned()));
    assert!(!object.properties_complete);
}

#[test]
fn pivot_diagnoses_proven_bad_arity_and_receiver() {
    let short = infer_catalog_call(&ENTRY, &call(vec![unknown(); 3], 1));
    assert!(!short.diagnostics.is_empty());
    let odd_options = infer_catalog_call(&ENTRY, &call(vec![unknown(); 5], 1));
    assert!(!odd_options.diagnostics.is_empty());
    let mut args = vec![unknown(); 4];
    args[0] = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    assert!(!infer_catalog_call(&ENTRY, &call(args, 1))
        .diagnostics
        .is_empty());
}
