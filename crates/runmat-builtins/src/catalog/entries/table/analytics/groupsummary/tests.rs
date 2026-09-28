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

fn table() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(standard::TABLE.owned()),
            properties: Default::default(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

fn numeric() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(3), Some(1)]),
        StorageFact::Dense,
    )
}

#[test]
fn groupsummary_catalog_declares_exact_binding_and_capabilities() {
    assert_eq!(ENTRY.bindings, &REQUIRED_DEFAULT_BINDING);
    assert_eq!(
        ENTRY.documentation.authority,
        BuiltinDocumentationAuthority::Catalog
    );
    assert_eq!(
        ENTRY.descriptor.output_mode,
        BuiltinOutputMode::ByRequestedOutputCount
    );
    assert_eq!(ENTRY.integer_capabilities.len(), 3);
    assert_eq!(ENTRY.extensions.len(), 2);
    assert_eq!(
        ENTRY.placement.accelerator,
        BuiltinAcceleratorPolicy::Forbidden
    );
    assert_eq!(ENTRY.placement.fusion, BuiltinFusionPolicy::Boundary);
    assert!(ENTRY.documentation.examples.iter().all(|example| matches!(
        example.verification,
        BuiltinExampleVerification::Assertions { .. }
    )));
}

#[test]
fn table_input_returns_table_identity_without_invented_schema() {
    let result = infer_catalog_call(&ENTRY, &call(vec![table(), unknown(), unknown()], 1));
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
    let ValueKindFact::Object(object) = &result.outputs[0].kind else {
        panic!("expected table output");
    };
    assert_eq!(object.runtime_class, Some(standard::TABLE.owned()));
    assert!(!object.properties_complete);
    let extra = infer_catalog_call(&ENTRY, &call(vec![table(), unknown(), unknown()], 2));
    assert!(!extra.diagnostics.is_empty());
}

#[test]
fn array_outputs_keep_unknown_summaries_and_double_counts() {
    let result = infer_catalog_call(&ENTRY, &call(vec![numeric(), numeric(), unknown()], 3));
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs.len(), 3);
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
    assert!(matches!(result.outputs[1].kind, ValueKindFact::Unknown));
    assert!(matches!(
        result.outputs[2].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            ..
        })
    ));
    assert_eq!(result.outputs[2].shape.known_dims(), Some(vec![None, None]));
}

#[test]
fn unknown_overload_is_not_claimed_as_a_table_or_numeric_result() {
    let result = infer_catalog_call(&ENTRY, &call(vec![unknown(), unknown(), unknown()], 1));
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
    let short = infer_catalog_call(&ENTRY, &call(vec![numeric(), unknown()], 1));
    assert!(!short.diagnostics.is_empty());
}
