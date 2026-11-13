use super::*;
#[test]
fn legacy_names_are_separate_and_unambiguous() {
    assert_eq!(
        FminsearchOptionName::from_public("tolx"),
        Some(FminsearchOptionName::TolX)
    );
    assert_eq!(FminsearchOptionName::from_public("Max"), None);
    assert_eq!(
        FminsearchOptionName::from_public("OutputF"),
        Some(FminsearchOptionName::OutputFcn)
    );
    assert!(crate::OptimSolver::parse("fminsearch").is_none());
    assert_eq!(FMINSEARCH_CATALOG_ENTRY.identity.name, "fminsearch");
    assert_eq!(
        FMINSEARCH_DESCRIPTOR.output_mode,
        crate::BuiltinOutputMode::ByRequestedOutputCount
    );
}

#[test]
fn inference_preserves_shape_class_metadata_and_opaque_callback_effects() {
    use runmat_types::*;
    let initial = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    let request = CallRequest {
        arguments: vec![
            ValueFact::unknown(DynamicReason::RuntimeValue),
            initial.clone(),
        ],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(4)),
    };
    let result = inference::infer(&request, &FMINSEARCH_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs.len(), 4);
    assert_eq!(result.outputs[0].shape, initial.shape);
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    let ValueKindFact::Struct(metadata) = &result.outputs[3].kind else {
        panic!("metadata")
    };
    assert!(metadata.fields_complete);
    for name in ["iterations", "funcCount", "algorithm", "message"] {
        assert!(metadata.fields.contains_key(name));
    }
    assert!(result.effects.0.contains(&EffectKind::Unknown));
    assert!(result.effects.0.contains(&EffectKind::HostCallback));
    assert!(result.effects.0.contains(&EffectKind::UserInterface));
    let mut invalid = request;
    invalid.arguments[1].kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Single,
        domain: NumericDomain::Real,
    });
    assert!(!inference::infer(&invalid, &FMINSEARCH_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}
