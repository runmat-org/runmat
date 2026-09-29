use super::*;
use runmat_types::*;

fn request(arguments: Vec<ValueFact>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

#[test]
fn five_scalar_inputs_return_host_double_and_excess_output_is_diagnosed() {
    let scalar = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let args = vec![scalar; 5];
    let inferred = crate::infer_catalog_call(&BLSIMPV_CATALOG_ENTRY, &request(args.clone(), 1));
    assert!(
        inferred.diagnostics.is_empty(),
        "{:?}",
        inferred.diagnostics
    );
    assert_eq!(
        inferred.outputs[0].numeric().map(|numeric| numeric.class),
        Some(NumericClass::Double)
    );
    assert_eq!(inferred.outputs[0].residency, ResidencyFact::Host);
    assert!(
        crate::infer_catalog_call(&BLSIMPV_CATALOG_ENTRY, &request(args, 2))
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-BLSIMPV-OUTPUTS")
    );
}

#[test]
fn option_semantics_and_legacy_visible_examples_are_preserved() {
    let signature = BLSIMPV_CATALOG_ENTRY
        .descriptor
        .signatures
        .iter()
        .find(|s| s.label.contains("Name, Value"))
        .expect("name-value signature");
    assert!(signature
        .inputs
        .iter()
        .any(|input| input.name == "NameValue"));
    assert_eq!(BLSIMPV_CATALOG_ENTRY.documentation.examples.len(), 2);
    for example in BLSIMPV_CATALOG_ENTRY.documentation.examples {
        assert!(example.program.contains("fprintf"));
        assert!(example.display_output.is_some());
        assert!(matches!(
            example.verification,
            BuiltinExampleVerification::Assertions { .. }
        ));
    }
}
