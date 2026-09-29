use super::*;
use runmat_types::*;

fn request(arguments: Vec<ValueFact>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

fn real(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}

#[test]
fn scalar_broadcast_and_output_count_are_sound() {
    let args = vec![
        real(ShapeFact::from(vec![Some(3), Some(1)])),
        real(ShapeFact::Scalar),
        real(ShapeFact::Scalar),
        real(ShapeFact::Scalar),
        real(ShapeFact::Scalar),
    ];
    let inferred = crate::infer_catalog_call(&BLSPRICE_CATALOG_ENTRY, &request(args.clone(), 2));
    assert!(
        inferred.diagnostics.is_empty(),
        "{:?}",
        inferred.diagnostics
    );
    assert_eq!(inferred.outputs.len(), 2);
    for output in inferred.outputs {
        assert_eq!(output.shape, ShapeFact::from(vec![Some(3), Some(1)]));
        assert_eq!(
            output.numeric().map(|numeric| numeric.class),
            Some(NumericClass::Double)
        );
        assert_eq!(output.residency, ResidencyFact::Host);
    }
    let excess = crate::infer_catalog_call(&BLSPRICE_CATALOG_ENTRY, &request(args, 3));
    assert!(excess
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-BLSPRICE-OUTPUTS"));
}

#[test]
fn resident_result_does_not_claim_host_double_or_specific_precision() {
    let mut resident = real(ShapeFact::from(vec![Some(3), Some(1)]));
    resident.residency = ResidencyFact::Unknown;
    let args = vec![
        resident,
        real(ShapeFact::Scalar),
        real(ShapeFact::Scalar),
        real(ShapeFact::Scalar),
        real(ShapeFact::Scalar),
    ];
    let inferred = crate::infer_catalog_call(&BLSPRICE_CATALOG_ENTRY, &request(args, 1));
    assert!(
        inferred.diagnostics.is_empty(),
        "{:?}",
        inferred.diagnostics
    );
    assert_eq!(inferred.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(1)])
    );
    assert_eq!(inferred.outputs[0].residency, ResidencyFact::Unknown);
}

#[test]
fn legacy_visible_examples_and_provider_contract_are_present() {
    assert_eq!(BLSPRICE_CATALOG_ENTRY.documentation.examples.len(), 2);
    for example in BLSPRICE_CATALOG_ENTRY.documentation.examples {
        assert!(example.program.contains("fprintf"));
        assert!(example.display_output.is_some());
        assert!(matches!(
            example.verification,
            BuiltinExampleVerification::Assertions { .. }
        ));
    }
    assert_eq!(
        BLSPRICE_CATALOG_ENTRY.placement.accelerator,
        BuiltinAcceleratorPolicy::Optional
    );
    assert_eq!(
        BLSPRICE_CATALOG_ENTRY.extensions,
        &super::super::contract::BLSPRICE_EXTENSIONS
    );
}
