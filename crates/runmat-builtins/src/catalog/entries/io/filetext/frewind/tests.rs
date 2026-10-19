use super::*;
use runmat_types::*;

fn request(input: ValueFact, count: usize) -> CallRequest {
    CallRequest {
        arguments: vec![input],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
    }
}

#[test]
fn admitted_and_unknown_file_identifiers_have_no_result_but_keep_effects() {
    for input in [
        ValueFact::unknown(DynamicReason::RuntimeValue),
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        })),
    ] {
        let inferred = inference::infer(&request(input, 0), &FREWIND_CATALOG_ENTRY);
        assert!(inferred.outputs.is_empty());
        assert!(inferred.diagnostics.is_empty());
        assert!(inferred.effects.0.contains(&EffectKind::EnvironmentWrite));
    }
}

#[test]
fn known_invalid_class_and_geometry_are_diagnosed() {
    for input in [
        ValueFact::scalar(ValueKindFact::Character),
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(2), Some(1)]),
            StorageFact::Dense,
        ),
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        })),
    ] {
        assert!(
            !inference::infer(&request(input, 0), &FREWIND_CATALOG_ENTRY)
                .diagnostics
                .is_empty()
        );
    }
    let adapter_request = inference::infer(
        &request(ValueFact::unknown(DynamicReason::RuntimeValue), 1),
        &FREWIND_CATALOG_ENTRY,
    );
    assert!(adapter_request.diagnostics.is_empty());
    assert_eq!(adapter_request.outputs.len(), 1);
    assert!(matches!(
        adapter_request.outputs[0].kind,
        ValueKindFact::Unknown
    ));
}
