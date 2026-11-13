use super::*;
use runmat_types::*;

fn request(input: ValueFact, outputs: usize) -> CallRequest {
    CallRequest {
        arguments: vec![input],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

#[test]
fn line_or_eof_is_dynamic_but_padding_is_double() {
    let input = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let result = inference::infer(&request(input, 2), &FGETL_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
    assert!(matches!(
        result.outputs[1].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real
        })
    ));
    assert!(result.effects.0.contains(&EffectKind::FilesystemRead));
}

#[test]
fn known_invalid_geometry_and_class_are_diagnosed() {
    for input in [
        ValueFact::scalar(ValueKindFact::Character),
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(1), Some(2)]),
            StorageFact::Dense,
        ),
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        })),
    ] {
        assert!(!inference::infer(&request(input, 1), &FGETL_CATALOG_ENTRY)
            .diagnostics
            .is_empty());
    }
}
