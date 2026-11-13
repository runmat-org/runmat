use super::*;
use runmat_types::*;

fn request(arguments: Vec<ValueFact>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

fn scalar_double() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}

#[test]
fn line_terminators_eof_and_padding_are_soundly_classified() {
    let inferred = inference::infer(
        &request(vec![scalar_double(), scalar_double()], 3),
        &FGETS_CATALOG_ENTRY,
    );
    assert!(inferred.diagnostics.is_empty());
    assert!(matches!(inferred.outputs[0].kind, ValueKindFact::Unknown));
    assert!(matches!(inferred.outputs[1].kind, ValueKindFact::Unknown));
    assert!(matches!(
        inferred.outputs[2].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real
        })
    ));
}

#[test]
fn known_invalid_controls_are_diagnosed_without_guessing_unknowns() {
    let invalid = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(1)]),
        StorageFact::Dense,
    );
    assert!(!inference::infer(
        &request(vec![scalar_double(), invalid], 1),
        &FGETS_CATALOG_ENTRY
    )
    .diagnostics
    .is_empty());
    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    assert!(inference::infer(
        &request(vec![unknown.clone(), unknown], 1),
        &FGETS_CATALOG_ENTRY
    )
    .diagnostics
    .is_empty());
}
