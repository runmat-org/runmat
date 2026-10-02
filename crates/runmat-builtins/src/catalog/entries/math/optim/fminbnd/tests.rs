use super::*;
use runmat_types::*;

fn scalar() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}

fn request(lower: f64, upper: f64, outputs: usize) -> CallRequest {
    CallRequest {
        arguments: vec![
            ValueFact::unknown(DynamicReason::RuntimeValue),
            scalar(),
            scalar(),
        ],
        literals: LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Number(lower),
            LiteralValue::Number(upper),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

#[test]
fn minimum_facts_preserve_conditional_empty_geometry_and_requested_outputs() {
    let valid = inference::infer(&request(0.0, 5.0, 4), &FMINBND_CATALOG_ENTRY);
    assert!(valid.diagnostics.is_empty());
    assert_eq!(valid.outputs[0].shape, ShapeFact::Scalar);
    assert_eq!(
        valid.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert!(matches!(valid.outputs[3].kind, ValueKindFact::Struct(_)));

    let reversed = inference::infer(&request(5.0, 0.0, 2), &FMINBND_CATALOG_ENTRY);
    assert!(reversed.diagnostics.is_empty());
    assert_eq!(reversed.outputs[0].shape, ShapeFact::Unknown);
    assert_eq!(
        reversed.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );

    let excessive = inference::infer(&request(0.0, 5.0, 5), &FMINBND_CATALOG_ENTRY);
    assert!(!excessive.diagnostics.is_empty());
    assert!(excessive
        .outputs
        .iter()
        .all(|fact| matches!(fact.kind, ValueKindFact::Unknown)));
}

#[test]
fn known_complex_bounds_and_nonstruct_options_are_diagnosed() {
    let mut call = request(0.0, 5.0, 1);
    call.arguments[1] = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Complex,
    }));
    call.arguments
        .push(ValueFact::scalar(ValueKindFact::String));
    let facts = inference::infer(&call, &FMINBND_CATALOG_ENTRY);
    assert!(facts.diagnostics.len() >= 2);
    assert!(matches!(facts.outputs[0].kind, ValueKindFact::Unknown));
}

#[test]
fn known_nonscalar_callback_text_is_not_inferred_as_valid() {
    let mut call = request(0.0, 5.0, 1);
    call.arguments[0] = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::from(vec![Some(1), Some(2)]),
        StorageFact::Dense,
    );
    let facts = inference::infer(&call, &FMINBND_CATALOG_ENTRY);
    assert!(!facts.diagnostics.is_empty());
    assert!(matches!(facts.outputs[0].kind, ValueKindFact::Unknown));
}
