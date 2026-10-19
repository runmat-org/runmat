use super::*;
use runmat_types::*;
fn request(length: LiteralValue, options: Vec<LiteralValue>) -> CallRequest {
    let mut literals = vec![length];
    literals.extend(options.iter().cloned());
    let mut arguments = vec![ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))];
    arguments.extend(
        options
            .iter()
            .map(|_| ValueFact::scalar(ValueKindFact::String)),
    );
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    }
}
#[test]
fn window_shape_class_and_invalid_options() {
    let entry = &BLACKMAN_CATALOG_ENTRY;
    let base = inference::infer(&request(LiteralValue::Number(8.0), vec![]), entry);
    assert!(base.diagnostics.is_empty());
    assert_eq!(
        base.outputs[0].shape,
        ShapeFact::from(vec![Some(8), Some(1)])
    );
    assert_eq!(
        base.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    let single = inference::infer(
        &request(
            LiteralValue::Number(8.0),
            vec![LiteralValue::String("single".into())],
        ),
        entry,
    );
    assert_eq!(
        single.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );
    let bad = inference::infer(&request(LiteralValue::Number(-1.0), vec![]), entry);
    assert!(!bad.diagnostics.is_empty());
    assert_eq!(bad.outputs[0].kind, ValueKindFact::Unknown);
    let misplaced = inference::infer(
        &request(
            LiteralValue::Number(4.0),
            vec![
                LiteralValue::String("single".into()),
                LiteralValue::String("periodic".into()),
            ],
        ),
        entry,
    );
    assert!(!misplaced.diagnostics.is_empty());
}

#[test]
fn character_row_option_is_valid_but_multirow_is_not() {
    let mut row = request(
        LiteralValue::Number(4.0),
        vec![LiteralValue::Character("periodic".into())],
    );
    row.arguments[1] = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), Some(8)]),
        StorageFact::Dense,
    );
    let accepted = inference::infer(&row, &BLACKMAN_CATALOG_ENTRY);
    assert!(accepted.diagnostics.is_empty());
    row.arguments[1].shape = ShapeFact::from(vec![Some(2), Some(4)]);
    let rejected = inference::infer(&row, &BLACKMAN_CATALOG_ENTRY);
    assert!(!rejected.diagnostics.is_empty());
    assert_eq!(rejected.outputs[0].kind, ValueKindFact::Unknown);
}
