use super::*;
use runmat_types::*;

fn call(input: ValueFact, count: Option<ValueFact>, literal: Option<LiteralValue>) -> CallRequest {
    let mut arguments = vec![input];
    let mut literals = vec![LiteralValue::Unknown];
    if let Some(count) = count {
        arguments.push(count);
        literals.push(literal.unwrap_or(LiteralValue::Unknown));
    }
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::default(),
    }
}

#[test]
fn head_preserves_known_class_and_trailing_dimensions_without_inventing_count() {
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(10), Some(2), Some(3)]),
        StorageFact::Dense,
    );
    let default = infer_catalog_call(&HEAD_ENTRY, &call(input.clone(), None, None));
    assert_eq!(
        default.outputs[0].shape,
        ShapeFact::from(vec![Some(8), Some(2), Some(3)])
    );
    assert_eq!(default.outputs[0].kind, input.kind);
    let selected = infer_catalog_call(
        &HEAD_ENTRY,
        &call(
            input.clone(),
            Some(ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: NumericDomain::Real,
            }))),
            Some(LiteralValue::Integer {
                text: "2".into(),
                class: NumericClass::UInt64,
            }),
        ),
    );
    assert_eq!(
        selected.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(2), Some(3)])
    );
    let dynamic = infer_catalog_call(
        &HEAD_ENTRY,
        &call(
            input,
            Some(ValueFact::unknown(DynamicReason::RuntimeValue)),
            None,
        ),
    );
    assert_eq!(
        dynamic.outputs[0].shape,
        ShapeFact::from(vec![None, Some(2), Some(3)])
    );
}

#[test]
fn head_diagnoses_bad_control_and_output_count() {
    let input = ValueFact::unknown(DynamicReason::RuntimeValue);
    let bad = infer_catalog_call(
        &HEAD_ENTRY,
        &call(
            input.clone(),
            Some(ValueFact::scalar(ValueKindFact::Character)),
            Some(LiteralValue::Character("x".into())),
        ),
    );
    assert!(!bad.diagnostics.is_empty());
    let zero = infer_catalog_call(
        &HEAD_ENTRY,
        &call(
            input.clone(),
            Some(ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }))),
            Some(LiteralValue::Number(0.0)),
        ),
    );
    assert!(!zero.diagnostics.is_empty());
    let request = CallRequest {
        arguments: vec![input],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
    };
    assert!(!infer_catalog_call(&HEAD_ENTRY, &request)
        .diagnostics
        .is_empty());
}
